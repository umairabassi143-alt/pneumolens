"""
Pneumonia Detection - Flask Web Application (Small Classifier)
------------------------------------------------------------------
Upload a chest X-ray -> get prediction + confidence + Grad-CAM overlay + PDF report.

Run as:
    python app.py

Then open: http://127.0.0.1:5000 in your browser.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, render_template, send_file, redirect, url_for, flash, abort
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import uuid

from models import db, User, Analysis
from decorators import role_required

app = Flask(__name__)

# ---------------------------------------------------------
# Security / session configuration
# ---------------------------------------------------------
# SECRET_KEY signs session cookies - required for Flask-Login sessions to
# be secure. In a real production deployment this MUST come from an
# environment variable, never hardcoded - see the setup notes at the end
# of this file for how to override it.
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///pneumonia_app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
MODEL_PATH = "model/best.pt"

# Below this confidence, the model's prediction is not trusted enough to
# present as a diagnosis — this catches cases where the model itself is
# unsure between the known categories.
CONFIDENCE_THRESHOLD = 0.60

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------------------------------------------
# Create database tables on first run (safe to call every startup -
# it only creates tables that don't already exist, never drops or
# overwrites existing data).
#
# Bootstrap problem: the very first admin account can't be created
# through the normal registration flow (which only creates "patient"
# accounts, for security - see the /register route below). So if the
# database is completely empty, we create one default admin account
# here and print its credentials to the console. Change this password
# immediately after first login in any real deployment.
# ---------------------------------------------------------
with app.app_context():
    db.create_all()
    if User.query.count() == 0:
        default_admin = User(
            name="Default Admin",
            email="admin@pneumolens.local",
            password_hash=generate_password_hash("ChangeMe123!"),
            role="admin",
        )
        db.session.add(default_admin)
        db.session.commit()
        print("=" * 60)
        print("No users found - created a default admin account:")
        print("  Email:    admin@pneumolens.local")
        print("  Password: ChangeMe123!")
        print("Log in and change this password / create real accounts.")
        print("=" * 60)

# ---------------------------------------------------------
# Load model once when the app starts
# ---------------------------------------------------------
print("Loading classification model...")
yolo_model = YOLO(MODEL_PATH)
raw_model = yolo_model.model
raw_model.eval()


class YOLOWrapper(nn.Module):
    """Wraps YOLO so it returns a single tensor instead of a tuple (needed for Grad-CAM)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        return outputs


cam_model = YOLOWrapper(raw_model)
cam_model.eval()
if torch.cuda.is_available():
    cam_model = cam_model.cuda()

target_layers = [raw_model.model[-2]]
# GradCAM++ replaces EigenCAM: EigenCAM does not use gradients or a class
# target at all (it computes the principal activation pattern via PCA),
# so it could never explain "why THIS class was predicted" -- only "what
# pattern dominates this layer's activations", which may or may not align
# with the predicted class. GradCAM++ is gradient-based and genuinely
# class-discriminative when given a specific ClassifierOutputTarget below.
cam = GradCAMPlusPlus(cam_model, target_layers)

print("Model loaded successfully.")

# ---------------------------------------------------------
# Core function: prediction + Grad-CAM for any image
# ---------------------------------------------------------


def crop_black_borders(img, threshold=15):
    """
    Crops away near-black borders from an X-ray image before it is fed to
    the model or Grad-CAM. Some source images have black margins baked
    into the file itself (e.g. from how they were originally captured or
    scanned). When resized to a square, these margins remain as real
    pixel content and create a sharp black-to-white edge - and
    gradient-based explainability methods like Grad-CAM are known to
    react strongly to high-contrast edges regardless of whether they are
    clinically meaningful. Cropping to the actual image content first
    removes this confound before resizing.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    col_means = gray.mean(axis=0)
    row_means = gray.mean(axis=1)

    cols = np.where(col_means > threshold)[0]
    rows = np.where(row_means > threshold)[0]

    if len(cols) == 0 or len(rows) == 0:
        return img  # nothing crossed the threshold - keep original as a fallback

    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1

    h, w = gray.shape
    # Safety check: never crop away more than half of either dimension in
    # one go, so a genuinely dark (but valid) X-ray isn't over-cropped
    if (x1 - x0) < w * 0.5 or (y1 - y0) < h * 0.5:
        return img

    return img[y0:y1, x0:x1]


def predict_and_explain(img_path, save_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_black_borders(img)
    img = cv2.resize(img, (416, 416))
    rgb_img = img.astype(np.float32) / 255.0

    input_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float()
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # Prediction uses Ultralytics' own preprocessing pipeline via
    # yolo_model.predict() - this is the pipeline the model was actually
    # trained and validated with, and is known to produce correct,
    # varied confidence scores. An earlier attempt to derive the
    # prediction from the same manually-built tensor used for Grad-CAM
    # broke this: the manual resize+normalize pipeline did not match
    # Ultralytics' internal preprocessing closely enough, causing
    # near-constant, unreliable confidence values regardless of input.
    # Grad-CAM still uses the manual tensor below (a minor, acceptable
    # preprocessing difference), but prediction correctness takes
    # priority and is restored here.
    results = yolo_model.predict(img_path, verbose=False)
    result = results[0]
    prediction = result.names[result.probs.top1]
    confidence = result.probs.top1conf.item()
    top1_idx = int(result.probs.top1)

    # Grad-CAM++, targeted specifically at the predicted class. This is
    # the class-discriminative signal that answers "why THIS class" -
    # unlike EigenCAM (which ignored the target) or a summed target
    # (which blended all three classes together).
    #
    # Ultralytics loads model weights with requires_grad=False on every
    # parameter by default (an inference-only optimisation), so there is
    # normally no computational graph to backpropagate through. EigenCAM
    # never hit this because it is gradient-free, but GradCAM++ needs a
    # real backward pass. Explicitly enabling requires_grad on the input
    # tensor forces PyTorch to build a graph along this forward pass
    # regardless of the frozen model parameters, which is enough for
    # GradCAM++'s backward() call to work correctly.
    input_tensor.requires_grad_(True)
    grad_target = [ClassifierOutputTarget(top1_idx)]
    grayscale_cam = cam(input_tensor, targets=grad_target)[0, :, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, cam_image_bgr)

    return prediction, confidence


# ---------------------------------------------------------
# Authentication routes
# ---------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # --- Input validation ---
        errors = []
        if not name:
            errors.append("Name is required.")
        if not email or "@" not in email:
            errors.append("A valid email is required.")
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long.")
        if password != confirm_password:
            errors.append("Passwords do not match.")
        if User.query.filter_by(email=email).first():
            errors.append("An account with this email already exists.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("register.html", name=name, email=email)

        # Public registration ONLY ever creates "patient" role accounts.
        # Doctor and admin accounts can only be created by an existing
        # admin (see /admin/users/new below) - this is enforced here,
        # server-side, regardless of anything sent in the request.
        new_user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            role="patient",
        )
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = bool(request.form.get("remember"))

        user = User.query.filter_by(email=email).first()

        # Deliberately generic error message for both "no such user" and
        # "wrong password" - this avoids leaking which emails are
        # registered (a common security best practice).
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "error")
            return render_template("login.html", email=email)

        if not user.is_active_account:
            flash("This account has been deactivated. Contact an administrator.", "error")
            return render_template("login.html", email=email)

        login_user(user, remember=remember)
        user.last_login = datetime.utcnow()
        db.session.commit()

        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


# ---------------------------------------------------------
# Dashboard routes (role-aware)
# ---------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    if current_user.role == "admin":
        return redirect(url_for("admin_dashboard"))
    if current_user.role == "doctor":
        return redirect(url_for("doctor_dashboard"))
    return redirect(url_for("patient_dashboard"))


@app.route("/dashboard/patient")
@login_required
@role_required("patient")
def patient_dashboard():
    recent = (
        Analysis.query.filter_by(user_id=current_user.id)
        .order_by(Analysis.created_at.desc())
        .limit(5)
        .all()
    )
    total_count = Analysis.query.filter_by(user_id=current_user.id).count()
    return render_template(
        "dashboard_user.html",
        recent_analyses=recent,
        total_count=total_count,
    )


@app.route("/dashboard/doctor")
@login_required
@role_required("doctor", "admin")
def doctor_dashboard():
    patients = User.query.filter_by(role="patient").all()
    total_analyses = Analysis.query.count()
    return render_template(
        "dashboard_doctor.html",
        patients=patients,
        total_analyses=total_analyses,
    )


@app.route("/dashboard/admin")
@login_required
@role_required("admin")
def admin_dashboard():
    total_users = User.query.filter_by(role="patient").count()
    total_doctors = User.query.filter_by(role="doctor").count()
    total_admins = User.query.filter_by(role="admin").count()
    total_analyses = Analysis.query.count()
    recent_users = User.query.order_by(User.account_created_at.desc()).limit(8).all()
    return render_template(
        "dashboard_admin.html",
        total_users=total_users,
        total_doctors=total_doctors,
        total_admins=total_admins,
        total_analyses=total_analyses,
        recent_users=recent_users,
    )


# ---------------------------------------------------------
# Profile and history
# ---------------------------------------------------------
@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html")


@app.route("/history")
@login_required
def history():
    if current_user.role == "patient":
        analyses = (
            Analysis.query.filter_by(user_id=current_user.id)
            .order_by(Analysis.created_at.desc())
            .all()
        )
    else:
        # Doctors and admins can review analyses across all patients.
        analyses = Analysis.query.order_by(Analysis.created_at.desc()).all()
    return render_template("history.html", analyses=analyses)


@app.route("/analysis/<int:analysis_id>")
@login_required
def view_analysis(analysis_id):
    analysis = db.session.get(Analysis, analysis_id)
    if analysis is None:
        abort(404)

    # A patient may only open their OWN analyses. Doctors/admins may open
    # any analysis. This check happens server-side and cannot be bypassed
    # by guessing a different URL.
    if current_user.role == "patient" and analysis.user_id != current_user.id:
        abort(403)

    is_uncertain = analysis.confidence < CONFIDENCE_THRESHOLD
    not_xray_like = analysis.prediction == "OTHER"

    return render_template(
        "result.html",
        prediction=analysis.prediction,
        confidence=f"{analysis.confidence:.1%}",
        is_uncertain=is_uncertain,
        not_xray_like=not_xray_like,
        original_image=analysis.original_image_path,
        gradcam_image=analysis.gradcam_image_path,
        report_id=analysis.id,
    )


# ---------------------------------------------------------
# Admin: user management
# ---------------------------------------------------------
@app.route("/admin/users")
@login_required
@role_required("admin")
def admin_users():
    search = request.args.get("q", "").strip()
    query = User.query
    if search:
        query = query.filter(
            db.or_(User.name.ilike(f"%{search}%"), User.email.ilike(f"%{search}%"))
        )
    users = query.order_by(User.account_created_at.desc()).all()
    return render_template("admin_users.html", users=users, search=search)


@app.route("/admin/users/new", methods=["GET", "POST"])
@login_required
@role_required("admin")
def admin_create_user():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "patient")

        # Only these three roles are ever valid, regardless of what is
        # submitted in the form - this is the server-side check that
        # prevents privilege escalation via a tampered request.
        if role not in ("patient", "doctor", "admin"):
            role = "patient"

        errors = []
        if not name:
            errors.append("Name is required.")
        if not email or "@" not in email:
            errors.append("A valid email is required.")
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long.")
        if User.query.filter_by(email=email).first():
            errors.append("An account with this email already exists.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("admin_create_user.html", name=name, email=email, role=role)

        new_user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            role=role,
        )
        db.session.add(new_user)
        db.session.commit()
        flash(f"{role.capitalize()} account created for {email}.", "success")
        return redirect(url_for("admin_users"))

    return render_template("admin_create_user.html")


@app.route("/admin/users/<int:user_id>/toggle", methods=["POST"])
@login_required
@role_required("admin")
def admin_toggle_user(user_id):
    user = db.session.get(User, user_id)
    if user is None:
        abort(404)
    if user.id == current_user.id:
        flash("You cannot deactivate your own account.", "error")
        return redirect(url_for("admin_users"))
    user.is_active_account = not user.is_active_account
    db.session.commit()
    status = "activated" if user.is_active_account else "deactivated"
    flash(f"{user.email} has been {status}.", "success")
    return redirect(url_for("admin_users"))


@app.route("/doctor/patient/<int:patient_id>")
@login_required
@role_required("doctor", "admin")
def doctor_view_patient(patient_id):
    patient = db.session.get(User, patient_id)
    if patient is None or patient.role != "patient":
        abort(404)
    analyses = (
        Analysis.query.filter_by(user_id=patient.id)
        .order_by(Analysis.created_at.desc())
        .all()
    )
    return render_template("patient_history.html", patient=patient, analyses=analyses)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
@login_required
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    unique_id = uuid.uuid4().hex[:8]
    original_filename = f"{unique_id}_original.jpg"
    gradcam_filename = f"{unique_id}_gradcam.jpg"

    original_path = os.path.join(UPLOAD_FOLDER, original_filename)
    gradcam_path = os.path.join(RESULTS_FOLDER, gradcam_filename)

    file.save(original_path)

    prediction, confidence = predict_and_explain(original_path, gradcam_path)

    is_uncertain = confidence < CONFIDENCE_THRESHOLD
    not_xray_like = prediction == "OTHER"

    # Save this result against the logged-in user's account, so it shows
    # up in their History and the Doctor/Admin dashboards.
    analysis = Analysis(
        user_id=current_user.id,
        original_image_path=original_path,
        gradcam_image_path=gradcam_path,
        prediction=prediction,
        confidence=confidence,
    )
    db.session.add(analysis)
    db.session.commit()

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=f"{confidence:.1%}",
        is_uncertain=is_uncertain,
        not_xray_like=not_xray_like,
        original_image=original_path,
        gradcam_image=gradcam_path,
        report_id=analysis.id,
    )


@app.route("/download_report/<int:report_id>")
@login_required
def download_report(report_id):
    analysis = db.session.get(Analysis, report_id)
    if analysis is None:
        abort(404)

    # Same ownership check as view_analysis: patients can only download
    # their own reports.
    if current_user.role == "patient" and analysis.user_id != current_user.id:
        abort(403)

    original_path = analysis.original_image_path
    gradcam_path = analysis.gradcam_image_path
    prediction = analysis.prediction
    confidence = analysis.confidence
    pdf_path = os.path.join(RESULTS_FOLDER, f"{analysis.id}_report.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Pneumonia Detection Report")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    c.drawString(50, height - 98, f"Patient/User: {analysis.user.name}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 130, f"Prediction: {prediction}")
    c.drawString(50, height - 155, f"Confidence: {confidence:.1%}")

    if confidence < CONFIDENCE_THRESHOLD:
        c.setFillColorRGB(0.85, 0.25, 0.25)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, height - 178, "LOW CONFIDENCE — this image may not be a valid chest X-ray. Result unreliable.")
        c.setFillColorRGB(0, 0, 0)

    if current_user.role in ("doctor", "admin"):
        # Full clinical view: original X-ray + Grad-CAM explainability side by side
        c.drawImage(original_path, 50, height - 430, width=220, height=220)
        c.drawImage(gradcam_path, 300, height - 430, width=220, height=220)

        c.setFont("Helvetica-Oblique", 9)
        c.drawString(50, height - 460, "Left: Original X-ray   |   Right: Grad-CAM explainability overlay")
    else:
        # Patient-facing report: original X-ray only - no Grad-CAM/model
        # attention visualisation, consistent with the web result page.
        c.drawImage(original_path, 160, height - 460, width=280, height=280)

        c.setFont("Helvetica-Oblique", 9)
        c.drawString(50, height - 475, "Your uploaded X-ray")

    c.setFont("Helvetica", 9)
    c.drawString(50, 60, "This report is generated by an AI decision-support tool and is not a substitute")
    c.drawString(50, 48, "for clinical judgement. Please consult a qualified radiologist or physician.")

    c.save()

    return send_file(pdf_path, as_attachment=True, download_name="pneumonia_report.pdf")


@app.errorhandler(401)
def unauthorized(e):
    flash("Please log in to access this page.", "error")
    return redirect(url_for("login"))


@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", code=403, message="You don't have permission to access this page."), 403


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="That page or record could not be found."), 404


if __name__ == "__main__":
    # host="0.0.0.0" makes the app reachable from other devices on the
    # same WiFi network (e.g. a phone), not just this laptop. See the
    # setup notes for how to find the correct address to type on mobile.
    app.run(debug=True, host="0.0.0.0", port=5000)
