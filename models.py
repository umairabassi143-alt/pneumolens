"""
Database models for the multi-user authentication system.

Two tables:
- User: stores account info, role (patient/doctor/admin), and login activity.
  Passwords are NEVER stored in plain text - only a secure hash.
- Analysis: stores each prediction result, linked to the user who ran it.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    # One of: "patient", "doctor", "admin". Never settable from the
    # frontend - only assigned at registration (patient) or by an admin
    # (doctor/admin), and always re-checked server-side on every
    # protected route via the role_required decorator.
    role = db.Column(db.String(20), nullable=False, default="patient")

    account_created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    is_active_account = db.Column(db.Boolean, default=True)

    analyses = db.relationship(
        "Analysis", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    # Flask-Login checks `is_active` to block disabled accounts from
    # logging in or staying logged in. We back it with our own DB column
    # (is_active_account) so an admin can deactivate an account.
    @property
    def is_active(self):
        return self.is_active_account

    def __repr__(self):
        return f"<User {self.email} ({self.role})>"


class Analysis(db.Model):
    __tablename__ = "analyses"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    original_image_path = db.Column(db.String(255), nullable=False)
    gradcam_image_path = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Analysis {self.id} user={self.user_id} {self.prediction}>"
