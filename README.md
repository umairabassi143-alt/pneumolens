# PneumoLens — AI-Powered Pneumonia Classification
MSc Artificial Intelligence dissertation project (University of South Wales).
Detects pneumonia from chest X-rays using YOLO26 and provides Grad-CAM 
explainability, deployed as a role-based Flask web application.
## Features
- YOLO26n-cls (3-class: NORMAL/PNEUMONIA/OTHER) classification model
- Grad-CAM++ explainability with quantitative validation
- Role-based access (Patient/Doctor/Admin)
- NHS-guidance-based patient safety information
- ## Dataset
This project uses the Kermany et al. (2018) paediatric chest X-ray dataset,
accessed via a Roboflow-hosted redistribution (version 4, by Mohamed Traore).
Available at:
https://universe.roboflow.com/mohamed-traore-2ekkp/chest-x-rays-qjmia/dataset/4
## Model Weights
Model weights (best.pt) are not included in this repository due to size.
Download from: [https://drive.google.com/file/d/1wPgupfxbDlZh_sPFWoOeczuBz7F6e_Zo/view?usp=drive_link]
## Setup
1. `pip install -r requirements.txt`
2. Place `best.pt` in `model/` folder
3. `python app.py`
## Note
The `static/uploads`, `static/results`, and `instance` folders are created
automatically by the application at runtime and are not included in this
repository.

## Author
Umair Zahoor, Student ID 30144981, MSc AI, University of South Wales
