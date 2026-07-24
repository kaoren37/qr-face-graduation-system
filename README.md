# 🎓 Smart Graduation System — QR & Face Recognition

A Streamlit-based attendance verification platform for graduation ceremonies, combining **QR Code verification** with **AI-powered face recognition** to deliver secure, fast, and reliable attendance tracking.

## Features

- **Student Registration** — register students with ID, name, and programme, upload 1–20 face photos, and auto-generate a unique QR code per student.
- **Model Training Center** — train and evaluate multiple face recognition approaches side by side:
  - Haar Cascade + LBPH
  - SVM + ArcFace + RetinaFace
  - CNN + FaceNet (pretrained, no training required)
  - YOLO + ResNet (pretrained, no training required)
- **Attendance Verification** — two-step flow: scan a student's QR code, then capture a live face photo for AI verification against the registered identity. Matches are recorded automatically with a timestamp.
- **Attendance Records** — view and export attendance history as CSV.

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- OpenCV / opencv-contrib — image processing, Haar Cascade, LBPH
- ONNX Runtime — FaceNet, ArcFace, RetinaFace inference
- PyTorch + Ultralytics YOLOv8 — face detection & ResNet embeddings
- scikit-learn — SVM classifier
- `qrcode` — QR code generation

## Project Structure

```
app.py                     # Home page
pages/
  1_Registration.py        # Student registration + QR generation
  2_Training.py             # Model training & evaluation
  3_Attendance.py            # QR scan + face verification
utils_common.py             # Shared helpers (student data I/O, dirs)
utils_vision.py              # Face preprocessing, QR decode/generate
utils_models.py               # FaceNet, LBPH, SVM+RetinaFace logic
utils_yolo_resnet.py           # YOLO detection + ResNet embeddings
utils_sidebar.py                 # Shared sidebar navigation
styles/styles.css                 # Custom UI styling
models/                             # Pretrained model weights (see Setup)
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/kaoren37/qr-face-graduation-system.git
   cd qr-face-graduation-system
   ```

2. **Install dependencies** (Python 3.10 recommended)
   ```bash
   pip install -r requirements.txt
   ```

3. **Model weights** — most weights are included in `models/`. `models/retinaface.onnx` is excluded from this repo (exceeds GitHub's file size limit) — download it separately and place it in `models/` before using the RetinaFace-based training/verification options.

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Register** a student with their details and face photos.
2. **Train** the LBPH and SVM+RetinaFace models once enough students are registered (FaceNet and YOLO+ResNet work out of the box with pretrained weights).
3. **Verify** attendance by scanning a student's QR code, then capturing their face for AI matching.

## License

For academic / educational use.
