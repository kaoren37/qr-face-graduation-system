import os
import pandas as pd

DATA_DIR = "data"
FACES_DIR = os.path.join(DATA_DIR, "faces")
QR_DIR = os.path.join(DATA_DIR, "qr_codes")
STUDENTS_CSV = os.path.join(DATA_DIR, "students.csv")

PREPROCESS_BASE = "data/preprocessed"
os.makedirs(PREPROCESS_BASE, exist_ok=True)

def init_dirs():
    os.makedirs(FACES_DIR, exist_ok=True)
    os.makedirs(QR_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(STUDENTS_CSV):
        df = pd.DataFrame(columns=["student_id","name","programme","face_path","qr_path"])
        df.to_csv(STUDENTS_CSV, index=False)

def load_students():
    init_dirs()
    return pd.read_csv(STUDENTS_CSV)

def save_students(df):
    df.to_csv(STUDENTS_CSV, index=False)
