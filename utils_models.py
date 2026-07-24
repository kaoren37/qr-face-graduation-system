import os
import cv2
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import io
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import onnxruntime as ort
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from utils_vision import preprocess_face

import onnxruntime as ort

FACENET_MODEL = "models/faceNet.onnx"

facenet_sess = ort.InferenceSession(
    FACENET_MODEL,
    providers=["CPUExecutionProvider"]
)
facenet_input_name = facenet_sess.get_inputs()[0].name


from utils_common import FACES_DIR, STUDENTS_CSV
from utils_vision import (
    detect_face_strong,
    detect_and_align,
    detect_face_and_landmarks,
    align_facenet_from_landmarks
)

def _uploaded_to_pil(image_input):
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if hasattr(image_input, "getvalue"):
        bytes_data = image_input.getvalue()
    else:
        bytes_data = image_input

    return Image.open(io.BytesIO(bytes_data)).convert("RGB")


# ============================================================
# LBPH MODEL
# ============================================================
LBPH_PATH = os.path.join("models", "lbph_model.xml")

def train_lbph():
    df = pd.read_csv(STUDENTS_CSV)
    if df.empty:
        return False, "No students.", None

    faces = []
    labels = []
    label_map = {}

    for idx, row in df.iterrows():
        sid = row["student_id"]
        if sid not in label_map.values():
            label_map[len(label_map)] = sid

    for label, sid in label_map.items():
        folder = df[df["student_id"] == sid]["face_path"].iloc[0]
        if not os.path.isdir(folder):
            continue

        for f in os.listdir(folder):
            if f.endswith(".png"):
                img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                labels.append(label)

    faces = np.array(faces)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        faces, labels, test_size=0.2, random_state=42, stratify=labels
    )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(X_train, y_train)
    recognizer.write(LBPH_PATH)

    with open("models/lbph_labels.json", "w") as fp:
        json.dump(label_map, fp)

    predictions = []
    for img in X_test:
        pred_label, conf = recognizer.predict(img)
        predictions.append(pred_label)

    accuracy = np.mean(np.array(predictions) == y_test)

    cm = confusion_matrix(y_test, predictions)
    cm_df = pd.DataFrame(cm)

    metrics = {
        "accuracy": float(accuracy),
        "confusion_matrix": cm_df
    }

    return True, "LBPH Training Complete.", metrics




def recognize_with_lbph(image_input, draw_box=False, display_name=None):
    if not os.path.exists(LBPH_PATH):
        return None, "LBPH not trained.", None

    with open("models/lbph_labels.json") as fp:
        label_map = {int(k): v for k, v in json.load(fp).items()}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(LBPH_PATH)

    pil = _uploaded_to_pil(image_input)
    np_img = np.array(pil)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    faces = detect_face_strong(gray)
    if len(faces) == 0:
        return None, "No face detected.", None

    x, y, w, h = faces[0]
    face_crop = gray[y:y+h, x:x+w]

    face_resized = cv2.resize(face_crop, (200, 200))
    label, conf = recognizer.predict(face_resized)
    sid = label_map[label]

    annotated = None
    if draw_box:
        img = np_img.copy()
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        annotated = Image.fromarray(img)

    return (sid, conf), None, annotated


# ============================================================
# FACENET
# ============================================================
facenet_sess = ort.InferenceSession(
    "models/facenet.onnx",  
    providers=["CPUExecutionProvider"]
)
def facenet_preprocess(img_rgb):
    """
    Preprocess for FaceNet ONNX (NHWC: 1x160x160x3)
    """
    img = cv2.resize(img_rgb, (160, 160))
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0        
    img = np.expand_dims(img, axis=0)  
    return img



def facenet_get_embedding(image_input, use_alignment=True):

    pil = _uploaded_to_pil(image_input)
    np_img = np.array(pil)
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    box, landmarks, _ = detect_face_and_landmarks(bgr)
    if box is None:
        return None, "Face not detected", None

    x1, y1, x2, y2 = map(int, box)
    h, w, _ = bgr.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face_crop = bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, "Invalid face crop", None

    if use_alignment and landmarks is not None and landmarks.shape == (5, 2):
        landmarks_crop = landmarks.copy()
        landmarks_crop[:, 0] -= x1
        landmarks_crop[:, 1] -= y1

        try:
            aligned_bgr = align_facenet_from_landmarks(face_crop, landmarks_crop)
        except Exception as e:
            return None, f"Alignment error: {e}", None
    else:
        aligned_bgr = cv2.resize(face_crop, (160, 160))

    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)

    hist_eq = cv2.equalizeHist(gray)

    aligned_rgb = cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2RGB)

    tensor = facenet_preprocess(aligned_rgb) 

    try:
        emb = facenet_sess.run(None, {facenet_input_name: tensor})[0]
        emb = emb.reshape(-1)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
    except Exception as e:
        return None, f"Embedding error: {e}", None

    annotated = np_img.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    annotated = Image.fromarray(annotated)

    return emb, None, annotated


def save_facenet_embedding(student_id, emb):
    os.makedirs("data/encodings_facenet", exist_ok=True)
    np.save(f"data/encodings_facenet/{student_id}.npy", emb)


def recognize_with_facenet(image_input, expected_sid, threshold=0.60):
    emb_live, err, annotated = facenet_get_embedding(image_input)
    if err:
        return None, err, None, None

    emb_path = f"data/encodings_facenet/{expected_sid}.npy"
    if not os.path.exists(emb_path):
        return None, f"No saved embedding for {expected_sid}", annotated, None

    emb_saved = np.load(emb_path)
    emb_saved /= np.linalg.norm(emb_saved)

    dist = 1 - np.dot(emb_live, emb_saved)

    is_match = dist < threshold

    return dist, None, annotated, is_match



# ============================================================
# RETINAFACE + ARCFACE + SVM
# ============================================================
arcface_sess = ort.InferenceSession(
    "models/w600k_mbf.onnx",
    providers=["CPUExecutionProvider"]
)

def get_arcface_embedding(aligned):
    img = aligned.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2,0,1))[None,...]

    input_name = arcface_sess.get_inputs()[0].name
    emb = arcface_sess.run(None, {input_name: img})[0]
    emb = emb.reshape(-1)
    emb = emb / np.linalg.norm(emb)
    return emb


def train_svm_retinaface(students_df):
    """
    Train and evaluate SVM using identity-based splitting
    with MULTIPLE test images per person (10 images),
    to obtain realistic FP/FN and non-perfect metrics.
    """

    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.svm import SVC
    from collections import defaultdict
    import os, cv2, joblib, random
    import numpy as np
    import pandas as pd

    TEST_PER_PERSON = 10  

    image_paths = []
    labels = []

    for _, row in students_df.iterrows():
        sid = str(row["student_id"])
        folder = row["face_path"]

        if not os.path.exists(folder):
            continue

        for f in os.listdir(folder):
            if f.lower().endswith(".png"):
                image_paths.append(os.path.join(folder, f))
                labels.append(sid)

    person_images = defaultdict(list)
    for path, label in zip(image_paths, labels):
        person_images[label].append(path)

    train_paths, test_paths = [], []
    y_train, y_test = [], []

    for label, paths in person_images.items():
        if len(paths) <= TEST_PER_PERSON:
            continue  

        random.shuffle(paths) 

        test_subset = paths[:TEST_PER_PERSON]
        train_subset = paths[TEST_PER_PERSON:]

        for p in test_subset:
            test_paths.append(p)
            y_test.append(label)

        for p in train_subset:
            train_paths.append(p)
            y_train.append(label)

    X_train_emb, y_train_clean = [], []

    for path, label in zip(train_paths, y_train):
        img = cv2.imread(path)
        if img is None:
            continue

        aligned, _ = detect_and_align(img)
        if aligned is None:
            continue

        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        aligned = cv2.resize(aligned, (112, 112))

        emb = get_arcface_embedding(aligned)
        if emb is None:
            continue

        X_train_emb.append(emb)
        y_train_clean.append(label)

    X_test_emb, y_test_clean = [], []

    for path, label in zip(test_paths, y_test):
        img = cv2.imread(path)
        if img is None:
            continue

        aligned, _ = detect_and_align(img)
        if aligned is None:
            continue

        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        aligned = cv2.resize(aligned, (112, 112))

        emb = get_arcface_embedding(aligned)
        if emb is None:
            continue

        X_test_emb.append(emb)
        y_test_clean.append(label)

    if len(set(y_train_clean)) < 2 or len(set(y_test_clean)) < 2:
        return False, "Not enough valid samples for SVM evaluation.", None

    X_train_emb = np.array(X_train_emb)
    X_test_emb = np.array(X_test_emb)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_clean)
    y_test_enc = le.transform(y_test_clean)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_emb)
    X_test_scaled = scaler.transform(X_test_emb)

    svm = SVC(kernel="linear", probability=True, C=5)
    svm.fit(X_train_scaled, y_train_enc)

    joblib.dump(svm, "models/svm_retina.pkl")
    joblib.dump(scaler, "models/svm_scaler.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    preds = svm.predict(X_test_scaled)

    accuracy = np.mean(preds == y_test_enc)
    precision = precision_score(y_test_enc, preds, average="weighted", zero_division=0)
    recall = recall_score(y_test_enc, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test_enc, preds, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test_enc, preds)
    cm_df = pd.DataFrame(cm)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm_df,
        "num_train": len(y_train_enc),
        "num_test": len(y_test_enc),
        "test_per_person": TEST_PER_PERSON,
        "evaluation_method": "Identity-based (10 images per person)"
    }

    return True, "SVM Training Complete (Identity-Based Evaluation).", metrics

    


def svm_predict_retinaface(bgr):
    if not os.path.exists("models/svm_retina.pkl"):
        return None, "SVM model not trained", None

    aligned, box = detect_and_align(bgr)
    if aligned is None:
        return None, "No face detected", None

    processed = preprocess_face(aligned) 

    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    processed = cv2.resize(processed, (112, 112))

    emb = get_arcface_embedding(processed)

    svm = joblib.load("models/svm_retina.pkl")
    scaler = joblib.load("models/svm_scaler.pkl")
    le = joblib.load("models/label_encoder.pkl")

    emb_scaled = scaler.transform([emb])
    pred_num = svm.predict(emb_scaled)[0]
    prob = np.max(svm.predict_proba(emb_scaled))

    sid = le.inverse_transform([pred_num])[0]

    if box is not None:
        box = list(map(int, box))

    return sid, prob, box

def evaluate_facenet_pretrained(max_images_per_student=10):
    import matplotlib.pyplot as plt

    df = pd.read_csv(STUDENTS_CSV)
    data = []  

    for _, row in df.iterrows():
        sid = row["student_id"]
        folder = row["face_path"]

        count = 0
        for f in os.listdir(folder):
            if not f.lower().endswith(".png"):
                continue
            if count >= max_images_per_student:
                break

            img = cv2.imread(os.path.join(folder, f))
            if img is None:
                continue

            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            emb, _, _ = facenet_get_embedding(pil_img)
            if emb is not None:
                data.append((sid, emb))
                count += 1

    same_scores = []
    diff_scores = []

    for i in range(len(data)):
        sid1, emb1 = data[i]
        for j in range(i+1, len(data)):
            sid2, emb2 = data[j]

            sim = float(np.dot(emb1, emb2))

            if sid1 == sid2:
                same_scores.append(sim)
            else:
                diff_scores.append(sim)

    same_mean = np.mean(same_scores)
    diff_mean = np.mean(diff_scores)
    threshold = (same_mean + diff_mean) / 2

    TAR = np.mean(np.array(same_scores) >= threshold)
    FAR = np.mean(np.array(diff_scores) >= threshold)
    FRR = 1 - TAR

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(same_scores, bins=20, alpha=0.6, label="Same Person")
    ax.hist(diff_scores, bins=20, alpha=0.6, label="Different Persons")
    ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.2f}")
    ax.legend()
    ax.set_title("FaceNet Similarity Distribution")

    return {
        "same_mean": same_mean,
        "diff_mean": diff_mean,
        "threshold": threshold,
        "TAR": TAR,
        "FAR": FAR,
        "FRR": FRR
    }, fig

