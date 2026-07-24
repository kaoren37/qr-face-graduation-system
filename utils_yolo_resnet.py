import pandas as pd
from ultralytics import YOLO
import torch
from torchvision.models import resnet50
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os


yolo_face = YOLO("models/yolov8n-face.pt")

resnet_model = resnet50(pretrained=True)
resnet_model.eval()
resnet_fc = torch.nn.Sequential(*list(resnet_model.children())[:-1])

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def pil_from_uploaded_file(file):
    if isinstance(file, Image.Image):
        return file
    return Image.open(file).convert("RGB")


def extract_embeddings_yolo(pil_img: Image.Image):
    img_np = np.array(pil_img)

    results = yolo_face(img_np)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    boxes = [b for b, c in zip(boxes, classes) if int(c) == 0]

    if len(boxes) == 0:
        return None, None, "YOLOv8-Face detected no valid face."

    best_box = None
    best_area = 0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None, None, "Invalid face box."

    x1, y1, x2, y2 = best_box
    face_crop = pil_img.crop((x1, y1, x2, y2))

    tensor = resnet_transform(face_crop).unsqueeze(0)

    with torch.no_grad():
        emb = resnet_fc(tensor).squeeze().numpy()
        emb = emb / np.linalg.norm(emb)

    return emb, best_box, None


def recognize_yolo_resnet(image_input, expected_sid, threshold=0.45):
    # ---------------------------------------------------
    # 1) Load image
    # ---------------------------------------------------
    pil_img = pil_from_uploaded_file(image_input)
    img_np = np.array(pil_img)

    # ---------------------------------------------------
    # 2) YOLO face detection
    # ---------------------------------------------------
    results = yolo_face(img_np)
    if len(results[0].boxes) == 0:
        return False, None, pil_img, "No face detected"

    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    box = boxes[np.argmax(areas)]

    x1, y1, x2, y2 = map(int, box)
    face = img_np[y1:y2, x1:x2]
    if face.size == 0:
        return False, None, pil_img, "Invalid face crop"

    # ---------------------------------------------------
    # 3) Extract ResNet embedding
    # ---------------------------------------------------
    face_pil = Image.fromarray(face)
    face_tensor = resnet_transform(face_pil).unsqueeze(0)

    with torch.no_grad():
        emb = resnet_fc(face_tensor).squeeze().numpy()

    emb = emb / (np.linalg.norm(emb) + 1e-10)

    # ---------------------------------------------------
    # 4) Compare with ALL registered embeddings
    # ---------------------------------------------------
    enc_dir = "data/encodings_resnet"

    best_dist = float("inf")
    best_sid = None

    for fname in os.listdir(enc_dir):
        if not fname.endswith(".npy"):
            continue

        sid = fname.replace(".npy", "")
        ref = np.load(os.path.join(enc_dir, fname))
        ref = ref / (np.linalg.norm(ref) + 1e-10)

        dist = 1 - np.dot(emb, ref)

        if dist < best_dist:
            best_dist = dist
            best_sid = sid

    # ---------------------------------------------------
    # 5) FINAL VERIFICATION DECISION
    # ---------------------------------------------------
    is_match = (best_sid == expected_sid) and (best_dist < threshold)

    # ---------------------------------------------------
    # 6) Annotate image
    # ---------------------------------------------------
    annotated = img_np.copy()
    color = (0, 255, 0) if is_match else (0, 0, 255)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    annotated = Image.fromarray(annotated)

    if not is_match:
        return False, best_dist, annotated, "Face does not match expected identity"

    return True, best_dist, annotated, None

def evaluate_yolo_resnet(max_images_per_student=10):
    import matplotlib.pyplot as plt
    df = pd.read_csv("data/students.csv")

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

            img = Image.open(os.path.join(folder, f)).convert("RGB")
            emb, _, _ = extract_embeddings_yolo(img)
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
    ax.axvline(threshold, color="red", linestyle="--")
    ax.legend()
    ax.set_title("YOLO + ResNet Similarity Distribution")

    metrics = {
        "same_mean": same_mean,
        "diff_mean": diff_mean,
        "threshold": threshold,
        "TAR": TAR,
        "FAR": FAR,
        "FRR": FRR
    }

    return metrics, fig
