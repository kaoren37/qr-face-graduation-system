import cv2
import numpy as np
from PIL import Image
import qrcode
import os
import io
import onnxruntime as ort
from datetime import datetime
from utils_common import FACES_DIR, QR_DIR, PREPROCESS_BASE


# ============================================================
#                   RETINAFACE + ARCface
# ============================================================

REFERENCE_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_facenet_from_landmarks(face_bgr, landmarks_crop):
    M = cv2.estimateAffinePartial2D(landmarks_crop, REFERENCE_5PTS)[0]
    aligned = cv2.warpAffine(face_bgr, M, (160, 160))  
    return aligned


retina_sess = ort.InferenceSession(
    "models/retinaface.onnx",
    providers=["CPUExecutionProvider"]
)

VARIANCE = [0.1, 0.2]

eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

def estimate_5_landmarks(face_bgr):
    """
    Estimate 5-point landmarks:
    left eye, right eye, nose (center), mouth left, mouth right
    using only Haar detectors (NO DLIB).
    Works well enough for FaceNet alignment.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    eyes = eye_detector.detectMultiScale(gray, 1.1, 4)
    if len(eyes) < 2:
        return None

    eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]

    (x1, y1, w1, h1) = eyes[0]
    (x2, y2, w2, h2) = eyes[1]

    eye1 = (x1 + w1//2, y1 + h1//2)
    eye2 = (x2 + w2//2, y2 + h2//2)

    left_eye, right_eye = sorted([eye1, eye2], key=lambda p: p[0])

    nose = ((left_eye[0] + right_eye[0])//2,
            int((left_eye[1] + right_eye[1])//2 + (face_bgr.shape[0] * 0.18)))

    mouth = mouth_detector.detectMultiScale(gray, 1.2, 20)
    if len(mouth) > 0:
        (mx, my, mw, mh) = max(mouth, key=lambda m: m[2]*m[3])
        mouth_left  = (mx, my + mh//2)
        mouth_right = (mx + mw, my + mh//2)
    else:
        mouth_left  = (nose[0] - 20, nose[1] + 40)
        mouth_right = (nose[0] + 20, nose[1] + 40)

    return np.array([left_eye, right_eye, nose, mouth_left, mouth_right], dtype=np.float32)


def preprocess_retina(img_bgr_640: np.ndarray) -> np.ndarray:
    img_float = img_bgr_640.astype(np.float32)
    img_float -= np.array([104, 117, 123], dtype=np.float32)
    blob = np.transpose(img_float, (2, 0, 1))[None, ...]
    return blob


def morphological_face_cleanup(gray):
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def generate_anchors():
    feature_maps = [[80, 80], [40, 40], [20, 20]]
    steps = [8, 16, 32]
    min_sizes = [[16, 32], [64, 128], [256, 512]]

    anchors = []
    for k, f in enumerate(feature_maps):
        for i in range(f[0]):
            for j in range(f[1]):
                for ms in min_sizes[k]:
                    s_kx = ms / 640.0
                    s_ky = ms / 640.0
                    cx = (j + 0.5) * steps[k] / 640.0
                    cy = (i + 0.5) * steps[k] / 640.0
                    anchors.append([cx, cy, s_kx, s_ky])
    return np.array(anchors, dtype=np.float32)


anchors = generate_anchors()


def decode_boxes(loc, priors, variance=VARIANCE):
    boxes = np.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variance[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variance[1]),
        ),
        axis=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landmarks(pre, priors, variance=VARIANCE):
    landms = np.concatenate(
        (
            priors[:, :2] + pre[:, 0:2] * variance[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variance[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variance[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variance[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variance[0] * priors[:, 2:],
        ),
        axis=1,
    )
    return landms


def decode_retina(outputs, conf_thres=0.6):
    loc = outputs[0][0]
    conf = outputs[1][0]
    landm = outputs[2][0]

    scores = conf[:, 1]
    boxes = decode_boxes(loc, anchors)
    landms = decode_landmarks(landm, anchors)

    faces = []
    for i, score in enumerate(scores):
        if score < conf_thres:
            continue

        x1, y1, x2, y2 = boxes[i] * 640.0
        lm = landms[i].reshape(5, 2) * 640.0

        faces.append({
            "box": [x1, y1, x2, y2],
            "landmarks": lm,
            "score": float(score),
        })

    return faces

def detect_face_and_landmarks(img_bgr):
    """
    Detect a face and 5 landmarks using RetinaFace.
    Returns:
        box  -> [x1, y1, x2, y2]
        lmk  -> 5x2 array of landmarks
        img  -> original image (BGR)
    """
    h, w = img_bgr.shape[:2]

    resized = cv2.resize(img_bgr, (640, 640))
    blob = preprocess_retina(resized)

    input_name = retina_sess.get_inputs()[0].name
    outputs = retina_sess.run(None, {input_name: blob})

    faces = decode_retina(outputs)
    if len(faces) == 0:
        return None, None, None

    face = max(faces, key=lambda x: x['score'])

    sx = w / 640
    sy = h / 640

    x1, y1, x2, y2 = face['box']
    box = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]

    lmk = face['landmarks'].astype(np.float32)
    lmk[:, 0] *= sx
    lmk[:, 1] *= sy

    return box, lmk, img_bgr


def align_face_arcface(img_bgr_640, lmk):
    template = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    lmk = np.array(lmk, dtype=np.float32)
    M = cv2.estimateAffinePartial2D(lmk, template)[0]
    aligned = cv2.warpAffine(img_bgr_640, M, (112, 112))
    return aligned


def detect_and_align(img_bgr):
    h, w = img_bgr.shape[:2]

    img_640 = cv2.resize(img_bgr, (640, 640))
    blob = preprocess_retina(img_640)
    input_name = retina_sess.get_inputs()[0].name
    outputs = retina_sess.run(None, {input_name: blob})

    faces = decode_retina(outputs)

    if len(faces) == 0:
        return None, None

    best = max(faces, key=lambda x: x["score"])
    box_640 = best["box"]
    lmk_640 = best["landmarks"]

    aligned = align_face_arcface(img_640, lmk_640)

    sx = w / 640.0
    sy = h / 640.0

    x1, y1, x2, y2 = box_640
    box_orig = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]

    return aligned, box_orig


# ============================================================
#                   HAAR / LBPH PREPROCESSING
# ============================================================

haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face_strong(gray):
    faces = haar.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        return faces

    faces = haar.detectMultiScale(gray, 1.05, 3)
    if len(faces) > 0:
        return faces

    return haar.detectMultiScale(gray, 1.2, 3, minSize=(50, 50))


def align_face(gray, face_box):
    x, y, w, h = face_box

    pad = int(min(w, h) * 0.15)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = x + w + pad, y + h + pad

    face_region = gray[y1:y2, x1:x2]
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(face_region)

    if len(eyes) >= 2:
        (ex1, ey1, ew1, eh1) = eyes[0]
        (ex2, ey2, ew2, eh2) = eyes[1]

        dy = (ey2 + eh2 / 2) - (ey1 + eh1 / 2)
        dx = (ex2 + ew2 / 2) - (ex1 + ew1 / 2)
        angle = np.degrees(np.arctan2(dy, dx))

        M = cv2.getRotationMatrix2D(
            (face_region.shape[1] // 2, face_region.shape[0] // 2),
            angle, 1
        )
        face_region = cv2.warpAffine(face_region, M, (face_region.shape[1], face_region.shape[0]))

    return face_region


# ============================================================
#                FACE PREPROCESSING + SAVING
# ============================================================

def apply_gamma_correction(gray, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)], dtype="uint8"
    )
    return cv2.LUT(gray, table)


def preprocess_face(img_np, sid="user", idx=1, save_steps=False):
    # --------------------------------------------------------
    # 1. DETECT FACE USING RETINAFACE (get bounding box)
    # --------------------------------------------------------
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    aligned, box = detect_and_align(bgr)

    if box is not None:
        # Use high-res crop from original image
        x1, y1, x2, y2 = map(int, box)
        face_crop_color = bgr[y1:y2, x1:x2]
    else:
        # fallback Haar
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = detect_face_strong(gray)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        face_crop_color = img_np[y:y+h, x:x+w]

    # Convert to grayscale
    face_crop = cv2.cvtColor(face_crop_color, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # 2. CREATE SAVE DIRECTORY
    # --------------------------------------------------------
    user_dir = os.path.join(PREPROCESS_BASE, str(sid))
    os.makedirs(user_dir, exist_ok=True)

    if save_steps:
        cv2.imwrite(f"{user_dir}/{sid}_{idx}_1_raw.png", face_crop)

    # --------------------------------------------------------
    # 3. APPLY ENHANCEMENT TECHNIQUES
    # --------------------------------------------------------

    # 3.1 Gamma Correction
    gamma_corrected = apply_gamma_correction(face_crop, gamma=1.1)

    # 3.2 Histogram Equalization
    hist_eq = cv2.equalizeHist(gamma_corrected)

    # 3.3 CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(hist_eq)

    # 3.4 Bilateral Filter (preserves edges)
    denoised = cv2.bilateralFilter(clahe_img, d=5, sigmaColor=30, sigmaSpace=30)

    # 3.5 Gaussian smoothing
    smooth = cv2.GaussianBlur(denoised, (3, 3), sigmaX=0.8)

    # 3.6 Light sharpening
    sharp = cv2.addWeighted(smooth, 1.15, cv2.GaussianBlur(smooth, (0, 0), 1), -0.15, 0)

    # 3.7 Morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(sharp, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    # --------------------------------------------------------
    # 4. FINAL RESIZE
    # --------------------------------------------------------
    final = cv2.resize(morph, (200, 200))

    if save_steps:
        cv2.imwrite(f"{user_dir}/{sid}_{idx}_final.png", final)

    return final


# ============================================================
#                  REGISTRATION FACE SAVING
# ============================================================

def preprocess_and_save_faces(image_files, student_id, student_name):
    folder_name = f"{student_id}_{student_name.replace(' ', '').upper()}"
    save_dir = os.path.join(FACES_DIR, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    saved_paths = []
    idx = 1

    for img_file in image_files:
        pil_img = _to_pil(img_file)
        img_np = np.array(pil_img)

        face = preprocess_face(
            img_np,
            sid=str(student_id),
            idx=idx,
            save_steps=True
        )

        if face is None:
            continue

        file_path = os.path.join(save_dir, f"{idx}.png")
        cv2.imwrite(file_path, face)
        saved_paths.append(file_path)
        idx += 1

    return saved_paths, save_dir


# ============================================================
#                           QR CODE
# ============================================================

def generate_qr_for_student(student_id, name, programme):
    data = f"{student_id}|{name}|{programme}"
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image()

    path = os.path.join(QR_DIR, f"{student_id}.png")
    img.save(path)
    return path


def preprocess_qr_image(img_bgr, sid="debug"):
    """
    Saves steps to: data/preprocessed/<sid>_qr/
    """
    qr_dir = os.path.join(PREPROCESS_BASE, f"{sid}_qr")
    os.makedirs(qr_dir, exist_ok=True)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{qr_dir}/1_gray.png", gray)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    cv2.imwrite(f"{qr_dir}/2_blur.png", blur)

    denoised = cv2.fastNlMeansDenoising(blur, None, 10, 7, 21)
    cv2.imwrite(f"{qr_dir}/3_denoised.png", denoised)

    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    cv2.imwrite(f"{qr_dir}/4_thresh.png", thresh)

    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f"{qr_dir}/5_morph_close.png", closing)

    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(f"{qr_dir}/6_morph_open.png", opening)

    edges = cv2.Laplacian(denoised, cv2.CV_64F)
    edges_uint8 = cv2.convertScaleAbs(edges)
    cv2.imwrite(f"{qr_dir}/7_edges.png", edges_uint8)

    contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        roi = img_bgr[y:y+h, x:x+w]
    else:
        roi = img_bgr

    cv2.imwrite(f"{qr_dir}/8_ROI.png", roi)

    return roi, opening, edges_uint8


def decode_qr_with_box(image_input, sid="debug"):
    pil_img = _to_pil(image_input)
    img_np = np.array(pil_img)

    roi, processed, edges = preprocess_qr_image(img_np, sid=sid)

    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(roi)

    annotated = roi.copy()

    if bbox is not None and len(bbox) > 0:
        pts = bbox[0].astype(int)
        for i in range(len(pts)):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % len(pts)])
            cv2.line(annotated, p1, p2, (0, 255, 0), 3)

    return data, Image.fromarray(annotated)


# ============================================================
#                          UTIL
# ============================================================

def _to_pil(image_input):
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if hasattr(image_input, "getvalue"):
        bytes_data = image_input.getvalue()
    else:
        bytes_data = image_input
    return Image.open(io.BytesIO(bytes_data)).convert("RGB")


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
