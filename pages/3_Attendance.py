import streamlit as st
import os
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from utils_sidebar import render_sidebar
from utils_common import load_students
from utils_vision import decode_qr_with_box, now_str
from utils_models import recognize_with_lbph, recognize_with_facenet, svm_predict_retinaface
from utils_yolo_resnet import recognize_yolo_resnet, pil_from_uploaded_file


ATTEND_CSV = "data/attendance.csv"

def model_status_banner(model, success):
    color = "#22c55e" if success else "#ef4444"
    text  = "SUCCESS" if success else "FAILED"
    icon  = "✅" if success else "❌"

    return f"""
    <div style="
        margin:12px 0;
        padding:12px;
        border-radius:12px;
        font-weight:700;
        text-align:center;
        background:{color};
        color:white;
        letter-spacing:1px;
        font-size:16px;
    ">
        {icon} {model.upper()} {text}
    </div>
    """

def load_css():
    css_path = Path("styles/styles.css")
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


def load_attendance():
    if os.path.exists(ATTEND_CSV):
        return pd.read_csv(ATTEND_CSV)
    return pd.DataFrame(columns=["timestamp", "student_id", "name", "programme", "model"])


def save_attendance(df):
    df.to_csv(ATTEND_CSV, index=False)



def main():

    render_sidebar()

    st.title("✅ Attendance Verification System")

    students_df = load_students()
    att_df = load_attendance()

    st.markdown("""
    <div class="block-card">
        <div class="section-title">
            <h3>📋 How It Works</h3>
        </div>

    <ul class="instruction-list">
        <li><b>Step 1:</b> Scan the student's QR code to retrieve their information.</li>
        <li><b>Step 2:</b> Capture a face photo for verification.</li>
        <li><b>Step 3:</b> AI compares the captured face with registered student data.</li>
        <li><b>Step 4:</b> Attendance is recorded automatically if matched.</li>
    </ul>
                
    </div>
    """, unsafe_allow_html=True)

    if "qr_data" not in st.session_state:
        st.session_state.qr_data = None
        st.session_state.qr_sid = None
        st.session_state.qr_name = None
        st.session_state.qr_prog = None

    st.markdown("""
    <div class="block-card">
        <div class="card-header">
            <div class="card-icon">📱</div>
            <h3>Step 1: Scan Student's QR Code</h3>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.qr_data is None:

        st.markdown("""
        <div class="info-box">
            📸 <b>Tip:</b> Position the QR code inside the camera view for automatic detection.
        </div>
        """, unsafe_allow_html=True)

        qr_image = st.camera_input("📷 Scan QR Code", key="qr_cam")

        if qr_image is not None:

            decoded, annotated = decode_qr_with_box(qr_image)

            if annotated is not None:
                st.image(annotated, caption="Detected QR Code", width=500)

            if decoded:
                parts = decoded.split("|")
                if len(parts) >= 3:
                    sid, name, prog = parts[0], parts[1], parts[2]

                    st.session_state.qr_data = decoded
                    st.session_state.qr_sid = sid
                    st.session_state.qr_name = name
                    st.session_state.qr_prog = prog

                    st.markdown(f"""
                    <div class="success-box centered">
                        <div class="emoji-large">✅</div>
                        <h4>QR Code Verified</h4>
                        <p><b>ID:</b> {sid}<br>
                           <b>Name:</b> {name}<br>
                           <b>Programme:</b> {prog}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("<div class='error-box'>❌ Invalid QR Format</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='warning-box'>⚠️ No QR Detected</div>", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="success-box">
            <h4>Student Information Loaded</h4>
            <p>
                <b>ID:</b> {st.session_state.qr_sid}<br>
                <b>Name:</b> {st.session_state.qr_name}<br>
                <b>Programme:</b> {st.session_state.qr_prog}
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Scan Another Student"):
            st.session_state.qr_data = None
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # STEP 2: FACE VERIFICATION
    # -------------------------------------------------------
    st.markdown("""
    <div class="block-card">
        <div class="card-header">
            <div class="card-icon">🤖</div>
            <h3>Step 2: AI Face Verification</h3>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.qr_data is None:
        st.markdown("<div class='warning-box'>⚠️ Please scan a QR code first.</div>", unsafe_allow_html=True)

    else:

        col1, col2 = st.columns(2)

        with col1:
            model_choice = st.selectbox(
                "Select Recognition Model:",
                ["Haar + LBPH", "CNN + FaceNet", "SVM + ArcFace + RetinaFace", "YOLO + ResNet"]
            )

        with col2:
            if model_choice == "Haar + LBPH":
                threshold = st.slider("Confidence Threshold", 0, 200, 100)
            elif model_choice == "YOLO + ResNet":
                threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.65)
            else:
                threshold = None

        st.markdown("""
        <div class="mini-card centered">
            <h4>📸 Capture Face</h4>
            <p>Ensure your face is clear and bright.</p>
        </div>
        """, unsafe_allow_html=True)

        face_img = st.camera_input("Capture Face", key="face_cam")

        if face_img:
            st.markdown("<div class='success-box'>📸 Face Captured</div>", unsafe_allow_html=True)

        if st.button("🎯 Verify & Record Attendance", use_container_width=True):

            record = False
            conf = None
            dist = None
            prob = None
            sim = None

            sid = st.session_state.qr_sid
            name = st.session_state.qr_name
            prog = st.session_state.qr_prog

            if face_img is None:
                st.markdown("<div class='error-box'>❌ No face image detected.</div>", unsafe_allow_html=True)
                return

            with st.spinner(f"Processing using {model_choice}..."):

                # ------------------ LBPH ------------------
                if model_choice == "Haar + LBPH":
                    result, err, annotated = recognize_with_lbph(face_img, True, name)

                    if annotated is not None:
                        st.image(annotated, caption="LBPH Annotated Result", width=450)

                    if err:
                        record = False
                    else:
                        pred_sid, conf = result
                        record = (str(pred_sid) == sid and conf <= threshold)


                # ------------------ FaceNet ------------------
                elif model_choice == "CNN + FaceNet":
                    
                    dist, err, annotated, is_match = recognize_with_facenet(face_img, sid)

                    if annotated is not None:
                        st.image(annotated, caption="FaceNet Annotated Result", width=450)

                    if err:
                        record = False
                    else:
                        record = is_match

                # ------------------ RetinaFace + SVM + ArcFace------------------
                elif model_choice == "SVM + ArcFace + RetinaFace":
                    pil_img = Image.open(face_img)
                    rgb = np.array(pil_img)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    pred_sid, prob, box = svm_predict_retinaface(bgr)

                    if box is not None:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    st.image(rgb, caption="RetinaFace Annotated Result", width=450)

                    record = (pred_sid is not None and str(pred_sid) == sid)

                # ------------------ YOLO + ResNet ------------------
                elif model_choice == "YOLO + ResNet":
                    face_pil = pil_from_uploaded_file(face_img)
                    match, sim, annotated, err = recognize_yolo_resnet(face_pil, sid, threshold)

                    if annotated is not None:
                        st.image(annotated, caption="YOLO + ResNet Annotated Result", width=450)

                    record = (sim is not None and match)

            result_box = "<div class='success-box centered'>" if record else "<div class='error-box centered'>"

            if record:
                result_box += f"""
                    <h2>Congratualation {name} 🥳🎉 </h2>
                    <p>graduating from <b>{prog}</b> !!!</p>
                    <p><b>{name} ({sid})</b> marked as present.</p>
                """
                st.balloons()

                timestamp = now_str()
                new_entry = {
                    "timestamp": timestamp,
                    "student_id": sid,
                    "name": name,
                    "programme": prog,
                    "model": model_choice
                }

                att_df = load_attendance()
                att_df.loc[len(att_df)] = new_entry
                save_attendance(att_df)

            else:
                result_box += "❌ Verification Failed — Face does not match."

            if model_choice == "Haar + LBPH":
                result_box += (
                    f"<p><b>LBPH Confidence:</b> {conf:.2f}</p>"
                    if conf is not None else
                    "<p><b>LBPH Confidence:</b> N/A</p>"
                )

            elif model_choice == "CNN + FaceNet":
                result_box += (
                    f"<p><b>FaceNet Distance:</b> {dist:.4f}</p>"
                    if dist is not None else
                    "<p><b>FaceNet Distance:</b> N/A (No face detected)</p>"
                )

            elif model_choice == "SVM + ArcFace + RetinaFace":
                if isinstance(prob, (int, float)):
                    result_box += f"<p><b>SVM Probability:</b> {prob:.4f}</p>"
                elif prob is not None:
                    result_box += f"<p><b>SVM Probability:</b> {prob}</p>"
                else:
                    result_box += "<p><b>SVM Probability:</b> N/A (No face detected)</p>"


            elif model_choice == "YOLO + ResNet":
                result_box += (
                    f"<p><b>ResNet Similarity:</b> {sim:.4f}</p>"
                    if sim is not None else
                    "<p><b>ResNet Similarity:</b> N/A (No face detected)</p>"
                )

            result_box += "</div>"
            st.markdown(
                model_status_banner(model_choice, record),
                unsafe_allow_html=True
            )
            st.markdown(result_box, unsafe_allow_html=True)

    st.markdown("""
    <div class="block-card">
        <div class="card-header">
            <div class="card-icon">📋</div>
            <h3>Attendance Records</h3>
        </div>
    """, unsafe_allow_html=True)

    if len(att_df) > 0:
        st.dataframe(att_df, use_container_width=True, hide_index=True)

        csv = att_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "attendance.csv", "text/csv")

    else:
        st.info("👋 No attendance recorded yet.")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
