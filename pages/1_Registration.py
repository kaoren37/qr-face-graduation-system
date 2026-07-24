import streamlit as st
import pandas as pd
import numpy as np
import os

from utils_sidebar import render_sidebar
from utils_common import load_students, save_students
from utils_vision import preprocess_and_save_faces, generate_qr_for_student
from utils_models import facenet_get_embedding, save_facenet_embedding
from utils_yolo_resnet import extract_embeddings_yolo, pil_from_uploaded_file

def load_css(path="styles/styles.css"):
    if os.path.exists(path):
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ---------------------------------------------------
# Main Page
# ---------------------------------------------------
def main():

    render_sidebar()

    st.title("🧑‍🎓 Student Registration")

    df = load_students()

    st.markdown("""
    <div class="block-card gradient-info">
    <div class="card-header">
        <span class="card-icon">📋</span>
        <h3>Registration Instructions</h3>
    </div>

    <ul class="instruction-list">
        <li>📝 Fill in the <b>Student ID</b>, <b>Name</b>, and <b>Programme Graduated</b></li>
        <li>📸 Upload <b>1–20 clear face images</b> (front-facing, good lighting)</li>
        <li>🤖 Photos will be processed by multiple AI models</li>
        <li>🔒 A unique <b>QR Code</b> will be generated automatically</li>
    </ul>

    <div class="info-box">
        💡 <b>Pro Tip:</b> Better photos = Better recognition accuracy!
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="block-card">', unsafe_allow_html=True)

    st.markdown("""
    <div class="card-header">
    <div class="card-icon">✍️</div>
    <h3>Register New Student</h3>
    </div>
    """, unsafe_allow_html=True)

    with st.form("reg_form", clear_on_submit=True):

        col1, col2 = st.columns(2)

        with col1:
            student_id = st.text_input("Student ID", placeholder="e.g., 2024001")
            name = st.text_input("Full Name", placeholder="e.g., John Smith")

        with col2:
            programme = st.text_input("Programme Completed",
                                      placeholder="e.g., BSc Computer Science")

        st.markdown("---")

        uploaded_imgs = st.file_uploader(
            "📷 Upload Face Images (1–20 photos)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if uploaded_imgs:
            st.success(f"{len(uploaded_imgs)} image(s) uploaded.")
            cols = st.columns(min(len(uploaded_imgs), 5))
            for idx, img in enumerate(uploaded_imgs[:5]):
                with cols[idx]:
                    st.image(img, caption=f"Photo {idx + 1}", width=150)

        submitted = st.form_submit_button("🎯 Register Student", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


    if submitted:

        if not str(student_id).strip().isdigit():
            st.markdown("""
            <div class="error-box">
                <strong>❌ Invalid Student ID</strong><br>
                Student ID must contain <b>numbers only</b>.<br>
                Example: <code>2408037</code>
            </div>
            """, unsafe_allow_html=True)
            return

        if not (student_id and name and programme and uploaded_imgs):
            st.markdown("""
            <div class="error-box">
            <strong>⚠️ Incomplete Form</strong><br>
            Please complete all fields and upload at least one image.
            </div>
            """, unsafe_allow_html=True)
            return
        
        existing_id = df[df["student_id"].astype(str) == str(student_id)]
        existing_name = df[
            (df["student_id"].astype(str) == str(student_id)) &
            (df["name"].str.lower() == name.lower())
        ]

        if not existing_id.empty:
            st.markdown(f"""
            <div class="error-box">
                <strong>❌ Duplicate Student ID</strong><br>
                Student ID <b>{student_id}</b> is already registered.<br>
                Please verify the ID or use a different one.
            </div>
            """, unsafe_allow_html=True)
            return

        if not existing_name.empty:
            st.markdown(f"""
            <div class="error-box">
                <strong>❌ Duplicate Registration</strong><br>
                Student <b>{name}</b> with ID <b>{student_id}</b> already exists.
            </div>
            """, unsafe_allow_html=True)
            return
        
        if len(uploaded_imgs) < 3:
            st.markdown("""
            <div class="error-box">
                <strong>❌ Insufficient Images</strong><br>
                Please upload <b>at least 3 clear face images</b> for registration.<br>
                This helps improve recognition accuracy.
            </div>
            """, unsafe_allow_html=True)
            return

        progress = st.progress(0)
        status = st.empty()

        # --- Step 1: Save Faces ---
        status.markdown("<div class='info-box'>🔄 Processing face images...</div>",
                        unsafe_allow_html=True)
        progress.progress(20)

        saved_paths, face_folder = preprocess_and_save_faces(
            uploaded_imgs, student_id, name
        )

        if len(saved_paths) == 0:
            st.markdown("""
            <div class="error-box"><strong>❌ No Face Detected</strong><br>
            No face detected. Please upload clearer images.
            </div>
            """, unsafe_allow_html=True)
            return

        # --- Step 2: FaceNet Embeddings ---
        status.markdown("<div class='info-box'>🧠 Generating FaceNet embeddings...</div>",
                        unsafe_allow_html=True)
        progress.progress(40)

        embeddings = []
        for img in uploaded_imgs:
            emb, err, _ = facenet_get_embedding(img, use_alignment=False)

            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            st.markdown("""
            <div class="error-box"><strong>❌ Embedding Error</strong><br>
            Unable to extract FaceNet embeddings.</div>
            """, unsafe_allow_html=True)
            return

        avg_emb = np.mean(embeddings, axis=0)
        avg_emb /= np.linalg.norm(avg_emb)
        save_facenet_embedding(student_id, avg_emb)


        # --- Step 3: ResNet Embedding ---
        status.markdown("<div class='info-box'>🔬 Processing ResNet embeddings...</div>",
                        unsafe_allow_html=True)
        progress.progress(60)

        os.makedirs("data/encodings_resnet", exist_ok=True)
        pil_img = pil_from_uploaded_file(uploaded_imgs[0])

        res_emb, box, err = extract_embeddings_yolo(pil_img)
        if err:
            st.markdown(f"<div class='error-box'>ResNet Error: {err}</div>",
                        unsafe_allow_html=True)
            return

        np.save(f"data/encodings_resnet/{student_id}.npy", res_emb)

        # --- Step 4: QR Code ---
        status.markdown("<div class='info-box'>🔒 Generating QR Code...</div>",
                        unsafe_allow_html=True)
        progress.progress(80)

        qr_path = generate_qr_for_student(student_id, name, programme)

        # --- Step 5: Save Student ---
        status.markdown("<div class='info-box'>💾 Saving Student...</div>",
                        unsafe_allow_html=True)
        progress.progress(95)

        new_row = {
            "student_id": student_id,
            "name": name,
            "programme": programme,
            "face_path": face_folder,
            "qr_path": qr_path,
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_students(df)

        progress.progress(100)
        status.empty()

        # Success Message
        st.markdown(f"""
        <div class="success-box centered">
        <div class="emoji-large">🎉</div>
        <h3>Registration Successful!</h3>
        <p><b>{name}</b> has been added.</p>
        <p>{len(saved_paths)} processed successfully.</p>
        </div>
        """, unsafe_allow_html=True)

        # QR Code Section
        st.markdown('<div class="block-card centered">', unsafe_allow_html=True)
        st.markdown("""
        <div class="card-header centered">
        <div class="card-icon">🔐</div>
        <h3>Generated QR Code</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(qr_path, width=260)
            with open(qr_path, "rb") as f:
                st.download_button(
                    "📥 Download QR Code",
                    f,
                    file_name=f"{student_id}_qr.png",
                    mime="image/png",
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # Student Table
    # ---------------------------------------------------
    st.markdown('<div class="block-card">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card-header space-between">
    <div class="card-header">
        <div class="card-icon">📚</div>
        <h3>Registered Students</h3>
    </div>
    <span class="badge badge-primary">Total: {len(df)} student(s)</span>
    </div>
    """, unsafe_allow_html=True)

    if len(df) > 0:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("👋 No students registered yet.")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
