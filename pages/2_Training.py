import streamlit as st
from pathlib import Path

from utils_sidebar import render_sidebar
from utils_models import train_lbph, train_svm_retinaface
from utils_common import load_students

def load_css():
    css_path = Path("styles/styles.css")
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# -----------------------------------------------------------
# MAIN PAGE
# -----------------------------------------------------------
def main():

    render_sidebar()

    st.title("⚙️ Model Training Center")

    students_df = load_students()
    student_count = len(students_df)

    # -------------------------------------------------------
    # TRAINING OVERVIEW CARD
    # -------------------------------------------------------
    status_text = (
        "<span style='color:#22c55e;'>Ready for training! ✅</span>"
        if student_count > 0 else
        "<span style='color:#ef4444;'>Please register students first. ⚠️</span>"
    )

    overview_html = f"""
    <div class="block-card">
        <div class="card-header">
            <span class="card-icon">🎯</span>
            <h3>Training Overview</h3>
        </div>

    <p style="margin-top: 0.7rem; line-height: 1.7;">
        Train AI models to recognize registered students.<br>
        Each model uses different techniques with different strengths.
    </p>

    <div class="info-box" style="margin-top: 1.2rem;">
        <strong>📊 Current Status:</strong><br>
        {student_count} student(s) registered — {status_text}
    </div>
    </div>
    """
    st.markdown(overview_html, unsafe_allow_html=True)

    # -------------------------------------------------------
    # TRAINING CARDS
    # -------------------------------------------------------
    st.markdown("""
    <div class="block-card">
        <div class="section-title">
            <span>🚀</span><h3>Train Recognition Models</h3>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ---------------------- LBPH -------------------------
    with col1:
        st.markdown("""
        <div class="neutral-card card">
            <div class="section-title">
                <span>🟦</span><h4>Haar Cascade + LBPH</h4>
            </div>
            <p>
                ✓ Texture-based recognition<br>
                ✓Fast and lightweight<br>
                ✓Works well in controlled lighting
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🎯 Train Haar Cascade + LBPH Model", use_container_width=True, disabled=(student_count == 0)):
            with st.spinner("Training LBPH model..."):
                success, message, lbph_metrics = train_lbph()  

            status_class = "success-box" if success else "error-box"
            st.markdown(f"<div class='{status_class}'>{message}</div>", unsafe_allow_html=True)

            if success:
                st.subheader("📊 LBPH Evaluation Metrics")
                st.write(f"**Accuracy:** {lbph_metrics['accuracy']:.2f}")

                st.write("**Confusion Matrix:**")
                st.dataframe(lbph_metrics["confusion_matrix"])


    # -------------------- RetinaFace + SVM ----------------
    with col2:
        st.markdown("""
        <div class="neutral-card card">
            <div class="section-title">
                <span>🟩</span><h4>SVM + ArcFace + RetinaFace</h4>
            </div>
            <p>
                ✓Deep learning detection<br>
                ✓High-quality embeddings<br>
                ✓Best accuracy overall
            </p>
        </div>
        """, unsafe_allow_html=True)

        min_students_for_svm = 2

        if st.button(
            "🎯 Train SVM + ArcFace + RetinaFace",
            use_container_width=True,
            disabled=(student_count < min_students_for_svm)
        ):

            with st.spinner("Training deep-learning model..."):
                success, message, svm_metrics = train_svm_retinaface(students_df)

            status_class = "success-box" if success else "error-box"
            st.markdown(f"<div class='{status_class}'>{message}</div>", unsafe_allow_html=True)

            if success:
                st.subheader("📊 SVM Evaluation Metrics")
                st.write(f"**Accuracy:** {svm_metrics['accuracy']:.2f}")
                st.write(f"**Precision:** {svm_metrics['precision']:.2f}")
                st.write(f"**Recall:** {svm_metrics['recall']:.2f}")
                st.write(f"**F1-score:** {svm_metrics['f1']:.2f}")

                st.write("**Confusion Matrix:**")
                st.dataframe(svm_metrics["confusion_matrix"])


    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # PRETRAINED MODELS SECTION
    # -------------------------------------------------------
    st.markdown("""
    <div class="block-card success-card">
        <div class="section-title">
            <span>✅</span><h3>Pre-trained Models (No Training Required)</h3>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <h4>🧠 CNN + FaceNet</h4>
            <p>
                ✓Embeddings auto-generated<br>
                ✓No training required<br>
                ✓High accuracy<br>
                ✓Deep learning
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📊 Evaluate CNN + FaceNet", use_container_width=True):
            from utils_models import evaluate_facenet_pretrained
            with st.spinner("Evaluating FaceNet performance..."):
                metrics, fig = evaluate_facenet_pretrained()

            st.success("FaceNet evaluation complete.")
            st.write(f"**Same-Person Similarity Mean:** {metrics['same_mean']:.4f}")
            st.write(f"**Different-Person Similarity Mean:** {metrics['diff_mean']:.4f}")
            st.write(f"**Recommended Threshold:** {metrics['threshold']:.2f}")
            st.write(f"**TAR:** {metrics['TAR']:.2f}")
            st.write(f"**FAR:** {metrics['FAR']:.2f}")
            st.write(f"**FRR:** {metrics['FRR']:.2f}")

            st.pyplot(fig)


    with col2:
        st.markdown("""
        <div class="card">
            <h4>🎯 YOLO + ResNet</h4>
            <p>
                ✓Pretrained detection<br>
                ✓Feature extraction<br>
                ✓Cosine similarity<br>
                ✓Real-time
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📊 Evaluate YOLO + ResNet", use_container_width=True):
            from utils_yolo_resnet import evaluate_yolo_resnet
            with st.spinner("Evaluating YOLO + ResNet performance..."):
                metrics, fig = evaluate_yolo_resnet()

            st.success("YOLO + ResNet evaluation complete.")
            st.write(f"**Same-Person Similarity Mean:** {metrics['same_mean']:.4f}")
            st.write(f"**Different-Person Similarity Mean:** {metrics['diff_mean']:.4f}")
            st.write(f"**Recommended Threshold:** {metrics['threshold']:.2f}")
            st.write(f"**TAR:** {metrics['TAR']:.2f}")
            st.write(f"**FAR:** {metrics['FAR']:.2f}")
            st.write(f"**FRR:** {metrics['FRR']:.2f}")

            st.pyplot(fig)


    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
