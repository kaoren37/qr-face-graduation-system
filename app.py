import streamlit as st
from utils_common import init_dirs

st.set_page_config(
    page_title="Graduation Face Recognition",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(path="styles/styles.css"):
    with open(path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

with st.sidebar:
        st.markdown("""
        <div class='sidebar-header'>
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🎓</div>
            <h2 style='margin: 0; font-size: 1.5rem; font-weight: 700;'>Smart Graduation</h2>
        </div>
        """, unsafe_allow_html=True)

        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/1_Registration.py", label="Registration", icon="🧑‍🎓")
        st.page_link("pages/2_Training.py", label="Training", icon="⚙️")
        st.page_link("pages/3_Attendance.py", label="Attendance", icon="📸")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class='sidebar-footer'>
            Image Processing © 
        </div>
        """, unsafe_allow_html=True)


def main():
    init_dirs()

    st.markdown("""
    <div class='hero-card'>
        <h1>🎓 Smart Graduation System</h1>
        <p class='hero-sub'>
            Secure • Fast • Intelligent Attendance Management
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="block-card gradient-info">

    <div class="card-header">
        <h3>👋 Welcome to the 2025 Graduation Ceremony</h3>
    </div>

    <p>
        Intelligent and secure platform designed to modernize graduation attendance through
        <b>QR Code Verification</b> and <b>AI-Powered Face Recognition</b>.
        The system ensures a smooth, accurate, and dignified graduation experience while
        minimizing manual processes, reducing human error, and improving operational efficiency.
    </p>

    <h4>Malaysia MADANI Values</h4>

    <ul class="instruction-list">
        <li>🤍 <b>Compassion</b> — Fair and inclusive verification for every graduate</li>
        <li>🌱 <b>Sustainability</b> — Reduced paper usage through digital workflows</li>
        <li>🚀 <b>Innovation</b> — Advanced AI technologies that enhance trust, transparency, and efficiency</li>
    </ul>

    <div class="info-box">
        🌱 <b>Digital • Sustainable • Inclusive</b><br>
        A future-ready graduation experience powered by AI.
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='block-card'>", unsafe_allow_html=True)
    st.subheader("🚀 Quick Start Guide")

    col1, col2, col3 = st.columns(3)

    col1.markdown("<div class='quick-step'>👤<h4>1. Register</h4><p>Add students + capture faces</p></div>", unsafe_allow_html=True)
    col2.markdown("<div class='quick-step'>⚙️<h4>2. Train</h4><p>Train AI recognition models</p></div>", unsafe_allow_html=True)
    col3.markdown("<div class='quick-step'>✅<h4>3. Verify</h4><p>Scan QR + match face</p></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
