import streamlit as st

def render_sidebar():
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
