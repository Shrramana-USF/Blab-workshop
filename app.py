import streamlit as st
import base64

# Import tabs (modes)
from upload_mode import upload_tab
from record_mode import record_tab

# Import utilities
from analysis_utils import *

# ---------------- Global Config ----------------
st.set_page_config(
    page_title="Acoustic Analysis Tool",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Custom CSS for Modern Look ----------------
st.markdown("""
<style>
    /* Main container - full width */
    .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: 1rem;
        max-width: 100%;
    }

    /* Header styling */
    .main-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 1.5rem;
    }

    .logo-title {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .logo-title img {
        height: 50px;
    }

    .logo-title h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
        color: #1f1f1f;
    }

    .logo-title p {
        margin: 0;
        font-size: 0.9rem;
        color: #666;
    }

    /* User info styling */
    .user-info {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem 1rem;
        background: #f8f9fa;
        border-radius: 8px;
    }

    .user-info span {
        font-weight: 500;
        color: #333;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Card-like sections */
    .stExpander {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }

    /* Radio buttons horizontal */
    .stRadio > div {
        flex-wrap: wrap;
    }

    /* DataFrame styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Success/Info/Warning boxes */
    .stAlert {
        border-radius: 8px;
    }

    /* File uploader */
    .stFileUploader {
        border-radius: 10px;
    }

    /* Hide default title */
    .main-title-hidden {
        display: none;
    }

    /* Logout button styling */
    .logout-btn button {
        background-color: #ff4b4b !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


def get_logo_base64():
    """Read and encode the logo as base64"""
    try:
        with open("main_logo_black.svg", "r") as f:
            svg_content = f.read()
        return base64.b64encode(svg_content.encode()).decode()
    except:
        return None


# ---------------- AUTHENTICATION ----------------
if not st.user.is_logged_in:
    # Login page with centered content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        logo_b64 = get_logo_base64()
        if logo_b64:
            st.markdown(f"""
                <div style="text-align: center; padding: 3rem 0;">
                    <img src="data:image/svg+xml;base64,{logo_b64}" style="height: 80px; margin-bottom: 1rem;">
                    <h1 style="margin: 0; font-size: 2rem;">Acoustic Analysis Tool</h1>
                    <p style="color: #666; margin-top: 0.5rem;">Powered by PRAAT & Gemini AI</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.title("Acoustic Analysis Tool")
            st.caption("Powered by PRAAT & Gemini AI")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔐 Log in with Google", use_container_width=True, type="primary"):
            st.login()
    st.stop()

# ---------------- Logged In Header ----------------
username = st.user.name
email = st.user.email
folder_id, is_new = handle_user_login(username, email)

# Header with logo and user info
logo_b64 = get_logo_base64()
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    if logo_b64:
        st.markdown(f"""
            <div class="logo-title">
                <img src="data:image/svg+xml;base64,{logo_b64}" alt="Logo">
                <div>
                    <h1>Acoustic Analysis Dashboard</h1>
                    <p>Powered by PRAAT & Gemini AI</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.title("Acoustic Analysis Dashboard")
        st.caption("Powered by PRAAT & Gemini AI")

with header_col2:
    st.markdown(f"""
        <div class="user-info">
            <span>👤 {username}</span>
        </div>
    """, unsafe_allow_html=True)
    if st.button("🚪 Log out", key="logout_btn"):
        st.logout()

st.markdown("<hr style='margin: 0.5rem 0 1.5rem 0; border: none; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)

# Welcome message
if is_new:
    st.success(f"🎉 Welcome! New profile created for {username}")
else:
    st.info(f"👋 Welcome back, {username}!")

# ---------------- MODE TABS ----------------
tab1, tab2 = st.tabs([
    "📤 Upload Audio",
    "🎙️ Record Audio"
])

with tab1:
    upload_tab(folder_id)
with tab2:
    record_tab(folder_id)
