import streamlit as st

# Import tabs (modes)
from upload_mode import upload_tab
from record_mode import record_tab

# Import utilities
from analysis_utils import *

# ---------------- Global Config ----------------
st.set_page_config(
    page_title="Acoustic Analysis Tool",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    /* Slightly wider centered content */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }

    /* Logo header */
    .logo-header {
        text-align: center;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #eee;
    }

    .logo-header img {
        height: 60px;
        margin-bottom: 0.5rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


def render_logo():
    """Render the logo from SVG file"""
    try:
        with open("main_logo_black.svg", "r") as f:
            svg = f.read()
        import base64
        b64 = base64.b64encode(svg.encode()).decode()
        return f'<img src="data:image/svg+xml;base64,{b64}" alt="Logo">'
    except:
        return ""


# ---------------- AUTHENTICATION ----------------
if not st.user.is_logged_in:
    logo_img = render_logo()
    st.markdown(f"""
        <div class="logo-header">
            {logo_img}
            <h2 style="margin: 0;">Acoustic Analysis Tool</h2>
            <p style="color: #666; margin: 0.25rem 0 0 0;">Powered by PRAAT & Gemini AI</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Log in with Google", type="primary", use_container_width=True):
        st.login()
    st.stop()

# ---------------- LOGGED IN ----------------
username = st.user.name
email = st.user.email
folder_id, is_new = handle_user_login(username, email)

# Header with logo
logo_img = render_logo()
st.markdown(f"""
    <div class="logo-header">
        {logo_img}
        <h2 style="margin: 0;">Acoustic Analysis Dashboard</h2>
        <p style="color: #666; margin: 0.25rem 0 0 0;">Powered by PRAAT & Gemini AI</p>
    </div>
""", unsafe_allow_html=True)

# User info row
col1, col2 = st.columns([3, 1])
with col1:
    if is_new:
        st.success(f"Welcome! New profile created for {username}")
    else:
        st.info(f"Welcome back, {username}")
with col2:
    if st.button("Log out", use_container_width=True):
        st.logout()

st.warning("Remember to log out when done!")

st.divider()

# ---------------- MODE TABS ----------------
tab1, tab2 = st.tabs(["📤 Upload", "🎙️ Record"])

with tab1:
    upload_tab(folder_id)
with tab2:
    record_tab(folder_id)
