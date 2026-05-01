import streamlit as st

# Import tabs (modes)
from upload_mode import upload_tab
from record_mode import record_tab

# Import utilities
from analysis_utils import *

# ---------------- Global Config ----------------
st.set_page_config(page_title="Acoustic Analysis Tool")

# Header with logo and title
logo_col, title_col = st.columns([1, 5])
with logo_col:
    st.image("main_logo_violet.svg", width=80)
with title_col:
    st.title("Acoustic Analysis Dashboard")
    st.caption("Backend processing with PRAAT")


# ---------------- AUTHENTICATION ----------------
if not st.user.is_logged_in:
    if st.button("Log in with Google"):
        st.login()
    st.stop()

username = st.user.name
email = st.user.email
folder_id, is_new = handle_user_login(username, email)

if is_new:
    st.success(f"New profile created for {username}")
else:
    st.info(f"Welcome back {username}")

if st.button("Log out"):
    st.logout()

st.warning("DO NOT FORGET TO LOGOUT!")

# ---------------- MODE TABS ----------------

tab1, tab2 = st.tabs([
    "📤 Upload",
    "🎧 Record"
])

with tab1:
    upload_tab(folder_id)
with tab2:
    record_tab(folder_id)
