import io
import os
import tempfile
import numpy as np
import pandas as pd
import soundfile as sf
import math
import parselmouth as pm
import matplotlib.pyplot as plt
from parselmouth.praat import call as praat_call
from box_sdk_gen import BoxClient, BoxCCGAuth, CCGConfig
from box_sdk_gen.managers.uploads import UploadFileAttributes, UploadFileAttributesParentField, UploadFileVersionAttributes
from box_sdk_gen.internal.utils import read_byte_stream
import streamlit as st
from streamlit_advanced_audio import audix, WaveSurferOptions
import google.generativeai as genai

# ---------------- BOX SETUP ----------------
BASE_FOLDER_ID = "341557643428"
CSV_FILENAME = "users.csv"


def get_box_client() -> BoxClient:
    """
    Create Box client using CCG (Client Credentials Grant).
    No refresh tokens needed - SDK handles everything automatically.

    Required secrets:
        [box]
        client_id = "your_client_id"
        client_secret = "your_client_secret"
        enterprise_id = "your_enterprise_id"
    """
    try:
        ccg_config = CCGConfig(
            client_id=st.secrets["box"]["client_id"],
            client_secret=st.secrets["box"]["client_secret"],
            enterprise_id=st.secrets["box"]["enterprise_id"]
        )
        auth = BoxCCGAuth(config=ccg_config)
        return BoxClient(auth=auth)

    except KeyError as e:
        st.error(f"Missing Box config: {e}")
        st.stop()

def get_users_csv(client: BoxClient):
    items = client.folders.get_folder_items(BASE_FOLDER_ID)
    csv_file = next((i for i in items.entries if i.name == CSV_FILENAME), None)
    if csv_file:
        byte_stream = client.downloads.download_file(csv_file.id)
        content = read_byte_stream(byte_stream)
        df = pd.read_csv(io.BytesIO(content))
        return df, csv_file.id
    else:
        return pd.DataFrame(columns=["username", "email", "folder_id"]), None

def update_users_csv(client: BoxClient, df: pd.DataFrame, file_id: str | None):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    if file_id:
        client.uploads.upload_file_version(
            file_id,
            UploadFileVersionAttributes(name=CSV_FILENAME),
            buf
        )
    else:
        uploaded_files = client.uploads.upload_file(
            UploadFileAttributes(
                name=CSV_FILENAME,
                parent=UploadFileAttributesParentField(id=BASE_FOLDER_ID),
            ),
            buf,
        )
        return uploaded_files.entries[0].id

def create_user_folder(client: BoxClient, folder_name: str):
    new_folder = client.folders.create_folder(
        name=folder_name,
        parent={"id": BASE_FOLDER_ID}
    )
    return new_folder.id

def handle_user_login(username: str, email: str):
    client = get_box_client()
    df, file_id = get_users_csv(client)
    user_row = df[df["email"] == email]
    if not user_row.empty:
        return user_row.iloc[0]["folder_id"], False
    folder_id = create_user_folder(client, email)
    new_row = pd.DataFrame({
        "username": [username],
        "email": [email],
        "folder_id": [folder_id]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    update_users_csv(client, df, file_id)
    return folder_id, True


# ---------------- Global Config ----------------
PITCH_FLOOR = 30.0
PITCH_CEILING = 600.0
AUDIX = True


def create_session_folder(client: BoxClient, user_folder_id: str):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    folder = client.folders.create_folder(
        name=f"session_{timestamp}",
        parent={"id": str(user_folder_id)}
    )
    return folder.id


def upload_to_user_box(client: BoxClient, folder_id: str, filename: str, data: bytes):
    """Uploads a file (bytes) to the user's Box folder with given filename."""
    buf = io.BytesIO(data)
    uploaded = client.uploads.upload_file(
        UploadFileAttributes(
            name=filename,
            parent=UploadFileAttributesParentField(id=folder_id),
        ),
        buf,
    )
    return uploaded.entries[0].id

def ensure_task_folder(client: BoxClient, user_folder_id: str, task_name: str, create_if_missing: bool = True):
    """Ensure a task-specific folder exists under the user's Box folder."""
    items = client.folders.get_folder_items(user_folder_id)
    task_folder = next((i for i in items.entries if i.type == "folder" and i.name == task_name), None)

    if task_folder:
        return task_folder.id

    if create_if_missing:
        new_folder = client.folders.create_folder(
            name=task_name,
            parent={"id": str(user_folder_id)}
        )
        return new_folder.id
    else:
        return None

def save_analysis_to_box(
    y: np.ndarray,
    sr: int,
    features: pd.DataFrame,
    figs: dict[str, plt.Figure],
    user_folder_id: str
):
    """Save analysis results (audio, features, existing plots) to Box."""

    client = get_box_client()
    session_folder_id = create_session_folder(client, user_folder_id)


    # --- Save audio ---
    audio_buf = io.BytesIO()
    sf.write(audio_buf, ensure_mono(y).astype(np.float32), sr, format="WAV")
    audio_buf.seek(0)
    upload_to_user_box(client, session_folder_id, "audio.wav", audio_buf.getvalue())

    # --- Save features CSV ---
    csv_buf = io.BytesIO()
    features.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    upload_to_user_box(client, session_folder_id, "features.csv", csv_buf.getvalue())

    # --- Save existing plots ---
    for name, fig in figs.items():
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png")
        plt.close(fig)
        img_buf.seek(0)
        upload_to_user_box(client, session_folder_id, f"{name}.png", img_buf.getvalue())

    st.success(f"Analysis data saved to Box")




# ---------------- Helpers ----------------

def read_audio_bytes(raw: bytes):
    """Read WAV or MP3 bytes into mono float32 numpy array + sample rate."""
    data, sr = sf.read(io.BytesIO(raw), always_2d=False)
    y = ensure_mono(np.asarray(data)).astype(np.float32)
    return y, sr


def ensure_mono(y: np.ndarray) -> np.ndarray:
    return y if y.ndim == 1 else np.mean(y, axis=1)


def save_temp_mono_wav(y: np.ndarray, sr: int) -> str:
    """Save mono audio to a temp wav file and return its path."""
    y_mono = ensure_mono(y).astype(np.float32)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tf.name, y_mono, sr, format="WAV")
    return tf.name


def estimate_f0_praat(pitch) -> float | None:
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if f0.size == 0:
        return None
    return float(np.median(f0))


def jitter_shimmer(snd: pm.Sound, pitch):
    """Compute PRAAT jitter & shimmer (local measures)."""
    pp = praat_call(snd, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)

    # Jitter
    jit_local = praat_call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = praat_call(pp, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)

    # Shimmer
    shim_local = praat_call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = praat_call([snd, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return {
        "jitter_local": float(jit_local),
        "jitter_abs": float(localabsoluteJitter),
        "shimmer_local": float(shim_local),
        "shimmer_db": float(localdbShimmer),
    }


def pitch_contour(pitch):
    xs = pitch.xs()
    f0 = pitch.selected_array["frequency"]
    f0[f0 == 0] = np.nan
    return xs, f0


def intensity_contour(intensity):
    xs = intensity.xs()
    values = intensity.values[0]
    return xs, values


def compute_cpp(snd: pm.Sound) -> float | None:
    """Compute CPP (Cepstral Peak Prominence Smoothed) from an existing Praat Sound object."""
    # Pitch Ceiling 300 as per the ADSV app setting
    try:
        pc = praat_call(
            snd, "To PowerCepstrogram", PITCH_FLOOR, 0.002, 300, 50.0)

        cpp = praat_call(
            pc, "Get CPPS", "no", 0.02, 0.0005, PITCH_FLOOR, 300,
            0.05,   # tolerance
            "parabolic", 0.001,0.05, "Exponential decay", "Robust slow")
        return float(cpp)
    except Exception as e:
        print(f"CPP computation error: {e}")
        return None



def compute_spectrogram(snd: pm.Sound):
    """Compute spectrogram from an existing Praat Sound object."""
    return snd.to_spectrogram(
        time_step=0.01,
        window_length=0.03,
        maximum_frequency=8000
    )


def plot_spectrogram(spectrogram):
    """Plot a given Praat Spectrogram object."""
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    fig, ax = plt.subplots()
    img = ax.pcolormesh(X, Y, sg_db, shading="auto", cmap="magma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")
    fig.colorbar(img, ax=ax, label="dB")
    return fig




def summarize_features(snd: pm.Sound, pitch, intensity):
    features = {}

    # Fundamental frequency
    f0 = estimate_f0_praat(pitch)
    features["Fundamental Freq (Hz)"] = f"{f0:.2f}" if f0 is not None else "—"

    features["Audio duration"] = f"{snd.get_total_duration():.2f}"

    # Jitter/Shimmer
    try:
        js = jitter_shimmer(snd, pitch)
        features["Jitter (local, %)"] = f"{js['jitter_local']*100:.2f}"
        features["Jitter (abs, ms)"] = f"{js['jitter_abs']*1000:.3f}"
        features["Shimmer (local, %)"] = f"{js['shimmer_local']*100:.2f}"
        features["Shimmer (dB)"] = f"{js['shimmer_db']:.2f}"
    except Exception:
        features["Jitter/Shimmer"] = "N/A"

    # Pitch contour stats
    xs, f0_contour = pitch_contour(pitch)

    pitch_values = pitch.selected_array['frequency']

    if np.any(~np.isnan(f0_contour)):
        features["Pitch Mean (Hz)"] = f"{np.nanmean(f0_contour):.2f}"
        features["Pitch Median (Hz)"] = f"{np.nanmedian(f0_contour):.2f}"
        features["Pitch Min (Hz)"] = f"{np.nanmin(f0_contour):.2f}"
        features["Pitch Max (Hz1)"] = pitch_values.max()
        features["Pitch Max (Hz)"] = f"{np.nanmax(f0_contour):.2f}"
        features["Pitch Range (Hz)"] = f"{float(features['Pitch Max (Hz)']) - float(features['Pitch Min (Hz)']):.2f}"
        features["Octave"] = f"{math.log2(float(features['Pitch Max (Hz)']) / float(features['Pitch Min (Hz)'])):.2f}"


    # Intensity contour stats ----- Changed to Energy
    xs, inten_contour = intensity_contour(intensity)
    if len(inten_contour) > 0:
        features["Energy Mean (dB)"] = f"{np.mean(inten_contour):.2f}"
        features["Energy Min (dB)"] = f"{np.min(inten_contour):.2f}"
        features["Energy Max (dB)"] = f"{np.max(inten_contour):.2f}"
        features["Energy Range (dB)"] = f"{float(features['Energy Max (dB)']) - float(features['Energy Min (dB)']):.2f}"


    # CPP
    cpp_val = compute_cpp(snd)
    features["CPP (dB)"] = f"{cpp_val:.2f}" if cpp_val is not None else "-"

    return features


def play_audio_wav_bytes(wav_bytes: bytes):
    """Play WAV bytes using Audix if available, else st.audio."""
    if AUDIX:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            tf.write(wav_bytes)
            temp_path = tf.name
        try:
            options = WaveSurferOptions(height=100)
            audix(temp_path, wavesurfer_options=options)
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    else:
        st.audio(wav_bytes, format="audio/wav")


# ---------------- Analytics + Box ----------------

def fetch_all_features(client: BoxClient, user_folder_id: str):
    """
    Returns a tuple:
    - DataFrame with all features
    - dict mapping session_name -> audio_file_id
    """
    all_data = []
    audio_map = {}  # session_name -> audio_file_id

    user_items = client.folders.get_folder_items(user_folder_id)
    for entry in user_items.entries:
        if entry.type == "folder" and entry.name.startswith("session_"):
            session_name = entry.name
            session_id = entry.id

            # Get files inside the session folder
            session_items = client.folders.get_folder_items(session_id)

            # Find features.csv
            feat_file = next((f for f in session_items.entries if f.name == "features.csv"), None)
            if feat_file:
                byte_stream = client.downloads.download_file(feat_file.id)
                content = read_byte_stream(byte_stream)
                df = pd.read_csv(io.BytesIO(content))
                df["session"] = session_name
                all_data.append(df)

            # Find audio.wav
            audio_file = next((f for f in session_items.entries if f.name == "audio.wav"), None)
            if audio_file:
                audio_map[session_name] = audio_file.id

    if all_data:
        return pd.concat(all_data, ignore_index=True), audio_map
    else:
        return pd.DataFrame(), audio_map


# ---------------- GEMINI AI FUNCTIONS ----------------

def init_gemini():
    """
    Gemini via AI Studio API key stored securely in Streamlit Secrets or env var.
    Prefer Streamlit Cloud Secrets: GOOGLE_API_KEY = "..."
    """
    api_key = None
    try:
        # Supports top-level: GOOGLE_API_KEY = "..."
        api_key = st.secrets.get("GOOGLE_API_KEY", None)
        # Supports sectioned:
        # [Gemini]
        # GOOGLE_API_KEY = "..."
        if not api_key:
            api_key = st.secrets.get("Gemini", {}).get("GOOGLE_API_KEY", None)
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return None, "Missing GOOGLE_API_KEY. Add it to Streamlit Secrets (recommended) or as an environment variable."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3.1-pro-preview")

    return model, None


def gemini_review_voice_with_audio(
    model,
    df_features: pd.DataFrame,
    audio_wav_bytes: bytes,
    task_name: str,
    reference_group: str
):
    """
    Sends features + AUDIO (wav bytes) to Gemini for interpretation.
    Does NOT infer/guess gender/sex/identity. Uses user-selected reference group.
    """
    rows = df_features.to_dict(orient="records")

    prompt = f"""
You are assisting with voice acoustics interpretation for the task: "{task_name}".

IMPORTANT:
- Use the selected reference group only: "{reference_group}".
- If reference group is "Unknown / show both", provide interpretation for typical adult male and typical adult female ranges, without guessing which applies.
- Do not provide a medical diagnosis. Use cautious, non-diagnostic language.

Input data:
1) Extracted acoustic features (Feature, Value): {rows}
2) Audio recording is attached (WAV).

Please produce:
A) Summary (2–5 sentences). IDENTIFY "gender/sex" and "age" based on the audio.
B) Range check vs reference group (bullets). If a feature is out of typical ranges, say so with uncertainty and mention it depends on recording/task.
C) Any potential flags (bullets) — only if supported by the data/audio; otherwise "No obvious flags."
D) Suggestions (bullets): e.g., repeat recording conditions, consult clinician if symptoms exist, etc.
"""

    audio_part = {
        "inline_data": {
            "mime_type": "audio/wav",
            "data": audio_wav_bytes
        }
    }

    resp = model.generate_content([prompt, audio_part], generation_config=genai.GenerationConfig(temperature=0.5))
    return resp.text if hasattr(resp, "text") else str(resp)


def gemini_byo_prompt(
    model,
    user_prompt: str,
    byo_option: str,
    df_features: pd.DataFrame = None,
    audio_wav_bytes: bytes = None
):
    """
    Sends user's custom prompt with optional audio and/or features to Gemini.

    byo_option can be:
    - "Only audio": sends prompt + audio WAV bytes (no features)
    - "Only extracted features": sends prompt + features DataFrame (no audio)
    - "Both audio and features": sends prompt + audio + features
    - "Just prompt": sends only the user's text prompt (no audio, no features)
    """
    content_parts = []

    if byo_option == "Only audio":
        full_prompt = f"{user_prompt}\n\n[Audio recording is attached below]"
        content_parts.append(full_prompt)
        if audio_wav_bytes:
            audio_part = {
                "inline_data": {
                    "mime_type": "audio/wav",
                    "data": audio_wav_bytes
                }
            }
            content_parts.append(audio_part)

    elif byo_option == "Only extracted features":
        rows = df_features.to_dict(orient="records") if df_features is not None else []
        full_prompt = f"{user_prompt}\n\nExtracted acoustic features (Feature, Value): {rows}"
        content_parts.append(full_prompt)

    elif byo_option == "Both audio and features":
        rows = df_features.to_dict(orient="records") if df_features is not None else []
        full_prompt = f"{user_prompt}\n\nExtracted acoustic features (Feature, Value): {rows}\n\n[Audio recording is attached below]"
        content_parts.append(full_prompt)
        if audio_wav_bytes:
            audio_part = {
                "inline_data": {
                    "mime_type": "audio/wav",
                    "data": audio_wav_bytes
                }
            }
            content_parts.append(audio_part)

    else:  # "Just prompt"
        content_parts.append(user_prompt)

    resp = model.generate_content(content_parts)
    return resp.text if hasattr(resp, "text") else str(resp)


# ---------------- SHARED UI COMPONENTS ----------------

# Task list used by both upload and record modes
TASKS = [
    "Rainbow passage",
    "Maximum sustained phonation on 'aaah'",
    "Comfortable sustained phonation on 'eeee'",
    "Glide up to your highest pitch on 'eeee'",
    "Glide down to your lowest pitch on 'eeee'",
    "Sustained 'aaah' at minimum volume",
    "Maximum loudness level (brief 'AAAH')",
    "Conversational speech",
    "Random"
]

# Analysis mode labels
MODE_LABELS = {
    "praat": "PRAAT Analysis Only",
    "ai": "Default Analysis with AI",
    "byo": "BYO Prompt"
}


def init_session_state(prefix: str):
    """Initialize session state variables for a given mode prefix (upload/record)."""
    if f"{prefix}_ai_df" not in st.session_state:
        st.session_state[f"{prefix}_ai_df"] = None
    if f"{prefix}_ai_gemini_text" not in st.session_state:
        st.session_state[f"{prefix}_ai_gemini_text"] = None
    if f"{prefix}_ai_last_task" not in st.session_state:
        st.session_state[f"{prefix}_ai_last_task"] = None
    if f"{prefix}_byo_gemini_text" not in st.session_state:
        st.session_state[f"{prefix}_byo_gemini_text"] = None
    if f"{prefix}_byo_mode_active" not in st.session_state:
        st.session_state[f"{prefix}_byo_mode_active"] = False
    if f"{prefix}_byo_chat_history" not in st.session_state:
        st.session_state[f"{prefix}_byo_chat_history"] = []
    if f"{prefix}_analysis_mode" not in st.session_state:
        st.session_state[f"{prefix}_analysis_mode"] = None


def clear_session_state(prefix: str):
    """Clear AI results when switching tasks."""
    st.session_state[f"{prefix}_ai_df"] = None
    st.session_state[f"{prefix}_ai_gemini_text"] = None
    st.session_state[f"{prefix}_byo_gemini_text"] = None
    st.session_state[f"{prefix}_byo_mode_active"] = False
    st.session_state[f"{prefix}_byo_chat_history"] = []
    st.session_state[f"{prefix}_analysis_mode"] = None


def render_analysis_mode_buttons(prefix: str, selected_task: str):
    """Render the 3 analysis mode buttons and return current mode."""
    st.markdown("---")
    st.markdown("#### Select Analysis Mode")
    col1, col2, col3 = st.columns(3)

    if col1.button("Analyze Audio", key=f"{prefix}_analyze_{selected_task}", use_container_width=True):
        st.session_state[f"{prefix}_analysis_mode"] = "praat"
        st.session_state[f"{prefix}_byo_mode_active"] = False

    if col2.button("Default Analysis with AI", key=f"{prefix}_analyze_ai_{selected_task}", use_container_width=True):
        st.session_state[f"{prefix}_analysis_mode"] = "ai"
        st.session_state[f"{prefix}_byo_mode_active"] = False

    if col3.button("BYO Prompt", key=f"{prefix}_analyze_byo_{selected_task}", use_container_width=True):
        st.session_state[f"{prefix}_analysis_mode"] = "byo"
        st.session_state[f"{prefix}_byo_mode_active"] = True

    # Show current selection
    if st.session_state[f"{prefix}_analysis_mode"]:
        st.success(f"Selected: **{MODE_LABELS[st.session_state[f'{prefix}_analysis_mode']]}**")

    return st.session_state[f"{prefix}_analysis_mode"]


def render_byo_config(prefix: str, selected_task: str):
    """
    Render BYO Prompt configuration UI.
    Returns (byo_option, byo_prompt, should_return) tuple.
    should_return is True if we're in "Just prompt" mode and handled the conversation.
    """
    byo_option = None
    byo_prompt = ""

    if not st.session_state[f"{prefix}_byo_mode_active"]:
        return byo_option, byo_prompt, False

    st.markdown("---")
    st.markdown("#### BYO Prompt Configuration")

    byo_option = st.radio(
        "What to send to AI:",
        options=["Only audio", "Only extracted features", "Both audio and features", "Just prompt"],
        index=2,
        horizontal=True,
        key=f"{prefix}_byo_option_{selected_task}",
    )

    # Different UI for "Just prompt" - conversation mode
    if byo_option == "Just prompt":
        st.markdown("##### Conversation Mode")
        st.caption("Have a conversation with Gemini")

        # Display chat history
        for msg in st.session_state[f"{prefix}_byo_chat_history"]:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                st.chat_message("assistant").markdown(msg["content"])

        # Chat input
        byo_chat_prompt = st.chat_input("Type your message...", key=f"{prefix}_byo_chat_input_{selected_task}")
        byo_submit_clicked = byo_chat_prompt is not None and byo_chat_prompt.strip() != ""

        # Clear conversation button
        if st.session_state[f"{prefix}_byo_chat_history"]:
            if st.button("Clear Conversation", key=f"{prefix}_byo_clear_chat_{selected_task}"):
                st.session_state[f"{prefix}_byo_chat_history"] = []
                st.rerun()

        # Handle conversation
        if byo_submit_clicked:
            model, err = init_gemini()
            if err:
                st.error(f" {err}")
            else:
                try:
                    # Build history for chat session
                    gemini_history = []
                    for msg in st.session_state[f"{prefix}_byo_chat_history"]:
                        gemini_history.append({
                            "role": msg["role"] if msg["role"] == "user" else "model",
                            "parts": [msg["content"]]
                        })

                    chat = model.start_chat(history=gemini_history)

                    st.session_state[f"{prefix}_byo_chat_history"].append({
                        "role": "user",
                        "content": byo_chat_prompt
                    })

                    with st.spinner("Gemini is thinking..."):
                        response = chat.send_message(byo_chat_prompt)
                        response_text = response.text if hasattr(response, "text") else str(response)

                    st.session_state[f"{prefix}_byo_chat_history"].append({
                        "role": "assistant",
                        "content": response_text
                    })

                    st.rerun()

                except Exception as e:
                    st.error(f"Gemini failed: {e}")

        st.markdown("---")
        return byo_option, byo_prompt, True  # Signal to return early
    else:
        # Standard text area for other options
        byo_prompt = st.text_area(
            "Enter your custom prompt:",
            placeholder="",
            height=150,
            key=f"{prefix}_byo_prompt_{selected_task}",
        )

    st.markdown("---")
    return byo_option, byo_prompt, False


def render_reference_group_selector(prefix: str, selected_task: str):
    """Render reference group selector for AI mode."""
    if st.session_state[f"{prefix}_analysis_mode"] == "ai":
        st.radio(
            "Reference group for typical ranges (self-reported):",
            options=["Unknown / show both", "Adult male (self-reported)", "Adult female (self-reported)"],
            index=0,
            horizontal=True,
            key=f"{prefix}_gemini_reference_group_{selected_task}",
        )


def get_audio_region(result, y, sr):
    """Extract audio region from audix result. Returns (y_region, info_message)."""
    if result and result.get("selectedRegion"):
        start = result["selectedRegion"]["start"]
        end = result["selectedRegion"]["end"]
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        return y[start_idx:end_idx], f"Analysing selected region: {start:.2f}s – {end:.2f}s"
    else:
        return y, "No region selected: Analysing entire file"


def run_praat_analysis(y_region, sr):
    """
    Run PRAAT analysis on audio region.
    Returns (snd, pitch, intensity, df, figs) or (None, None, None, None, None) if failed.
    """
    import parselmouth as pm

    snd = pm.Sound(y_region, sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=None, pitch_floor=30, pitch_ceiling=600)
    intensity = snd.to_intensity()

    f0 = estimate_f0_praat(pitch)
    if f0 is None:
        st.warning("No stable fundamental frequency detected.")
        return None, None, None, None, None

    features = summarize_features(snd, pitch, intensity)
    df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])

    # Display features
    st.subheader("Extracted Features")
    st.dataframe(df, width="stretch", hide_index=True)

    # Create plots
    figs = {}

    xs, f0_contour = pitch_contour(pitch)
    fig, ax = plt.subplots()
    ax.plot(xs, f0_contour, color="blue")
    ax.set_title("Pitch contour")
    st.pyplot(fig)
    figs["pitch"] = fig

    xs, inten_contour = intensity_contour(intensity)
    fig, ax = plt.subplots()
    ax.plot(xs, inten_contour, color="green")
    ax.set_title("Intensity contour")
    st.pyplot(fig)
    figs["intensity"] = fig

    return snd, pitch, intensity, df, figs


def run_ai_analysis(prefix: str, selected_task: str, df, y_region, sr):
    """Run default AI analysis with Gemini."""
    model, err = init_gemini()
    if err:
        st.session_state[f"{prefix}_ai_gemini_text"] = f" {err}"
        return

    reference_group = st.session_state.get(
        f"{prefix}_gemini_reference_group_{selected_task}", "Unknown / show both"
    )

    # Convert to WAV bytes
    region_temp_path = save_temp_mono_wav(y_region, sr)
    try:
        with open(region_temp_path, "rb") as f:
            region_wav_bytes = f.read()
    finally:
        try:
            os.unlink(region_temp_path)
        except Exception:
            pass

    try:
        with st.spinner("Sending features + audio to Gemini..."):
            st.session_state[f"{prefix}_ai_gemini_text"] = gemini_review_voice_with_audio(
                model=model,
                df_features=df,
                audio_wav_bytes=region_wav_bytes,
                task_name=selected_task,
                reference_group=reference_group,
            )
    except Exception as e:
        st.session_state[f"{prefix}_ai_gemini_text"] = f"Gemini failed: {e}"


def run_byo_analysis(prefix: str, byo_option: str, byo_prompt: str, df, y_region, sr):
    """Run BYO prompt analysis with Gemini."""
    if not byo_prompt or not byo_prompt.strip():
        st.warning("Please enter a custom prompt before running analysis.")
        return

    model, err = init_gemini()
    if err:
        st.session_state[f"{prefix}_byo_gemini_text"] = f" {err}"
        return

    # Prepare audio bytes if needed
    region_wav_bytes = None
    if byo_option in ["Only audio", "Both audio and features"]:
        region_temp_path = save_temp_mono_wav(y_region, sr)
        try:
            with open(region_temp_path, "rb") as f:
                region_wav_bytes = f.read()
        finally:
            try:
                os.unlink(region_temp_path)
            except Exception:
                pass

    try:
        with st.spinner("Sending BYO prompt to Gemini..."):
            st.session_state[f"{prefix}_byo_gemini_text"] = gemini_byo_prompt(
                model=model,
                user_prompt=byo_prompt,
                byo_option=byo_option,
                df_features=df if byo_option in ["Only extracted features", "Both audio and features"] else None,
                audio_wav_bytes=region_wav_bytes,
            )
    except Exception as e:
        st.session_state[f"{prefix}_byo_gemini_text"] = f"Gemini failed: {e}"


def display_gemini_results(prefix: str):
    """Display Gemini responses."""
    if st.session_state[f"{prefix}_ai_gemini_text"]:
        st.subheader("Gemini Response")
        st.markdown(st.session_state[f"{prefix}_ai_gemini_text"])

    if st.session_state[f"{prefix}_byo_gemini_text"]:
        st.subheader("BYO Prompt Response")
        st.markdown(st.session_state[f"{prefix}_byo_gemini_text"])


def display_previous_results(prefix: str):
    """Display results from previous analysis run."""
    if st.session_state[f"{prefix}_ai_df"] is not None:
        st.subheader("Extracted Features (previous run)")
        st.dataframe(st.session_state[f"{prefix}_ai_df"], width="stretch", hide_index=True)

    if st.session_state[f"{prefix}_ai_gemini_text"]:
        st.subheader("Gemini Response (previous run)")
        st.markdown(st.session_state[f"{prefix}_ai_gemini_text"])

    if st.session_state[f"{prefix}_byo_gemini_text"]:
        st.subheader("BYO Prompt Response (previous run)")
        st.markdown(st.session_state[f"{prefix}_byo_gemini_text"])
