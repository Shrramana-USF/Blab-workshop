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
