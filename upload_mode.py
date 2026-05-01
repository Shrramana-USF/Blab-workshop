import streamlit as st
import os, io, tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parselmouth as pm
import soundfile as sf
import time
from analysis_utils import *
from streamlit_advanced_audio import audix
from PIL import Image
import google.generativeai as genai


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
    model = genai.GenerativeModel("gemini-3-flash-preview")
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

    resp = model.generate_content([prompt, audio_part])
    return resp.text if hasattr(resp, "text") else str(resp)


# ---------------- UPLOAD TAB ----------------
def upload_tab(folder_id):
    st.subheader("Upload Audio for Task")

    # --- Step 1: Task Selection ---
    tasks = [ "Rainbow passage", "Maximum sustained phonation on 'aaah'", "Comfortable sustained phonation on 'eeee'",
             "Glide up to your highest pitch on 'eeee'", "Glide down to your lowest pitch on 'eeee'",
             "Sustained 'aaah' at minimum volume", "Maximum loudness level (brief 'AAAH')", "Conversational speech"]

    selected_task = st.radio(
        "Select a task to continue:",
        options=tasks,
        index=None,
        horizontal=True
    )

    # --- Step 2: Stop here if no task chosen yet ---
    # Stop until a task is chosen
    if selected_task is None:
        st.info("Please select a task to enable uploading.")
        return

    st.markdown(f"### Selected Task: {selected_task}")

    # --- Create / get task folder ---
    client = get_box_client()
    task_folder_id = ensure_task_folder(client, folder_id, selected_task)

    # --- Task-scoped widget keys to force reset when task changes ---
    uploader_key   = f"upload_uploader_{selected_task}"
    save_auto_key  = f"upload_save_auto_{selected_task}"
    analyze_btn_key = f"upload_analyze_{selected_task}"

    # --- Uploader appears only after task is selected ---
    up = st.file_uploader(
        "Upload audio (WAV only)",
        type=["wav"],
        key=uploader_key,        # <-- new widget when task changes -> empty
    )
    if up is None:
        st.info("Upload a WAV file to begin analysis.")
        return

    # Read audio
    raw = up.read()
    try:
        y, sr = read_audio_bytes(raw)
    except Exception:
        st.error("Could not decode this WAV file.")
        return

    st.caption(f"Sample rate: {sr} Hz  ·  Duration: {len(y)/sr:.2f} s")

    # Ensure mono file for waveform widget
    temp_path = save_temp_mono_wav(y, sr)
    result = audix(temp_path)

    # cleanup temp
    try:
        os.unlink(temp_path)
    except Exception:
        pass

    st.caption("Trim the audio to analyse a selected portion")
    save_auto = st.checkbox("Save the analysis automatically", key=save_auto_key)

    # --- Persist AI outputs across reruns ---
    if "upload_ai_df" not in st.session_state:
        st.session_state.upload_ai_df = None
    if "upload_ai_gemini_text" not in st.session_state:
        st.session_state.upload_ai_gemini_text = None
    if "upload_ai_last_task" not in st.session_state:
        st.session_state.upload_ai_last_task = None

    # Clear previous AI results when switching tasks
    if st.session_state.upload_ai_last_task != selected_task:
        st.session_state.upload_ai_df = None
        st.session_state.upload_ai_gemini_text = None
        st.session_state.upload_ai_last_task = selected_task

    # --- Two buttons side by side ---
    col1, col2 = st.columns(2)
    analyze_clicked = col1.button("Analyse Audio", key=analyze_btn_key)
    analyze_ai_clicked = col2.button("Analyse Audio with AI", key=f"upload_analyze_ai_{selected_task}")

    y_region = None

    if analyze_clicked or analyze_ai_clicked:
        if result and result.get("selectedRegion"):
            start = result["selectedRegion"]["start"]
            end = result["selectedRegion"]["end"]
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            y_region = y[start_idx:end_idx]
            st.info(f"Analysing selected region: {start:.2f}s – {end:.2f}s")
        else:
            y_region = y
            st.info("No region selected: Analysing entire file")

    # Quick reference group selector (self-reported; used by AI button)
    st.radio(
        "Reference group for typical ranges (self-reported):",
        options=["Unknown / show both", "Adult male (self-reported)", "Adult female (self-reported)"],
        index=0,
        horizontal=True,
        key=f"upload_gemini_reference_group_{selected_task}",
    )

    if y_region is not None and len(y_region) > 0:
        # Analysis
        snd = pm.Sound(y_region, sampling_frequency=sr)
        pitch = pm.praat.call(snd, "To Pitch (filtered autocorrelation)", 0.0, 30.0, 600.0, 15, "no", 0.03, 0.09, 0.50, 0.055, 0.35, 0.14)
        intensity = snd.to_intensity()

        f0 = estimate_f0_praat(pitch)
        if f0 is None:
            st.warning("No stable fundamental frequency detected.")
        else:
            features = summarize_features(snd, pitch, intensity)
            df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])

            # Always show extracted features after analysis
            st.subheader("Extracted Features")
            st.dataframe(df, width="stretch", hide_index=True)

            # Save features in session_state so they persist after reruns
            st.session_state.upload_ai_df = df

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

            spectrogram = compute_spectrogram(snd)
            fig = plot_spectrogram(spectrogram)
            st.pyplot(fig)
            figs["spectrogram"] = fig

            # If "Analyse Audio with AI" was clicked, call Gemini now and persist response
            if analyze_ai_clicked:
                model, err = init_gemini()
                if err:
                    st.session_state.upload_ai_gemini_text = f" {err}"
                else:
                    reference_group = st.session_state.get(
                        f"upload_gemini_reference_group_{selected_task}", "Unknown / show both"
                    )

                    # Convert analysed region to WAV bytes to send to Gemini
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
                            st.session_state.upload_ai_gemini_text = gemini_review_voice_with_audio(
                                model=model,
                                df_features=df,
                                audio_wav_bytes=region_wav_bytes,
                                task_name=selected_task,
                                reference_group=reference_group,
                            )
                    except Exception as e:
                        st.session_state.upload_ai_gemini_text = f"Gemini failed: {e}"

            # Display persisted Gemini response (if any)
            if st.session_state.upload_ai_gemini_text:
                st.subheader("Gemini Response")
                st.markdown(st.session_state.upload_ai_gemini_text)

            if save_auto:
                with st.spinner("Saving the analysis", show_time=True):
                    # save under user/<task>/session_...
                    save_analysis_to_box(y_region, sr, df, figs, task_folder_id)
                st.success("Analysed and saved results.")
                st.toast("Analysed and saved results.")
            else:
                st.info("Analysis completed (not saved). Check 'Save automatically' to reanalyse and store results.")
                st.toast("Analysis completed (not saved).")
    else:
        # If user previously ran AI and we reran, still show persisted outputs
        if st.session_state.upload_ai_df is not None:
            st.subheader("Extracted Features (previous run)")
            st.dataframe(st.session_state.upload_ai_df, width="stretch", hide_index=True)

        if st.session_state.upload_ai_gemini_text:
            st.subheader("Gemini Response (previous run)")
            st.markdown(st.session_state.upload_ai_gemini_text)
