import streamlit as st
import os
import parselmouth as pm
import matplotlib.pyplot as plt
import pandas as pd
from st_audiorec import st_audiorec
from streamlit_advanced_audio import audix
from analysis_utils import *
import io
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


# (Kept for compatibility; not used for Gemini audio mode)
def fig_to_pil_image(fig):
    """
    Convert a Matplotlib figure to a PIL Image (PNG).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


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


def record_tab(folder_id):
    # --- Task selection (added) ---
    st.subheader("Record Audio for Task")
    tasks = [
        "Rainbow passage",
        "Maximum sustained phonation on 'aaah'",
        "Comfortable sustained phonation on 'eeee'",
        "Glide up to your highest pitch on 'eeee'",
        "Glide down to your lowest pitch on 'eeee'",
        "Sustained 'aaah' at minimum volume",
        "Maximum loudness level (brief 'AAAH')",
        "Conversational speech"
    ]
    selected_task = st.radio(
        "Select a task to continue:",
        options=tasks,
        index=None,  # none pre-selected
        horizontal=True,
        key="record_task_radio"
    )

    # --- Reset UI and reload recorder when switching tasks ---
    if "prev_task_record" not in st.session_state:
        st.session_state.prev_task_record = None
    if selected_task != st.session_state.prev_task_record:
        st.session_state.prev_task_record = selected_task
        st.session_state.recorder_reload_key = f"recorder_{selected_task}"
        st.rerun()

    if selected_task is None:
        st.info("Please select a task to enable recording.")
        return

    # --- Ensure Box subfolder for selected task ---
    client = get_box_client()
    folder_id = ensure_task_folder(client, folder_id, selected_task)

    # ---------------- RECORD MODE ----------------
    st.caption("Click to record, then stop. The widget shows a waveform while recording.")
    recorder_key = st.session_state.get("recorder_reload_key", f"recorder_{selected_task}")
    wav_audio_data = st_audiorec()  # cannot take key argument; reload handled via rerun

    if wav_audio_data is not None:
        try:
            y, sr = read_audio_bytes(wav_audio_data)
        except Exception:
            st.error("Couldn't parse recorded WAV. Try again.")
            y, sr = None, None

        if y is not None:
            st.caption(f"Sample rate: {sr} Hz  ·  Duration: {len(y)/sr:.2f} s")

            temp_path = save_temp_mono_wav(y, sr)
            result = audix(temp_path)

            try:
                os.unlink(temp_path)
            except Exception:
                pass

            st.caption("Trim the audio to analyse a selected portion")
            save_auto = st.checkbox("Save the analysis automatically", key="record_save_auto")

            # --- Persist AI outputs across reruns ---
            if "ai_df" not in st.session_state:
                st.session_state.ai_df = None
            if "ai_gemini_text" not in st.session_state:
                st.session_state.ai_gemini_text = None
            if "ai_last_task" not in st.session_state:
                st.session_state.ai_last_task = None

            # Clear previous AI results when switching tasks
            if st.session_state.ai_last_task != selected_task:
                st.session_state.ai_df = None
                st.session_state.ai_gemini_text = None
                st.session_state.ai_last_task = selected_task

            # --- Two buttons side by side ---
            col1, col2 = st.columns(2)
            analyze_clicked = col1.button("Analyse Audio", key="record_analyze")
            analyze_ai_clicked = col2.button("Analyse Audio with AI", key="record_analyze_ai")

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
                key="gemini_reference_group_quick",
            )

            if y_region is not None and len(y_region) > 0:
                snd = pm.Sound(y_region, sampling_frequency=sr)
                pitch = snd.to_pitch(time_step=None, pitch_floor=30, pitch_ceiling=600)
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
                    st.session_state.ai_df = df

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
                            st.session_state.ai_gemini_text = f" {err}"
                        else:
                            reference_group = st.session_state.get(
                                "gemini_reference_group_quick", "Unknown / show both"
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
                                    st.session_state.ai_gemini_text = gemini_review_voice_with_audio(
                                        model=model,
                                        df_features=df,
                                        audio_wav_bytes=region_wav_bytes,
                                        task_name=selected_task,
                                        reference_group=reference_group,
                                    )
                            except Exception as e:
                                st.session_state.ai_gemini_text = f"Gemini failed: {e}"

                    # Display persisted Gemini response (if any)
                    if st.session_state.ai_gemini_text:
                        st.subheader("Gemini Response")
                        st.markdown(st.session_state.ai_gemini_text)

                    if save_auto:
                        with st.spinner("Saving the analysis", show_time=True):
                            save_analysis_to_box(y_region, sr, df, figs, folder_id)
                        st.success("Analysed and Saved results")
                    else:
                        st.info("Analysis completed (not saved). Check 'Save automatically' to reanalyse and store the results.")
            else:
                # If user previously ran AI and we reran, still show persisted outputs
                if st.session_state.ai_df is not None:
                    st.subheader("Extracted Features (previous run)")
                    st.dataframe(st.session_state.ai_df, width="stretch", hide_index=True)

                if st.session_state.ai_gemini_text:
                    st.subheader("Gemini Response (previous run)")
                    st.markdown(st.session_state.ai_gemini_text)
