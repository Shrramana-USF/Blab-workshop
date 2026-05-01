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
    model = genai.GenerativeModel("gemma-4-31b-it")
    # gemini-3-flash-preview
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

    # # DEBUG: Show the prompt being sent
    # with st.expander("🔍 DEBUG: Prompt sent to API", expanded=False):
    #     st.code(prompt, language="text")
    #     st.caption(f"+ Audio WAV attached ({len(audio_wav_bytes)} bytes)")

    resp = model.generate_content([prompt, audio_part])
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
    debug_info = []

    if byo_option == "Only audio":
        full_prompt = f"{user_prompt}\n\n[Audio recording is attached below]"
        content_parts.append(full_prompt)
        debug_info.append(f"Prompt:\n{full_prompt}")
        if audio_wav_bytes:
            audio_part = {
                "inline_data": {
                    "mime_type": "audio/wav",
                    "data": audio_wav_bytes
                }
            }
            content_parts.append(audio_part)
            debug_info.append(f"+ Audio WAV attached ({len(audio_wav_bytes)} bytes)")

    elif byo_option == "Only extracted features":
        rows = df_features.to_dict(orient="records") if df_features is not None else []
        full_prompt = f"{user_prompt}\n\nExtracted acoustic features (Feature, Value): {rows}"
        content_parts.append(full_prompt)
        debug_info.append(f"Prompt:\n{full_prompt}")

    elif byo_option == "Both audio and features":
        rows = df_features.to_dict(orient="records") if df_features is not None else []
        full_prompt = f"{user_prompt}\n\nExtracted acoustic features (Feature, Value): {rows}\n\n[Audio recording is attached below]"
        content_parts.append(full_prompt)
        debug_info.append(f"Prompt:\n{full_prompt}")
        if audio_wav_bytes:
            audio_part = {
                "inline_data": {
                    "mime_type": "audio/wav",
                    "data": audio_wav_bytes
                }
            }
            content_parts.append(audio_part)
            debug_info.append(f"+ Audio WAV attached ({len(audio_wav_bytes)} bytes)")

    else:  # "Just prompt"
        content_parts.append(user_prompt)
        debug_info.append(f"Prompt:\n{user_prompt}")

    # # DEBUG: Show the prompt being sent
    # with st.expander("🔍 DEBUG: BYO Prompt sent to API", expanded=False):
    #     st.code("\n".join(debug_info), language="text")
    #     st.caption(f"Mode: {byo_option}")

    resp = model.generate_content(content_parts)
    return resp.text if hasattr(resp, "text") else str(resp)


def record_tab(folder_id):
    # --- Task selection ---
    st.subheader("Record Audio for Analysis")

    # --- Persist AI outputs across reruns ---
    if "record_ai_df" not in st.session_state:
        st.session_state.record_ai_df = None
    if "record_ai_gemini_text" not in st.session_state:
        st.session_state.record_ai_gemini_text = None
    if "record_ai_last_task" not in st.session_state:
        st.session_state.record_ai_last_task = None
    # BYO prompt session state
    if "record_byo_gemini_text" not in st.session_state:
        st.session_state.record_byo_gemini_text = None
    if "record_byo_mode_active" not in st.session_state:
        st.session_state.record_byo_mode_active = False
    # Chat conversation state for "Just prompt" mode
    if "record_byo_chat_history" not in st.session_state:
        st.session_state.record_byo_chat_history = []
    # Track selected analysis mode
    if "record_analysis_mode" not in st.session_state:
        st.session_state.record_analysis_mode = None

    tasks = [
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
        # Clear previous results when switching tasks
        st.session_state.record_ai_df = None
        st.session_state.record_ai_gemini_text = None
        st.session_state.record_byo_gemini_text = None
        st.session_state.record_byo_mode_active = False
        st.session_state.record_byo_chat_history = []
        st.session_state.record_analysis_mode = None
        st.rerun()

    if selected_task is None:
        st.info("Please select a task to continue.")
        return

    st.markdown(f"### Selected Task: {selected_task}")

    # --- Step 2: Analysis Mode Selection ---
    st.markdown("#### Select Analysis Mode")
    col1, col2, col3 = st.columns(3)

    if col1.button("Analyse Audio", key="record_analyze", use_container_width=True):
        st.session_state.record_analysis_mode = "praat"
        st.session_state.record_byo_mode_active = False

    if col2.button("Default Analysis with AI", key="record_analyze_ai", use_container_width=True):
        st.session_state.record_analysis_mode = "ai"
        st.session_state.record_byo_mode_active = False

    if col3.button("BYO Prompt", key="record_analyze_byo", use_container_width=True):
        st.session_state.record_analysis_mode = "byo"
        st.session_state.record_byo_mode_active = True

    # Show current selection
    mode_labels = {
        "praat": "PRAAT Analysis Only",
        "ai": "Default Analysis with AI",
        "byo": "BYO Prompt"
    }

    if st.session_state.record_analysis_mode:
        st.success(f"Selected: **{mode_labels[st.session_state.record_analysis_mode]}**")

    # --- BYO Prompt Configuration (if BYO mode selected) ---
    byo_option = None
    byo_prompt = ""
    byo_submit_clicked = False

    if st.session_state.record_byo_mode_active:
        st.markdown("---")
        st.markdown("#### BYO Prompt Configuration")

        byo_option = st.radio(
            "What to send to AI:",
            options=["Only audio", "Only extracted features", "Both audio and features", "Just prompt"],
            index=2,  # Default to "Both audio and features"
            horizontal=True,
            key="record_byo_option",
        )

        # Different UI for "Just prompt" - conversation mode
        if byo_option == "Just prompt":
            st.markdown("##### Conversation Mode")
            st.caption("Have a back-and-forth conversation with Gemini")

            # Display chat history
            for msg in st.session_state.record_byo_chat_history:
                if msg["role"] == "user":
                    st.chat_message("user").markdown(msg["content"])
                else:
                    st.chat_message("assistant").markdown(msg["content"])

            # Chat input
            byo_prompt = st.chat_input("Type your message...", key="record_byo_chat_input")
            byo_submit_clicked = byo_prompt is not None and byo_prompt.strip() != ""

            # Clear conversation button
            if st.session_state.record_byo_chat_history:
                if st.button("Clear Conversation", key="record_byo_clear_chat"):
                    st.session_state.record_byo_chat_history = []
                    st.rerun()

            # Handle "Just prompt" conversation - no recording needed
            if byo_submit_clicked:
                model, err = init_gemini()
                if err:
                    st.error(f" {err}")
                else:
                    try:
                        # Build history for chat session from stored history
                        gemini_history = []
                        for msg in st.session_state.record_byo_chat_history:
                            gemini_history.append({
                                "role": msg["role"] if msg["role"] == "user" else "model",
                                "parts": [msg["content"]]
                            })

                        # # DEBUG: Show conversation context being sent
                        # with st.expander("🔍 DEBUG: Chat API Call", expanded=False):
                        #     st.write("**History being sent:**")
                        #     if gemini_history:
                        #         for i, h in enumerate(gemini_history):
                        #             st.text(f"[{i}] {h['role']}: {h['parts'][0][:100]}...")
                        #     else:
                        #         st.text("(No history - new conversation)")
                        #     st.write("**New message:**")
                        #     st.code(byo_prompt, language="text")

                        # Create chat session with history
                        chat = model.start_chat(history=gemini_history)

                        # Add user message to our history
                        st.session_state.record_byo_chat_history.append({
                            "role": "user",
                            "content": byo_prompt
                        })

                        # Send message and get response
                        with st.spinner("Gemini is thinking..."):
                            response = chat.send_message(byo_prompt)
                            response_text = response.text if hasattr(response, "text") else str(response)

                        # Add assistant response to history
                        st.session_state.record_byo_chat_history.append({
                            "role": "assistant",
                            "content": response_text
                        })

                        # Rerun to show updated chat
                        st.rerun()

                    except Exception as e:
                        st.error(f"Gemini failed: {e}")

            st.markdown("---")
            return  # "Just prompt" doesn't need recording
        else:
            # Standard text area for other options
            byo_prompt = st.text_area(
                "Enter your custom prompt:",
                placeholder="",
                height=150,
                key="record_byo_prompt",
            )

        st.markdown("---")

    # Stop if no analysis mode selected
    if st.session_state.record_analysis_mode is None:
        st.info("Please select an analysis mode to continue.")
        return

    # --- Step 3: Record Audio ---
    st.markdown("#### Record Audio")

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

            # Reference group selector (used by AI mode)
            if st.session_state.record_analysis_mode == "ai":
                st.radio(
                    "Reference group for typical ranges (self-reported):",
                    options=["Unknown / show both", "Adult male (self-reported)", "Adult female (self-reported)"],
                    index=0,
                    horizontal=True,
                    key="record_gemini_reference_group",
                )

            # --- Step 4: Run Analysis Button ---
            analyze_clicked = st.button("Run Analysis", type="primary", key="record_run_analysis")

            y_region = None

            if analyze_clicked:
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

            if y_region is not None and len(y_region) > 0:
                # PRAAT Analysis
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
                    st.session_state.record_ai_df = df

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

                    # If AI mode, call Gemini
                    if st.session_state.record_analysis_mode == "ai":
                        model, err = init_gemini()
                        if err:
                            st.session_state.record_ai_gemini_text = f" {err}"
                        else:
                            reference_group = st.session_state.get(
                                "record_gemini_reference_group", "Unknown / show both"
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
                                    st.session_state.record_ai_gemini_text = gemini_review_voice_with_audio(
                                        model=model,
                                        df_features=df,
                                        audio_wav_bytes=region_wav_bytes,
                                        task_name=selected_task,
                                        reference_group=reference_group,
                                    )
                            except Exception as e:
                                st.session_state.record_ai_gemini_text = f"Gemini failed: {e}"

                    # If BYO mode with audio/features options
                    if st.session_state.record_analysis_mode == "byo" and byo_option != "Just prompt":
                        if not byo_prompt or not byo_prompt.strip():
                            st.warning("Please enter a custom prompt before running analysis.")
                        else:
                            model, err = init_gemini()
                            if err:
                                st.session_state.record_byo_gemini_text = f" {err}"
                            else:
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
                                        st.session_state.record_byo_gemini_text = gemini_byo_prompt(
                                            model=model,
                                            user_prompt=byo_prompt,
                                            byo_option=byo_option,
                                            df_features=df if byo_option in ["Only extracted features", "Both audio and features"] else None,
                                            audio_wav_bytes=region_wav_bytes,
                                        )
                                except Exception as e:
                                    st.session_state.record_byo_gemini_text = f"Gemini failed: {e}"

                    # Display persisted Gemini response (if any)
                    if st.session_state.record_ai_gemini_text:
                        st.subheader("Gemini Response")
                        st.markdown(st.session_state.record_ai_gemini_text)

                    # Display persisted BYO Gemini response (if any)
                    if st.session_state.record_byo_gemini_text:
                        st.subheader("BYO Prompt Response")
                        st.markdown(st.session_state.record_byo_gemini_text)

                    if save_auto:
                        with st.spinner("Saving the analysis", show_time=True):
                            save_analysis_to_box(y_region, sr, df, figs, folder_id)
                        st.success("Analysed and Saved results")
                        st.toast("Analysed and saved results.")
                    else:
                        st.info("Analysis completed (not saved). Check 'Save automatically' to reanalyse and store the results.")
                        st.toast("Analysis completed (not saved).")
            else:
                # If user previously ran AI and we reran, still show persisted outputs
                if st.session_state.record_ai_df is not None:
                    st.subheader("Extracted Features (previous run)")
                    st.dataframe(st.session_state.record_ai_df, width="stretch", hide_index=True)

                if st.session_state.record_ai_gemini_text:
                    st.subheader("Gemini Response (previous run)")
                    st.markdown(st.session_state.record_ai_gemini_text)

                if st.session_state.record_byo_gemini_text:
                    st.subheader("BYO Prompt Response (previous run)")
                    st.markdown(st.session_state.record_byo_gemini_text)
