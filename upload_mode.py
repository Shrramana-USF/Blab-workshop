import streamlit as st
import os
from analysis_utils import *
from streamlit_advanced_audio import audix


def upload_tab(folder_id):
    st.subheader("Upload Audio for Analysis")

    # Initialize session state
    init_session_state("upload")

    # --- Step 1: Task Selection ---
    selected_task = st.radio(
        "Select a task to continue:",
        options=TASKS,
        index=None,
        horizontal=True
    )

    if selected_task is None:
        st.info("Please select a task to continue.")
        return

    st.markdown(f"### Selected Task: {selected_task}")

    # Clear previous results when switching tasks
    if st.session_state.upload_ai_last_task != selected_task:
        clear_session_state("upload")
        st.session_state.upload_ai_last_task = selected_task

    # --- Step 2: File Upload ---
    st.markdown("#### Upload Audio File")

    client = get_box_client()
    task_folder_id = ensure_task_folder(client, folder_id, selected_task)

    uploader_key = f"upload_uploader_{selected_task}"
    save_auto_key = f"upload_save_auto_{selected_task}"

    up = st.file_uploader("Upload audio (WAV only)", type=["wav"], key=uploader_key)
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

    # Display waveform
    temp_path = save_temp_mono_wav(y, sr)
    result = audix(temp_path)
    try:
        os.unlink(temp_path)
    except Exception:
        pass

    st.caption("Trim the audio to analyze a selected portion")
    save_auto = st.checkbox("Save the analysis automatically", key=save_auto_key)

    # --- Step 3: Analysis Mode Selection ---
    analysis_mode = render_analysis_mode_buttons("upload", selected_task)

    # --- BYO Prompt Configuration ---
    byo_option, byo_prompt, should_return = render_byo_config("upload", selected_task, y=y, sr=sr)
    if should_return:
        return  # Chat mode handled

    # Stop if no analysis mode selected
    if analysis_mode is None:
        st.info("Please select an analysis mode to continue.")
        return

    # Reference group selector (AI mode)
    render_reference_group_selector("upload", selected_task)

    # --- Step 4: Run Analysis ---
    analyze_clicked = st.button("Run Analysis", type="primary", key=f"upload_run_analysis_{selected_task}")

    if analyze_clicked:
        y_region, info_msg = get_audio_region(result, y, sr)
        st.info(info_msg)

        if y_region is not None and len(y_region) > 0:
            # BYO "Only audio" - skip PRAAT, just send audio to Gemini
            if analysis_mode == "byo" and byo_option == "Only audio":
                run_byo_analysis("upload", byo_option, byo_prompt, None, y_region, sr)
                display_gemini_results("upload")
                st.toast("Analysis completed.")
            else:
                # Run PRAAT analysis for other modes
                snd, pitch, intensity, df, figs = run_praat_analysis(y_region, sr)

                if df is not None:
                    st.session_state.upload_ai_df = df

                    # Run AI analysis if selected
                    if analysis_mode == "ai":
                        run_ai_analysis("upload", selected_task, df, y_region, sr)

                    # Run BYO analysis if selected (features or both)
                    if analysis_mode == "byo" and byo_option in ["Only extracted features", "Both audio and features"]:
                        run_byo_analysis("upload", byo_option, byo_prompt, df, y_region, sr)

                    # Display Gemini results
                    display_gemini_results("upload")

                    # Save to Box
                    if save_auto:
                        with st.spinner("Saving the analysis", show_time=True):
                            save_analysis_to_box(y_region, sr, df, figs, task_folder_id)
                        st.success("Analyzed and saved results.")
                        st.toast("Analyzed and saved results.")
                    else:
                        st.info("Analysis completed (not saved). Check 'Save automatically' to reanalyze and store results.")
                        st.toast("Analysis completed (not saved).")
    else:
        # Display previous results if any
        display_previous_results("upload")
