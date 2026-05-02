import streamlit as st
import os
from st_audiorec import st_audiorec
from streamlit_advanced_audio import audix
from analysis_utils import *


def record_tab(folder_id):
    st.subheader("Record Audio for Analysis")

    # Initialize session state
    init_session_state("record")

    # --- Step 1: Task Selection ---
    selected_task = st.radio(
        "Select a task to continue:",
        options=TASKS,
        index=None,
        horizontal=True,
        key="record_task_radio"
    )

    # Reset UI when switching tasks
    if "prev_task_record" not in st.session_state:
        st.session_state.prev_task_record = None
    if selected_task != st.session_state.prev_task_record:
        st.session_state.prev_task_record = selected_task
        st.session_state.recorder_reload_key = f"recorder_{selected_task}"
        clear_session_state("record")
        if selected_task is not None:
            st.rerun()

    if selected_task is None:
        st.info("Please select a task to continue.")
        return

    st.markdown(f"### Selected Task: {selected_task}")

    # --- Step 2: Record Audio ---
    st.markdown("#### Record Audio")

    client = get_box_client()
    task_folder_id = ensure_task_folder(client, folder_id, selected_task)

    st.caption("Click to record, then stop. The widget shows a waveform while recording.")
    wav_audio_data = st_audiorec()

    if wav_audio_data is None:
        return

    try:
        y, sr = read_audio_bytes(wav_audio_data)
    except Exception:
        st.error("Couldn't parse recorded WAV. Try again.")
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
    save_auto = st.checkbox("Save the analysis automatically", key="record_save_auto")

    # --- Step 3: Analysis Mode Selection ---
    analysis_mode = render_analysis_mode_buttons("record", selected_task)

    # --- BYO Prompt Configuration ---
    byo_option, byo_prompt, should_return = render_byo_config("record", selected_task)
    if should_return:
        return  # "Just prompt" mode handled

    # Stop if no analysis mode selected
    if analysis_mode is None:
        st.info("Please select an analysis mode to continue.")
        return

    # Reference group selector (AI mode)
    render_reference_group_selector("record", selected_task)

    # --- Step 4: Run Analysis ---
    analyze_clicked = st.button("Run Analysis", type="primary", key="record_run_analysis")

    if analyze_clicked:
        y_region, info_msg = get_audio_region(result, y, sr)
        st.info(info_msg)

        if y_region is not None and len(y_region) > 0:
            # Run PRAAT analysis
            snd, pitch, intensity, df, figs = run_praat_analysis(y_region, sr)

            if df is not None:
                st.session_state.record_ai_df = df

                # Run AI analysis if selected
                if analysis_mode == "ai":
                    run_ai_analysis("record", selected_task, df, y_region, sr)

                # Run BYO analysis if selected
                if analysis_mode == "byo" and byo_option != "Just prompt":
                    run_byo_analysis("record", byo_option, byo_prompt, df, y_region, sr)

                # Display Gemini results
                display_gemini_results("record")

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
        display_previous_results("record")
