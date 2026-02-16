"""Streamlit UI helper functions to keep app.py lean."""

import gc

import pandas as pd
import streamlit as st

from config import HANZI_ONLY_RE


def sanitize_dataframe_name(name):
    sanitized = name.replace(",", "").strip()
    return sanitized or "Unnamed text"


def init_session_state(state):
    defaults = {
        "word_results_dict": {},
        "char_results_dict": {},
        "custom_vocab_set": None,
        "uploader_key": 0,
        "custom_vocab_filename": None,
        "vocab_uploader_key": 0,
        "show_vocab_warning": None,
        "upload_status_toast": None,
    }
    for key, value in defaults.items():
        if key not in state:
            state[key] = value.copy() if isinstance(value, dict) else value

def show_custom_vocab_section(
    state,
    clear_vocab_cache,
    get_tokenizer,
    analyse_text,
    resolve_vocab,
    resolve_hsk_level,
    resolve_topn_rank,
):
    results_exist = bool(state.word_results_dict)

    st.subheader("OPTIONAL: Custom vocabulary list")
    custom_vocab_file = st.file_uploader(
        "Upload known vocab list (.txt, one word per line)",
        type=["txt"],
        key=f"vocab_{state.vocab_uploader_key}",
        disabled=results_exist,
    )
    st.caption(
        "Lines containing non-Hanzi characters (including Pleco category headers such as //) are ignored."
    )

    if custom_vocab_file:
        has_file_results = any(
            not key.startswith("(Custom vocab)") for key in state.word_results_dict
        )
        if has_file_results:
            state.show_vocab_warning = "Clear all analyses before loading or replacing a custom vocab list."
            state.vocab_uploader_key += 1
            st.rerun()

        vocab_text = custom_vocab_file.read().decode("utf-8-sig", errors="ignore")
        cleaned = []
        removed_lines = []
        for raw_line in vocab_text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if HANZI_ONLY_RE.match(stripped):
                cleaned.append(stripped)
            else:
                removed_lines.append(stripped)
        cleaned_unique = sorted(set(cleaned))
        vocab_set = set(cleaned_unique)

        toast_level = "success"
        toast_message = f"Custom vocab file {custom_vocab_file.name} uploaded."
        if removed_lines:
            state.show_vocab_warning = f"{len(removed_lines)} lines removed: {', '.join(removed_lines)}"

        clear_vocab_cache(state.custom_vocab_set)
        state.custom_vocab_set = vocab_set
        state.custom_vocab_filename = custom_vocab_file.name
        state.upload_status_toast = (toast_level, toast_message)

        tokenizer = get_tokenizer(state.custom_vocab_set)

        if not state.word_results_dict:
            word_stats, char_stats = analyse_text(
                "\n".join(cleaned_unique),
                tokenizer,
                vocab_set,
                resolve_vocab,
                resolve_hsk_level,
                resolve_topn_rank,
            )
            row_name = sanitize_dataframe_name(f"(Custom vocab) {custom_vocab_file.name}")
            state.word_results_dict[row_name] = word_stats
            state.char_results_dict[row_name] = char_stats

        state.vocab_uploader_key += 1
        st.rerun()

    custom_vocab_set = state.custom_vocab_set
    has_file_results = any(
        not key.startswith("(Custom vocab)") for key in state.word_results_dict
    )
    if custom_vocab_set:
        st.caption(
            f"{len(custom_vocab_set):,} vocab items loaded • use Clear All to remove or replace this list"
        )
    elif has_file_results:
        st.caption(
            "No custom vocab list currently loaded • no custom vocab can be uploaded after files are analysed; use Clear All to upload custom vocab"
        )
    else:
        st.caption("No custom vocab list currently loaded")

    return custom_vocab_set


def show_input_section(
    state,
    get_available_name,
    get_tokenizer,
    analyse_text,
    resolve_vocab,
    resolve_hsk_level,
    resolve_topn_rank,
    extract_text_from_epub,
    extract_text_from_pdf,
):
    st.divider()
    st.header("Files for analysis")
    input_method = st.radio(
        "Input method",
        ["Upload file(s)", "Upload folder", "Paste text"],
        horizontal=True,
    )

    if input_method == "Paste text":
        pasted_text_name = st.text_input(
            "Text name",
            placeholder='e.g. "Chinese article", "Chapter 1"',
            key="pasted_text_name",
        )
        pasted_text = st.text_area(
            "Paste Chinese text",
            max_chars=15000,
            height=180,
            key="pasted_chinese_text",
        )

        if st.button("Upload text"):
            stripped_name = pasted_text_name.strip()
            effective_name = sanitize_dataframe_name(stripped_name or "Pasted text")

            if not pasted_text.strip():
                st.warning("Please paste Chinese text before uploading.")
            elif len(pasted_text) > 15000:
                st.error(
                    "This text exceeds the 15,000 character limit. Please save it as a .txt file and upload it instead."
                )
            else:
                existing_names = set(state.word_results_dict.keys())
                final_name = get_available_name(effective_name, existing_names)

                progress_bar = st.progress(0, text="Preparing text upload...")
                tokenizer = get_tokenizer(state.custom_vocab_set)
                progress_bar.progress(40, text=f"Analysing text: {final_name}")
                word_stats, char_stats = analyse_text(
                    pasted_text,
                    tokenizer,
                    state.custom_vocab_set,
                    resolve_vocab,
                    resolve_hsk_level,
                    resolve_topn_rank,
                )
                progress_bar.progress(85, text=f"Saving analysis: {final_name}")
                state.word_results_dict[final_name] = word_stats
                state.char_results_dict[final_name] = char_stats
                progress_bar.progress(100, text="Upload complete!")

                if not stripped_name:
                    state.upload_status_toast = (
                        "warning",
                        f"No name provided. Using default name '{final_name}'.",
                    )
                elif final_name != effective_name:
                    state.upload_status_toast = (
                        "warning",
                        f"Name already existed. Using '{final_name}' instead.",
                    )
                else:
                    state.upload_status_toast = (
                        "success",
                        f"Uploaded as '{final_name}'.",
                    )

                st.rerun()

    if input_method in ["Upload file(s)", "Upload folder"]:
        accepts = True if input_method == "Upload file(s)" else "directory"
        upload_label = (
            "Upload one or more text files"
            if input_method == "Upload file(s)"
            else "Upload a folder of text files"
        )
        uploaded_files = st.file_uploader(
            upload_label,
            type=["txt", "epub", "pdf", "csv"],
            accept_multiple_files=accepts,
            key=f"files_{state.uploader_key}",
        )

        if uploaded_files:
            tokenizer = get_tokenizer(state.custom_vocab_set)
            new_files = uploaded_files

            if not new_files:
                st.info("All selected files are already loaded.")
            else:
                total_files = len(new_files)
                uploaded_names = [file.name for file in new_files]
                progress_bar = st.progress(0, text="Starting analysis...")

                for i, file in enumerate(new_files):
                    progress_bar.progress(i / total_files, text=f"Reading file: {file.name}")

                    if file.name.endswith((".txt", ".csv")):
                        text = file.read().decode("utf-8", errors="ignore")
                    elif file.name.endswith(".epub"):
                        text = extract_text_from_epub(file)
                    elif file.name.endswith(".pdf"):
                        text = extract_text_from_pdf(file)
                    else:
                        continue

                    progress_bar.progress(
                        (i + 0.5) / total_files, text=f"Analysing file: {file.name}"
                    )

                    word_stats, char_stats = analyse_text(
                        text,
                        tokenizer,
                        state.custom_vocab_set,
                        resolve_vocab,
                        resolve_hsk_level,
                        resolve_topn_rank,
                    )

                    existing_names = set(state.word_results_dict.keys())
                    cleaned_name = sanitize_dataframe_name(file.name)
                    result_name = get_available_name(cleaned_name, existing_names)

                    state.word_results_dict[result_name] = word_stats
                    state.char_results_dict[result_name] = char_stats
                    text = None
                    gc.collect()

                    progress_bar.progress(
                        (i + 1) / total_files, text=f"Analysis complete: {file.name}"
                    )

                progress_bar.progress(1.0, text="All analyses complete!")

                if total_files == 1:
                    state.upload_status_toast = (
                        "success",
                        f"Uploaded file: {uploaded_names[0]}",
                    )
                else:
                    state.upload_status_toast = (
                        "success",
                        f"Uploaded {total_files} files: {', '.join(uploaded_names)}",
                    )

                state.uploader_key += 1
                st.rerun()


def manage_loaded_files(state):
    if state.word_results_dict:
        with st.expander("Manage loaded files"):
            for fname in list(state.word_results_dict.keys()):
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.write(f"📄 {fname}")
                with col2:
                    if fname.startswith("(Custom vocab)"):
                        st.caption("Use Clear All to remove")
                    elif st.button("Remove", key=f"remove_{fname}"):
                        state.word_results_dict.pop(fname, None)
                        state.char_results_dict.pop(fname, None)
                        state.uploader_key += 1
                        st.rerun()


def show_export_and_clear(state, clear_vocab_cache):
    word_df = None
    char_df = None
    st.divider()
    clear_clicked = False

    if state.word_results_dict:
        def format_list_for_csv(entries):
            if not isinstance(entries, list):
                return entries
            if entries and all(
                isinstance(item, (tuple, list)) and len(item) == 2 for item in entries
            ):
                return "; ".join(f"{item[0]}:{item[1]}" for item in entries)
            return "; ".join(str(item) for item in entries)

        def to_download_df(data_dict):
            df = pd.DataFrame.from_dict(data_dict, orient="index")
            df.index.name = "Text source file"
            return df

        def columns_for_group(df, group_name):
            group_matchers = {
                "Core metrics": lambda col: not any(
                    [
                        col.startswith("Top "),
                        col.startswith("HSK "),
                        "(with frequencies)" in col,
                        "extract" in col.lower(),
                    ]
                ),
                "Top N coverage": lambda col: col.startswith("Top "),
                "HSK coverage": lambda col: col.startswith("HSK "),
                "Frequency lists": lambda col: "(with frequencies)" in col,
                "1000-token text extract": lambda col: "extract" in col.lower(),
            }
            matcher = group_matchers[group_name]
            return [col for col in df.columns if matcher(col)]

        def build_combined_export_df(word_df, char_df, selected_options):
            ordered_option_groups = [
                ("Core metrics (words/tokens)", word_df, "Core metrics"),
                ("Core metrics (chars)", char_df, "Core metrics"),
                ("Top N coverage (words/tokens)", word_df, "Top N coverage"),
                ("Top N coverage (chars)", char_df, "Top N coverage"),
                ("HSK coverage (words/tokens)", word_df, "HSK coverage"),
                ("HSK coverage (chars)", char_df, "HSK coverage"),
                ("Frequency lists (words/tokens)", word_df, "Frequency lists"),
                ("Frequency lists (chars)", char_df, "Frequency lists"),
                ("1000-token text extract", word_df, "1000-token text extract"),
            ]

            selected_frames = []
            for option_label, source_df, group_name in ordered_option_groups:
                if not selected_options.get(option_label, False):
                    continue
                group_columns = columns_for_group(source_df, group_name)
                if group_columns:
                    selected_frames.append(source_df[group_columns])

            if selected_frames:
                export_df = pd.concat(selected_frames, axis=1)
            else:
                export_df = pd.DataFrame(index=word_df.index)

            for col in export_df.columns:
                if export_df[col].apply(lambda v: isinstance(v, list)).any():
                    export_df[col] = export_df[col].apply(format_list_for_csv)

            export_df.index.name = "Text source file"
            return export_df

        word_df_for_export = to_download_df(state.word_results_dict)
        char_df_for_export = to_download_df(state.char_results_dict)

        col1, col2 = st.columns(2, vertical_alignment="bottom")
        with col1:
            clear_clicked = st.button("Clear All", width="stretch", type="secondary")
        with col2:
            with st.popover("Configure CSV Export", width="stretch"):
                export_options = [
                    "Core metrics (words/tokens)",
                    "Core metrics (chars)",
                    "Top N coverage (words/tokens)",
                    "Top N coverage (chars)",
                    "HSK coverage (words/tokens)",
                    "HSK coverage (chars)",
                    "Frequency lists (words/tokens)",
                    "Frequency lists (chars)",
                    "1000-token text extract",
                ]
                default_option_state = {
                    "Frequency lists (words/tokens)": False,
                    "Frequency lists (chars)": False,
                    "1000-token text extract": False,
                }

                st.caption("Choose which sections to include in your CSV export.")
                selected_options = {
                    option: st.checkbox(
                        option,
                        value=default_option_state.get(option, True),
                        key=f"csv_export_option_{option}",
                    )
                    for option in export_options
                }

                combined_export_df = build_combined_export_df(
                    word_df_for_export,
                    char_df_for_export,
                    selected_options,
                )

                csv_bytes = combined_export_df.to_csv(index=True).encode("utf-8-sig")

                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name="chinese-text-lexical-analysis.csv",
                    mime="text/csv",
                    width="stretch",
                    type="secondary",
                    key="combined_csv_download",
                )

    if clear_clicked:
        state.word_results_dict = {}
        state.char_results_dict = {}
        clear_vocab_cache(state.custom_vocab_set)
        state.custom_vocab_set = None
        state.custom_vocab_filename = None
        state.pop("pasted_chinese_text", None)
        state.pop("pasted_text_name", None)
        state.pop("demo_text", None)
        state.uploader_key += 1
        state.vocab_uploader_key += 1
        state.upload_status_toast = (
            "success",
            "All files and custom vocab have been cleared.",
        )
        st.rerun()

    if state.word_results_dict:
        word_df = pd.DataFrame.from_dict(state.word_results_dict, orient="index")
        char_df = pd.DataFrame.from_dict(state.char_results_dict, orient="index")
        word_df.index.name = "Text source file"
        char_df.index.name = "Text source file"

    return word_df, char_df
