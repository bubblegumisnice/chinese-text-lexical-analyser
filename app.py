# -*- coding: utf-8 -*-
"""
Chinese text lexical analyser app
"""

import csv
import sys
import math
import numbers
import re
import streamlit as st
import pandas as pd
import urllib.parse

# Ignore harmless Jieba syntax warnings
import warnings

warnings.filterwarnings(
    "ignore",
    message="invalid escape sequence",
    category=SyntaxWarning,
)

sys.modules.setdefault("app", sys.modules[__name__])

from config import (
    WORD_FREQ_PATH,
    HSK_PATH,
    WINDOW_SIZE,
    STEP_SIZE,
    MAX_WINDOWS,
    DEFAULT_DEMO_TEXT,
    MODE_EXPLANATIONS,
    HANZI_RE,
    HANZI_ONLY_RE,
    SENTENCE_SPLIT_RE,
)

from tokenizers import get_tokenizer
from analysis import analyse_text, extract_text_from_epub, extract_text_from_pdf
from rendering import (
    render_highlighted_text,
    render_legend,
    render_sentence_segmented_tokens_html,
)
from ui_handlers import (
    init_session_state,
    show_custom_vocab_section,
    show_input_section,
    show_export_and_clear,
)

# Global caches
VOCAB_RESOLVE_CACHE = {}
HSK_RESOLVE_CACHE = {}
TOPN_RESOLVE_CACHE = {}


# =========================================================

st.set_page_config(page_title="Chinese Text Lexical Analyser", layout="wide")
st.title("Chinese Text Lexical Analyser")
st.caption("Frequency coverage • HSK coverage • Lexical diversity • Sentence statistics")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("About this tool")

st.sidebar.markdown("""
## What it does

This tool takes one or more **.txt files** and analyses the Chinese text inside them.

- Only Chinese characters (Hanzi) are analysed.
- English words, numbers, and punctuation are ignored.
- This makes it suitable for **graded readers** that include inline English translations.

Each file is analysed independently, and results are displayed in a comparative table and interactive charts.
""")

st.sidebar.markdown("""
## Data sources

### HSK vocabulary

Uses the **latest official HSK 3.0 word list (December 2025 version)**, found [here](https://www.marteagency.com/pdf/%E6%96%B0%E7%89%88HSK%E8%80%83%E8%AF%95%E5%A4%A7%E7%BA%B2%E8%AF%8D%E6%B1%87.pdf).

Coverage is calculated cumulatively:
- HSK 1
- HSK 1–2
- HSK 1–3
- ...
- HSK 1–7–9

This shows what proportion of the unique vocabulary in a text is covered up to each level.
""")

st.sidebar.markdown("""
### Word frequency data

Uses the open-source [**WordFreq**](https://github.com/rspeer/wordfreq) database.
This is a snapshot of language usage up to approximately 2021 (largely untainted by AI-generated content).

It combines:

- Wikipedia (encyclopedic text)
- Subtitles (OPUS OpenSubtitles 2018 + SUBTLEX)
- News (NewsCrawl 2014 + GlobalVoices)
- Books (Google Books Ngrams 2012)
- Web text (OSCAR)
- Twitter (short-form social media)
- Miscellaneous word frequencies: a free wordlist that comes with the Jieba word segmenter.
""")

st.sidebar.markdown("""
## Metrics explained

### Total tokens
Total number of segmented Chinese word tokens in the text.

### Unique words
Number of distinct Chinese words.

### Total characters
Total number of Hanzi characters.

### Unique characters
Number of distinct Hanzi characters.

### Average tokens per sentence
Mean number of Chinese word tokens per sentence.

### Phrasing variety (per 1000 tokens)
Median number of unique words in sliding 1000-token windows.  
This gives a stable measure of lexical diversity that is less sensitive to text length.

### Top-N frequency coverage
Cumulative percentage of unique words that fall within the top:
- 1k most frequent words
- 2k
- ...
- 10k

Based on WordFreq rankings.

### HSK cumulative coverage
Cumulative percentage of unique words covered by:
- HSK 1
- HSK 1 to 2
- ...
- HSK 1 to 7–9

Based on the official HSK 3.0 word list.
""")

st.sidebar.markdown("""
---

                word_sections.append(("Not in Custom Vocab", wrow.get("Words not in vocab (with frequencies)", [])))
- Graded readers  
- Web novels  
                char_sections.append(("Not in Custom Vocab", crow.get("Characters not in vocab (with frequencies)", [])))
- Learning materials  

                options = [x for x in options if "vocab" not in x[0].lower()]
""")


# =========================================================
# LOAD WORD FREQUENCY LIST (RANK-AWARE)
# =========================================================
@st.cache_data
def load_wordfreq_with_rank(path):
    rank = {}
    current_rank = 1

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["Word"].strip()
            if HANZI_ONLY_RE.match(word):
                if word not in rank:
                    rank[word] = current_rank
                    current_rank += 1
    return rank


# =========================================================
# LOAD HSK LIST (LEVEL-AWARE)
# =========================================================
@st.cache_data
def load_hsk(path):
    hsk = {}
    current_level = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("//"):
                level_str = line.split("HSK")[-1]
                if level_str == "7-9":
                    current_level = 7
                else:
                    current_level = int(level_str)
                continue

            word = line.split(",")[0].strip()
            if HANZI_ONLY_RE.match(word):
                if word not in hsk or current_level < hsk[word]:
                    hsk[word] = current_level
    return hsk


@st.cache_data
def load_core_lexical_data(word_freq_path, hsk_path):
    word_rank = load_wordfreq_with_rank(word_freq_path)
    hsk_map = load_hsk(hsk_path)
    segmentation_words = set(word_rank.keys()) | set(hsk_map.keys())  # words only; ignore rank/level values
    return word_rank, hsk_map, segmentation_words


WORD_RANK, HSK_MAP, SEGMENTATION_WORDS = load_core_lexical_data(WORD_FREQ_PATH, HSK_PATH)


# =========================================================
# LOOKUP HELPERS
# =========================================================


def clear_vocab_cache(vocab_set=None):
    VOCAB_RESOLVE_CACHE.clear()


def resolve_vocab(word, vocab_set):
    if not vocab_set:
        return False
    cache_key = (id(vocab_set), word)
    if cache_key in VOCAB_RESOLVE_CACHE:
        return VOCAB_RESOLVE_CACHE[cache_key]
    result = word in vocab_set
    VOCAB_RESOLVE_CACHE[cache_key] = result
    return result


def get_available_name(proposed_name, existing_names):
    if proposed_name not in existing_names:
        return proposed_name

    suffix = 1
    while f"{proposed_name}_{suffix}" in existing_names:
        suffix += 1
    return f"{proposed_name}_{suffix}"


def resolve_hsk_level(word):
    if word in HSK_RESOLVE_CACHE:
        return HSK_RESOLVE_CACHE[word]
    lvl = HSK_MAP.get(word, 99)
    HSK_RESOLVE_CACHE[word] = lvl
    return lvl


def resolve_topn_rank(word):
    if word in TOPN_RESOLVE_CACHE:
        return TOPN_RESOLVE_CACHE[word]
    rank = WORD_RANK.get(word, 10**9)
    TOPN_RESOLVE_CACHE[word] = rank
    return rank


def non_list_columns(df):
    return tuple(
        col
        for col in df.columns
        if not df[col].map(lambda value: isinstance(value, (list, tuple))).any()
    )


def drop_list_columns(df):
    filtered_df = df.loc[:, list(non_list_columns(df))]
    return filtered_df.drop(columns=["Middle 1000-token extract"], errors="ignore")

def filter_raw_metrics(df):
    def should_include(column_name):
        if not isinstance(column_name, str):
            return True
        stripped = column_name.strip()
        if stripped.endswith("(with frequencies)"):
            return False
        if "Middle 1000-token extract" in stripped:
            return False
        return True

    columns = [col for col in df.columns if should_include(col)]
    return df.loc[:, columns]


FREQUENCY_TABLE_LIMIT = 100


def _sanitize_key_component(value):
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(value))
    safe = safe.strip("_")
    return safe or "item"


def display_frequency_dataframe(df, label, file_name, prefix):
    row_count = len(df)
    if row_count == 0:
        st.dataframe(df, width="stretch")
        return

    widget_key = None
    show_all = False
    if row_count > FREQUENCY_TABLE_LIMIT:
        widget_key = (
            f"{prefix}_{_sanitize_key_component(file_name)}_"
            f"{_sanitize_key_component(label)}_show_all"
        )
        show_all = st.session_state.get(widget_key, False)

    display_df = df if (row_count <= FREQUENCY_TABLE_LIMIT or show_all) else df.head(FREQUENCY_TABLE_LIMIT)
    st.dataframe(display_df, width="stretch")

    if widget_key and row_count > FREQUENCY_TABLE_LIMIT:
        selected = st.checkbox(
            f"Show all {row_count:,}",
            key=widget_key,
        )
        caption_text = "Showing all rows" if selected else f"Showing first {FREQUENCY_TABLE_LIMIT:,} entries"
        st.caption(caption_text)


def format_metric_value(value):
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return "—"
    except TypeError:
        pass
    if isinstance(value, numbers.Integral):
        return f"{int(value):,}"
    if isinstance(value, numbers.Real):
        abs_val = abs(value)
        if abs_val >= 100:
            formatted = f"{value:,.0f}"
        elif abs_val >= 10:
            formatted = f"{value:,.1f}"
        else:
            formatted = f"{value:,.2f}"
        formatted = formatted.rstrip("0").rstrip(".") if "." in formatted else formatted
        return formatted
    return str(value)


def format_percentage(value):
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except TypeError:
        pass
    if isinstance(value, numbers.Real):
        formatted = f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{formatted}%"
    return f"{value}%"


def zipf_to_words_per_occurrence(zipf_value):
    if zipf_value is None or not isinstance(zipf_value, numbers.Real):
        return None
    try:
        if math.isnan(zipf_value):
            return None
    except TypeError:
        return None
    return 10 ** (9 - zipf_value)


def format_occurrence_rate(words_per_occurrence):
    if words_per_occurrence is None or words_per_occurrence <= 0:
        return "rate unavailable"
    if words_per_occurrence >= 1:
        if words_per_occurrence >= 1000:
            return f"~1 per {words_per_occurrence:,.0f} words"
        if words_per_occurrence >= 100:
            return f"~1 per {words_per_occurrence:,.0f} words"
        if words_per_occurrence >= 10:
            return f"~1 per {words_per_occurrence:,.1f} words"
        return f"~1 per {words_per_occurrence:,.2f} words"
    occurrences_per_word = 1 / words_per_occurrence
    return f"~{occurrences_per_word:,.1f} times per word"


def render_compact_table(df):
    if df is None or df.empty:
        st.info("No data available.")
        return
    st.dataframe(df, use_container_width=True)


def render_explanation_dropdown(explanation_items):
    items = [item for item in explanation_items if item]
    if not items:
        return
    with st.expander(":small[Explanation]", expanded=False):
        for item in items:
            st.caption(item)


def render_metric_group(title, rows, word_metrics, char_metrics, explanation=None):
    st.markdown(f"#### {title}")
    table_rows = []
    for row in rows:
        if not row.get("word_key") and not row.get("char_key"):
            continue
        word_value = word_metrics.get(row.get("word_key")) if row.get("word_key") else None
        char_value = char_metrics.get(row.get("char_key")) if row.get("char_key") else None
        if word_value is None and char_value is None:
            continue
        word_formatter = format_percentage if row.get("word_percentage") else format_metric_value
        char_formatter = format_percentage if row.get("char_percentage") else format_metric_value
        table_rows.append(
            {
                "Metric": row["label"],
                "Words": word_formatter(word_value) if row.get("word_key") else "—",
                "Characters": char_formatter(char_value) if row.get("char_key") else "—",
            }
        )
    if table_rows:
        display_df = pd.DataFrame(table_rows).set_index("Metric")
        display_df.index.name = None
        render_compact_table(display_df)
    explanation_lines = []
    if explanation:
        explanation_lines.append(explanation)
    for row in rows:
        if row.get("explanation"):
            explanation_lines.append(f"**{row['label']}** — {row['explanation']}")
    render_explanation_dropdown(explanation_lines)


def format_zipf_entry(value):
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except TypeError:
        pass
    occurrence_rate = format_occurrence_rate(zipf_to_words_per_occurrence(value))
    return f"{format_metric_value(value)} ({occurrence_rate})"


def render_zipf_section(word_metrics, char_metrics):
    st.markdown("#### Zipf frequency profile")

    table_df = pd.DataFrame(
        {
            "Words": {
                "Token median Zipf": word_metrics.get("Median zipf (all tokens)"),
                "Unique median Zipf": word_metrics.get("Median zipf (unique words)"),
            },
            "Characters": {
                "Token median Zipf": char_metrics.get("Median zipf (all characters)"),
                "Unique median Zipf": char_metrics.get("Median zipf (unique characters)"),
            },
        }
    )
    zipf_styler = (
        table_df.style
        .format(format_zipf_entry)
        .background_gradient(cmap="RdYlBu_r", axis=None, vmin=3, vmax=8)
    )
    st.dataframe(zipf_styler, use_container_width=True)

    render_explanation_dropdown([
        "Zipf is the base-10 log of occurrences per billion words in the WordFreq corpus. "
        "Zipf 8 ≈ once every 10 words, Zipf 7 ≈ once every 100 words, and Zipf 3 ≈ once per million words. "
        "For context, 的 scores 7.79 (~1 per 16 words), while rarer literary words such as 玄冥 hover near Zipf 3 (~1 per million words). "
        "Unique-column Zipf values summarise how rare the vocabulary set is overall, whereas token-column Zipf values show what readers repeatedly encounter. "
        "A big gap between the two means rare words appear but only sporadically; similar values mean those rare terms keep showing up."
    ])


def build_topn_coverage_df(word_metrics, char_metrics):
    rows = []
    for n in range(1, 11):
        label = f"Top {n}k"
        rows.append(
            {
                "Band": label,
                "Total tokens": word_metrics.get(f"Top {n}k token coverage (%)"),
                "Total characters": char_metrics.get(f"Top {n}k character coverage (%)"),
                "Unique words": word_metrics.get(f"Top {n}k unique word coverage (%)"),
                "Unique characters": char_metrics.get(f"Top {n}k unique character coverage (%)"),
            }
        )
    return pd.DataFrame(rows).set_index("Band")


def build_hsk_coverage_df(word_metrics, char_metrics):
    rows = []
    level_labels = [
        ("HSK 1", "HSK 1 to 1"),
        ("HSK 1 to 2", "HSK 1 to 2"),
        ("HSK 1 to 3", "HSK 1 to 3"),
        ("HSK 1 to 4", "HSK 1 to 4"),
        ("HSK 1 to 5", "HSK 1 to 5"),
        ("HSK 1 to 6", "HSK 1 to 6"),
        ("HSK 1 to 7–9", "HSK 1 to 7–9"),
    ]
    for label, metric_prefix in level_labels:
        rows.append(
            {
                "Band": label,
                "Total tokens": word_metrics.get(f"{metric_prefix} token coverage (%)"),
                "Total characters": char_metrics.get(f"{metric_prefix} character coverage (%)"),
                "Unique words": word_metrics.get(f"{metric_prefix} unique word coverage (%)"),
                "Unique characters": char_metrics.get(f"{metric_prefix} unique character coverage (%)"),
            }
        )
    return pd.DataFrame(rows).set_index("Band")


def build_custom_vocab_coverage_df(word_metrics, char_metrics):
    return pd.DataFrame(
        {
            "Words": {
                "Total": word_metrics.get("Custom vocab token coverage (%)"),
                "Unique": word_metrics.get("Custom vocab unique word coverage (%)"),
            },
            "Characters": {
                "Total": char_metrics.get("Custom vocab character coverage (%)"),
                "Unique": char_metrics.get("Custom vocab unique character coverage (%)"),
            },
        }
    )


def style_coverage(df):
    return (
        df.style
        .format(format_percentage)
        .background_gradient(cmap="RdYlBu_r", axis=None, vmin=0, vmax=100)
    )


def render_coverage_highlights(word_metrics, char_metrics, has_custom_vocab=False):
    st.markdown("#### Coverage highlights")

    if has_custom_vocab:
        custom_df = build_custom_vocab_coverage_df(word_metrics, char_metrics)
        st.markdown("**Custom vocab coverage**")
        st.dataframe(style_coverage(custom_df), use_container_width=True)
        render_explanation_dropdown([
                "**Total** compares running tokens/characters against your vocab list to show immediate reading coverage. "
                "**Unique** checks distinct types, highlighting whether gaps come from brand-new words or glyphs. "
                "Comparing columns reveals if unknown words are built from already-known characters or if entirely new Hanzi appear."
        ])

    hsk_df = build_hsk_coverage_df(word_metrics, char_metrics)
    st.markdown("**HSK 1–7–9 coverage**")
    st.dataframe(style_coverage(hsk_df), use_container_width=True)
    render_explanation_dropdown([
        "Rows accumulate upward through the official HSK 3.0 bands, letting you see how quickly the syllabus covers your text. "
        "Token coverage answers 'How much of the running text sits within this syllabus tier?', whereas Unique coverage asks 'How much of the vocabulary inventory is already taught by this level?'. "
        "Cool blues mean low coverage, warm reds mean high coverage, so you can gauge at a glance which tiers dominate."
    ])

    topn_df = build_topn_coverage_df(word_metrics, char_metrics)
    st.markdown("**Top 10k WordFreq coverage**")
    st.dataframe(style_coverage(topn_df), use_container_width=True)
    render_explanation_dropdown([
        "Each row aggregates everything up to that frequency band, so the Top 3k row reflects coverage from the Top 1k, 2k, and 3k buckets combined. "
        "Token columns show how often readers will *see* those bands, while Unique columns show what proportion of the vocabulary inventory comes from them. "
        "The color bar runs from blue (low coverage) to red (high coverage), making it easy to spot whether the text stays in high-frequency territory."
    ])


def render_stat_guide(word_metrics, char_metrics, has_custom_vocab=False):
    render_metric_group(
        "Volume overview",
        [
            {
                "label": "Total length",
                "word_key": "Total tokens",
                "char_key": "Total characters",
                "explanation": (
                    "Word column counts segmented tokens; character column counts every Hanzi. "
                    "If tokens vastly outnumber characters, the text reuses a small set of characters to form many multi-character words. "
                    "If the numbers are close, the text leans on single-character words and feels more repetitive."
                ),
            },
            {
                "label": "Unique vocabulary",
                "word_key": "Unique words",
                "char_key": "Unique characters",
                "explanation": (
                    "Shows how many different word types and characters appear. "
                    "High word totals with modest character counts mean familiar components are recombined into new compounds; "
                    "high character totals signal broader glyph coverage that pushes recognition skills."
                ),
            },
            {
                "label": "Unique rate (% of total)",
                "word_key": "Unique words (% of tokens)",
                "char_key": "Unique characters (% of characters)",
                "word_percentage": True,
                "char_percentage": True,
                "explanation": (
                    "Type–token ratios reveal how quickly new vocabulary appears. "
                    "A high word percentage but low character percentage means new words are built from known characters, "
                    "whereas high percentages in both columns indicate constant introduction of unfamiliar glyphs."
                ),
            },
        ],
        word_metrics,
        char_metrics,
    )

    render_metric_group(
        "Sentence rhythm & local variety",
        [
            {
                "label": "Average sentence length",
                "word_key": "Average tokens per sentence",
                "char_key": "Average characters per sentence",
                "explanation": (
                    "Long sentences in tokens point to dense clause structures. "
                    "Comparing the character column tells you whether the length comes from many short words or a few long compounds."
                ),
            },
            {
                "label": "Median unique per 1000-window",
                "word_key": "Median unique words per 1000-token window",
                "char_key": "Median unique characters per 1000-character window",
                "explanation": (
                    "Measures lexical diversity inside sliding 1000-unit windows. "
                    "High word medians mean the text keeps refreshing vocabulary; high character medians mean it keeps introducing new glyphs."
                ),
            },
        ],
        word_metrics,
        char_metrics,
    )

    render_zipf_section(word_metrics, char_metrics)
    render_coverage_highlights(word_metrics, char_metrics, has_custom_vocab)

@st.cache_data
def build_char_level_maps(hsk_map, word_rank):
    char_hsk_level = {}
    for hsk_word, lvl in hsk_map.items():
        for ch in hsk_word:
            char_hsk_level[ch] = min(lvl, char_hsk_level.get(ch, 99))

    char_topn_rank = {}
    for top_word, rank in word_rank.items():
        for ch in top_word:
            char_topn_rank[ch] = min(rank, char_topn_rank.get(ch, 10**9))

    return char_hsk_level, char_topn_rank


CHAR_HSK_LEVEL, CHAR_TOPN_RANK = build_char_level_maps(HSK_MAP, WORD_RANK)


# =========================================================
# STREAMLIT UI
# =========================================================
# ----------------------------
# Session state initialisation
# ----------------------------
init_session_state(st.session_state)

if st.session_state.upload_status_toast:
    level, message = st.session_state.upload_status_toast

    if level == "warning":
        st.toast(message, icon="⚠️")
    else:
        st.toast(message, icon="✅")

    st.session_state.upload_status_toast = None


# =========================================================
# HOW THE TOOL WORKS + SEGMENTATION DEMO
# =========================================================

with st.expander("How does this tool work?", expanded=False):
    if "demo_text" not in st.session_state:
        st.session_state.demo_text = DEFAULT_DEMO_TEXT

    st.markdown("""
This tool analyses Chinese text at two levels:

- **Word tokens** (segmented using jieba)
- **Characters** (individual Hanzi)

A *token* is a segmented word. For example, 中国 is treated as one token, even though it contains two characters.

The tool then calculates:
- Total tokens
- Unique words
- Total characters
- Unique characters
- Sentence lengths
- Vocabulary coverage statistics

How segmentation works in practice:
- Jieba uses dictionary-based matching plus word-frequency statistics to choose the most likely word boundaries.
- When multiple segmentations are possible, higher-probability/frequency paths are preferred.
- In this tool, if you upload a custom vocab list, those words are added to the internal tokeniser lookup to guide segmentation.

The example below demonstrates how segmentation works.
""")

    st.subheader("Segmentation demo")

    demo_text = st.text_area(
        "Enter Chinese text (max 200 characters)",
        key="demo_text",
        max_chars=200,
        height=100,
    )

    if demo_text.strip():
        demo_tokenizer = get_tokenizer(st.session_state.custom_vocab_set)

        sentence_spans = []
        start = 0
        for m in SENTENCE_SPLIT_RE.finditer(demo_text):
            end = m.end()
            sentence_spans.append((start, end))
            start = end
        if start < len(demo_text):
            sentence_spans.append((start, len(demo_text)))

        demo_tokens = []
        word_positions = []

        for token, start_idx, end_idx in demo_tokenizer.tokenize(demo_text, HMM=False):
            if HANZI_ONLY_RE.match(token):
                demo_tokens.append(token)
                word_positions.append((start_idx, end_idx))

        demo_chars = HANZI_RE.findall(demo_text)

        total_tokens = len(demo_tokens)
        unique_tokens = len(set(demo_tokens))
        total_chars = len(demo_chars)
        unique_chars = len(set(demo_chars))

        words_per_sentence = []
        chars_per_sentence = []

        wi = 0
        for s_start, s_end in sentence_spans:
            token_count = 0

            while wi < len(word_positions):
                w_start, w_end = word_positions[wi]

                if w_end <= s_start:
                    wi += 1
                    continue
                if w_start >= s_end:
                    break

                token_count += 1
                wi += 1

            if token_count > 0:
                words_per_sentence.append(token_count)

            chars_in_sentence = HANZI_RE.findall(demo_text[s_start:s_end])
            if chars_in_sentence:
                chars_per_sentence.append(len(chars_in_sentence))

        st.markdown("### Segmented tokens")
        demo_highlight_options = [
            ("None", None),
            ("Word: HSK", "word_hsk"),
            ("Word: Top-N", "word_topn"),
            ("Char: HSK", "char_hsk"),
            ("Char: Top-N", "char_topn"),
        ]
        if st.session_state.custom_vocab_set:
            demo_highlight_options.insert(1, ("Word: vocab", "word_vocab"))
            demo_highlight_options.insert(4, ("Char: vocab", "char_vocab"))

        demo_label_to_mode = dict(demo_highlight_options)
        demo_labels = [label for label, _ in demo_highlight_options]
        current_demo_label = st.session_state.get("demo_highlight_mode", demo_labels[0])
        if current_demo_label not in demo_label_to_mode:
            current_demo_label = demo_labels[0]
        demo_highlight_mode = demo_label_to_mode[current_demo_label]

        sentence_lines_html = render_sentence_segmented_tokens_html(
            demo_tokens,
            word_positions,
            sentence_spans,
            highlight_mode=demo_highlight_mode,
            custom_vocab_set=st.session_state.custom_vocab_set,
            resolve_vocab_fn=resolve_vocab,
            resolve_hsk_level_fn=resolve_hsk_level,
            resolve_topn_rank_fn=resolve_topn_rank,
            char_hsk_level=CHAR_HSK_LEVEL,
            char_topn_rank=CHAR_TOPN_RANK,
        )
        if sentence_lines_html:
            st.markdown(sentence_lines_html, unsafe_allow_html=True)
        else:
            st.info("No Chinese tokens found.")

        demo_selected_label = st.radio(
            "Highlight mode",
            demo_labels,
            horizontal=True,
            key="demo_highlight_mode",
        )
        demo_highlight_mode = demo_label_to_mode[demo_selected_label]
        
        if demo_highlight_mode in MODE_EXPLANATIONS:
            st.caption(MODE_EXPLANATIONS[demo_highlight_mode])

        render_legend(demo_highlight_mode)
        
        st.markdown("### Counts")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total tokens", total_tokens)
        col2.metric("Unique words", unique_tokens)
        col3.metric("Total characters", total_chars)
        col4.metric("Unique characters", unique_chars)

        sentence_df = pd.DataFrame(
            {
                "Sentence #": range(1, len(words_per_sentence) + 1),
                "Tokens": words_per_sentence,
                "Characters": chars_per_sentence,
            }
        ) if words_per_sentence else pd.DataFrame(columns=["Sentence #", "Tokens", "Characters"])

        token_df = (
            pd.Series(demo_tokens)
            .value_counts()
            .rename_axis("Word")
            .reset_index(name="Frequency")
        )

        char_df = (
            pd.Series(demo_chars)
            .value_counts()
            .rename_axis("Character")
            .reset_index(name="Frequency")
        )

        table_col1, table_col2, table_col3 = st.columns(3)
        with table_col1:
            st.markdown("### Sentence lengths")
            st.dataframe(sentence_df, width="stretch", hide_index=True)
        with table_col2:
            st.markdown("### Token frequencies")
            st.dataframe(token_df, width="stretch", hide_index=True)
        with table_col3:
            st.markdown("### Character frequencies")
            st.dataframe(char_df, width="stretch", hide_index=True)


if st.session_state.show_vocab_warning:
    st.toast(st.session_state.show_vocab_warning, icon="⚠️", duration="infinite")
    st.session_state.show_vocab_warning = None

custom_vocab_set = show_custom_vocab_section(
    st.session_state,
    clear_vocab_cache,
    get_tokenizer,
    analyse_text,
    resolve_vocab,
    resolve_hsk_level,
    resolve_topn_rank,
)

show_input_section(
    st.session_state,
    get_available_name,
    get_tokenizer,
    analyse_text,
    resolve_vocab,
    resolve_hsk_level,
    resolve_topn_rank,
    extract_text_from_epub,
    extract_text_from_pdf,
)

word_df, char_df = show_export_and_clear(st.session_state, clear_vocab_cache)

custom_vocab_set = st.session_state.custom_vocab_set
has_custom_vocab = bool(custom_vocab_set)

if word_df is not None and char_df is not None and not word_df.empty:
    analysis_files = [
        file_name
        for file_name in word_df.index
        if not str(file_name).startswith("(Custom vocab)")
    ]

    if not analysis_files:
        st.info("Upload a file or paste text to view statistics.")
    else:
        active_file = analysis_files[0]
        word_row = word_df.loc[active_file]
        char_row = char_df.loc[active_file]

        filtered_word_metrics_df = filter_raw_metrics(word_df.loc[[active_file]])
        filtered_char_metrics_df = filter_raw_metrics(char_df.loc[[active_file]])
        word_metrics_series = filtered_word_metrics_df.loc[active_file]
        char_metrics_series = filtered_char_metrics_df.loc[active_file]

        st.subheader(f"Analysis for {active_file}")

        stats_tab, words_tab, chars_tab, sample_tab = st.tabs([
            "Statistics",
            "All Words",
            "All Characters",
            "Sample Text",
        ])

        with stats_tab:
            render_stat_guide(word_metrics_series, char_metrics_series, has_custom_vocab)

        with words_tab:
            wrow = word_row

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total tokens", f"{int(wrow['Total tokens']):,}")
            with col2:
                st.metric("Unique words", f"{int(wrow['Unique words']):,}")
        
            total_tokens = int(wrow["Total tokens"])
            
            word_sections = [("All Words", wrow.get("Unique words (with frequencies)", []))]
            if has_custom_vocab:
                word_sections.append(("Not in Custom Vocab", wrow.get("Words not in vocab (with frequencies)", [])))
            word_sections.extend([
                ("Not in HSK Wordlist", wrow.get("Words not in HSK (with frequencies)", [])),
                ("Not in Top 10k Most Common Words", wrow.get("Words not in top 10k most common words (with frequencies)", [])),
            ])

            for tab, (label, data) in zip(st.tabs([x[0] for x in word_sections]), word_sections):
                with tab:
                    if data:
                        df_word = pd.DataFrame(data, columns=["Word", "Frequency"])

                        df_word["Occurs every __ words"] = (
                            total_tokens / df_word["Frequency"]
                        ).round(0).astype(int).map(lambda x: f"1 in {x:,}")

                        df_word.index = range(1, len(df_word) + 1)
                        df_word.index.name = "#"
                    else:
                        df_word = pd.DataFrame(columns=["Word", "Frequency", "Occurs every __ words"])

                    if label == "Not in Custom Vocab":
                        st.markdown(f"Words not in custom vocab: **{len(df_word):,}**")

                    elif label == "Not in HSK Wordlist":
                        st.markdown(f"Words not in HSK wordlist: **{len(df_word):,}**")

                    elif label == "Not in Top 10k Most Common Words":
                        st.markdown(f"Words not in top 10k most common words: **{len(df_word):,}**")

                    display_frequency_dataframe(df_word, label, active_file, "word_freq")

        with chars_tab:
            crow = char_row

            st.caption(
                "Character frequencies may be higher than the frequency of the same single-character word. This is because characters are counted wherever they appear in the text, including inside multi-character words."
                " For example, 的 may appear as a standalone word, but it also appears inside words like 真的 — so the total character count for 的 can exceed the word count for 的."
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total characters", f"{int(crow['Total characters']):,}")
            with col2:
                st.metric("Unique characters", f"{int(crow['Unique characters']):,}")

            total_chars = int(crow["Total characters"])
            
            char_sections = [("All Characters", crow.get("Unique characters (with frequencies)", []))]
            if has_custom_vocab:
                char_sections.append(("Not in Custom Vocab", crow.get("Characters not in vocab (with frequencies)", [])))
            char_sections.extend([
                ("Not in HSK Wordlist", crow.get("Characters not in HSK (with frequencies)", [])),
                ("Not in Top 10k Most Common Words", crow.get("Characters not in top 10k most common words (with frequencies)", [])),
            ])

            for tab, (label, data) in zip(st.tabs([x[0] for x in char_sections]), char_sections):
                with tab:
                    if data:
                        df_char = pd.DataFrame(data, columns=["Character", "Frequency"])

                        df_char["Occurs every __ chars"] = (
                            total_chars / df_char["Frequency"]
                        ).round(0).astype(int).map(lambda x: f"1 in {x:,}")

                        df_char.index = range(1, len(df_char) + 1)
                        df_char.index.name = "#"

                    else:
                        df_char = pd.DataFrame(columns=["Character", "Frequency", "Occurs every __ chars"])

                    if label == "Not in Custom Vocab":
                        st.markdown(f"Characters not in custom vocab: **{len(df_char):,}**")

                    elif label == "Not in HSK Wordlist":
                        st.markdown(f"Characters not in HSK wordlist: **{len(df_char):,}**")

                    elif label == "Not in Top 10k Most Common Words":
                        st.markdown(f"Characters not in top 10k most common words: **{len(df_char):,}**")

                    display_frequency_dataframe(df_char, label, active_file, "char_freq")

        with sample_tab:
            wrow = word_row

            st.caption("Extract of the original text from the midpoint containing 1,000 Hanzi word tokens.")

            text_display_mode = st.radio(
                "Display mode",
                ["Display only Hanzi", "Display all text (including punctuation and non-hanzi words)"],
                index=0,
                horizontal=True,
            )
            
            options = [
                ("None", None),
                ("Word: vocab", "word_vocab"),
                ("Word: HSK", "word_hsk"),
                ("Word: Top-N", "word_topn"),
                ("Character: vocab", "char_vocab"),
                ("Character: HSK", "char_hsk"),
                ("Character: Top-N", "char_topn"),
            ]
            if not has_custom_vocab:
                options = [x for x in options if "vocab" not in x[0].lower()]

            labels = [x[0] for x in options]
            label_to_mode = dict(options)
            selected_label = st.radio("Highlight mode", labels, horizontal=True)
            highlight_mode = label_to_mode[selected_label]

            if highlight_mode in MODE_EXPLANATIONS:
                st.caption(MODE_EXPLANATIONS[highlight_mode])

            render_legend(highlight_mode)

            extract_text = wrow["Middle 1000-token extract"]
            if not isinstance(extract_text, str):
                extract_text = "" if pd.isna(extract_text) else str(extract_text)

            if text_display_mode == "Display only Hanzi":
                extract_text = "".join(HANZI_RE.findall(extract_text))

            extract_tokens = []
            cursor = 0
            extract_tokenizer = get_tokenizer(custom_vocab_set)
            for token, start_idx, end_idx in extract_tokenizer.tokenize(extract_text, HMM=False):
                if start_idx > cursor:
                    extract_tokens.append(extract_text[cursor:start_idx])
                extract_tokens.append(token)
                cursor = end_idx
            if cursor < len(extract_text):
                extract_tokens.append(extract_text[cursor:])

            html_text = render_highlighted_text(
                extract_tokens,
                highlight_mode=highlight_mode,
                custom_vocab_set=custom_vocab_set,
                separator="",
                resolve_vocab_fn=resolve_vocab,
                resolve_hsk_level_fn=resolve_hsk_level,
                resolve_topn_rank_fn=resolve_topn_rank,
                char_hsk_level=CHAR_HSK_LEVEL,
                char_topn_rank=CHAR_TOPN_RANK,
            )

            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="padding: 8px 2px; font-size: 16px; line-height: 1.8;">
                    {html_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            query = urllib.parse.quote(extract_text)
            st.markdown(f"[Open in Google Translate](https://translate.google.com/?sl=zh-CN&tl=en&text={query}&op=translate)")
