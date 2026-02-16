"""Rendering helpers shared across the Streamlit app."""

import html

import pandas as pd
import streamlit as st

from config import TOPN_COLORS, HSK_COLORS
from config import HANZI_ONLY_RE


def render_highlighted_text(
    words,
    highlight_mode=None,
    custom_vocab_set=None,
    separator=" ",
    *,
    resolve_vocab_fn=None,
    resolve_hsk_level_fn=None,
    resolve_topn_rank_fn=None,
    char_hsk_level=None,
    char_topn_rank=None,
):
    rendered = []
    vocab_chars = {ch for item in custom_vocab_set for ch in item} if custom_vocab_set else set()

    for word in words:
        safe_word = html.escape(word)
        style = ""
        is_hanzi_word = bool(HANZI_ONLY_RE.match(word))

        if (
            highlight_mode == "word_vocab"
            and custom_vocab_set
            and resolve_vocab_fn
            and is_hanzi_word
            and resolve_vocab_fn(word, custom_vocab_set)
        ):
            style = "background-color:rgba(122,132,251,0.35);"
        elif highlight_mode == "word_topn" and resolve_topn_rank_fn and is_hanzi_word:
            rank = resolve_topn_rank_fn(word)
            if rank <= 10000:
                band = min((rank - 1) // 1000, 9)
                style = f"background-color:{TOPN_COLORS[band]};"
        elif highlight_mode == "word_hsk" and resolve_hsk_level_fn and is_hanzi_word:
            level = resolve_hsk_level_fn(word)
            if level <= 7:
                style = f"background-color:{HSK_COLORS[min(level, 7)]};"
        elif highlight_mode and highlight_mode.startswith("char_"):
            char_spans = []
            for ch in word:
                ch_style = ""
                if highlight_mode == "char_vocab" and custom_vocab_set and ch in vocab_chars:
                    ch_style = "background-color:rgba(122,132,251,0.35);"
                elif highlight_mode == "char_topn" and char_topn_rank is not None:
                    rank = char_topn_rank.get(ch, 10**9)
                    if rank <= 10000:
                        band = min((rank - 1) // 1000, 9)
                        ch_style = f"background-color:{TOPN_COLORS[band]};"
                elif highlight_mode == "char_hsk" and char_hsk_level is not None:
                    level = char_hsk_level.get(ch, 99)
                    if level <= 7:
                        ch_style = f"background-color:{HSK_COLORS[min(level, 7)]};"

                safe_ch = html.escape(ch)
                if ch_style:
                    char_spans.append(f"<span style='{ch_style} padding:2px; border-radius:3px;'>{safe_ch}</span>")
                else:
                    char_spans.append(safe_ch)
            rendered.append("".join(char_spans))
            continue

        if style:
            rendered.append(f"<span style='{style} padding:2px; border-radius:3px;'>{safe_word}</span>")
        else:
            rendered.append(safe_word)

    return separator.join(rendered)


def render_legend(highlight_mode):
    if highlight_mode in {"word_topn", "char_topn"}:
        labels = [f"{i}k" for i in range(1, 11)]
        colors = TOPN_COLORS
    elif highlight_mode in {"word_hsk", "char_hsk"}:
        labels = [f"HSK {i}" for i in range(1, 7)] + ["HSK 7–9"]
        colors = [HSK_COLORS[i] for i in range(1, 8)]
    elif highlight_mode in {"word_vocab", "char_vocab"}:
        labels = ["Custom vocab"]
        colors = ["rgba(122, 132, 251,0.35)"]
    else:
        return

    cols = st.columns(len(labels))

    for col, label, color in zip(cols, labels, colors):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:{color};
                    height:20px;
                    border-radius:4px;
                    margin-bottom:4px;
                "></div>
                <div style="text-align:center; font-size:12px;">
                    {label}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_sentence_segmented_tokens(tokens, word_positions, sentence_spans):
    sentence_lines = []

    for idx, (s_start, s_end) in enumerate(sentence_spans, start=1):
        sentence_tokens = []
        for token, (w_start, w_end) in zip(tokens, word_positions):
            if w_end <= s_start or w_start >= s_end:
                continue
            sentence_tokens.append(token)

        if not sentence_tokens:
            continue

        sentence_lines.append(f"Sentence {idx}:  {' ｜ '.join(sentence_tokens)}")

    return sentence_lines


def render_sentence_segmented_tokens_html(
    tokens,
    word_positions,
    sentence_spans,
    highlight_mode=None,
    custom_vocab_set=None,
    **highlight_kwargs,
):
    sentence_lines = []
    cursor = 0

    for idx, (s_start, s_end) in enumerate(sentence_spans, start=1):
        sentence_tokens = []

        while cursor < len(word_positions) and word_positions[cursor][1] <= s_start:
            cursor += 1

        lookahead = cursor
        while lookahead < len(word_positions):
            w_start, w_end = word_positions[lookahead]
            if w_end <= s_start or w_start >= s_end:
                if w_start >= s_end:
                    break
                lookahead += 1
                continue
            sentence_tokens.append(tokens[lookahead])
            lookahead += 1

        cursor = lookahead

        if not sentence_tokens:
            continue

        highlighted = render_highlighted_text(
            sentence_tokens,
            highlight_mode=highlight_mode,
            custom_vocab_set=custom_vocab_set,
            separator=" ｜ ",
            **highlight_kwargs,
        )
        sentence_lines.append(f"Sentence {idx}:  {highlighted}")

    return "<br>".join(sentence_lines)


def style_numeric_gradient(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    float_cols = df.select_dtypes(include="float").columns.tolist()

    styler = df.style.format(na_rep="")
    if float_cols:
        styler = styler.format({col: "{:.1f}" for col in float_cols})

    if not numeric_cols:
        return styler

    gradient_cols = [col for col in numeric_cols if df[col].notna().any()]
    if gradient_cols:
        styler = styler.background_gradient(
            cmap="coolwarm",
            axis=0,
            low=0.2,
            high=0.2,
            subset=gradient_cols,
        )

    return styler.map(
        lambda val: "background-color: transparent;" if pd.isna(val) else "",
        subset=numeric_cols,
    )


def format_summary_value(value):
    if pd.isna(value):
        return ""
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


def build_summary_statistics_table(df):
    numeric_cols = df.select_dtypes(include="number").columns
    valid_df = df[~df.index.str.startswith("(Custom vocab)")]
    if len(valid_df) < 2 or len(numeric_cols) == 0:
        return None

    summary_rows = []
    for metric in numeric_cols:
        series = valid_df[metric]
        if series.empty:
            continue

        min_value = series.min()
        median_value = series.median()
        max_value = series.max()

        summary_rows.append(
            {
                "Metric": metric,
                "Min": format_summary_value(min_value),
                "Median": format_summary_value(median_value),
                "Max": format_summary_value(max_value),
                "Min file": series.idxmin(),
                "Max file": series.idxmax(),
            }
        )

    if not summary_rows:
        return None

    return pd.DataFrame(summary_rows).set_index("Metric")


def maybe_hide_custom_vocab_row(df, include_custom_vocab_row):
    if include_custom_vocab_row:
        return df
    return df[~df.index.to_series().str.startswith("(Custom vocab)")]
