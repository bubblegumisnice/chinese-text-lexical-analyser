"""Plotly helper functions for the Streamlit UI."""

import plotly.graph_objects as go
import streamlit as st

PLOTLY_STATIC_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "zoom2d",
        "pan2d",
        "select2d",
        "lasso2d",
        "resetScale2d",
        "autoScale2d",
        "zoomIn2d",
        "zoomOut2d",
        "toggleSpikelines",
    ],
}


def smart_round(x, min_decimals=0, string=False):
    if x >= 1:
        rounded = round(x, min_decimals)
        if string:
            if min_decimals > 0:
                return f"{rounded:.{min_decimals}f}"
            return f"{int(rounded)}"
        return rounded
    rounded = float(f"{x:.1g}")  # 1 significant figure for values < 1
    if string:
        return f"{rounded:g}"
    return rounded


def render_chart(fig, height=400):
    fig.update_layout(dragmode=False, height=height)
    st.plotly_chart(fig, width="stretch", config=PLOTLY_STATIC_CONFIG)


def plot_vocab_overlap(
    row,
    unique_count_key="Unique words",
    unique_coverage_key="Custom vocab unique word coverage (%)",
    item_label="words",
    title="Custom vocab unique-word coverage",
    total_label="Unique words",
    overlap_label="In vocab list (unique)",
):
    if unique_coverage_key not in row:
        return None

    unique_words = row[unique_count_key]
    overlap_pct = row[unique_coverage_key]
    overlap_count = int(overlap_pct * unique_words / 100)

    labels = [total_label, overlap_label]
    values = [unique_words, overlap_count]

    text_labels = [
        f"{unique_words:,} {item_label}<br>100%",
        f"{overlap_count:,} {item_label}<br>{smart_round(overlap_pct, 0, True)}%",
    ]

    ymax = max(values) * 1.2 if max(values) else 1

    fig = go.Figure()

    fig.add_bar(
        x=labels,
        y=values,
        text=text_labels,
        textposition="outside",
        opacity=0.85,
    )

    fig.update_layout(
        title=title,
        yaxis_title="Count",
        yaxis=dict(range=[0, ymax]),
        template="plotly",
        height=400,
    )

    return fig


def plot_vocab_token_overlap(
    row,
    total_count_key="Total tokens",
    token_coverage_key="Custom vocab token coverage (%)",
    item_label="tokens",
    title="Custom vocab token coverage",
    total_label="Total tokens",
    overlap_label="In vocab list (tokens)",
):
    if token_coverage_key not in row:
        return None

    total_tokens = int(row[total_count_key])
    token_pct = row[token_coverage_key]
    token_count = int(token_pct * total_tokens / 100)

    labels = [total_label, overlap_label]
    values = [total_tokens, token_count]

    text_labels = [
        f"{total_tokens:,} {item_label}<br>100%",
        f"{token_count:,} {item_label}<br>{smart_round(token_pct, 0, True)}%",
    ]

    ymax = max(values) * 1.2 if max(values) else 1

    fig = go.Figure()

    fig.add_bar(
        x=labels,
        y=values,
        text=text_labels,
        textposition="outside",
        opacity=0.85,
    )

    fig.update_layout(
        title=title,
        yaxis_title="Count",
        yaxis=dict(range=[0, ymax]),
        template="plotly",
        height=400,
    )

    return fig


def plot_word_counts(row):
    total_tokens = int(row["Total tokens"])
    unique_words = int(row["Unique words"])

    labels = ["Total tokens", "Unique words"]
    values = [total_tokens, unique_words]

    unique_pct = unique_words / total_tokens * 100 if total_tokens else 0

    text_labels = [
        f"{total_tokens:,} tokens<br>100%",
        f"{unique_words:,} words<br>{smart_round(unique_pct, 0, True)}%",
    ]

    ymax = max(values) * 1.2 if max(values) else 1

    fig = go.Figure()

    fig.add_bar(
        x=labels,
        y=values,
        text=text_labels,
        textposition="outside",
        opacity=0.85,
    )

    fig.update_layout(
        title="Word counts",
        yaxis_title="Count",
        yaxis=dict(range=[0, ymax]),
        template="plotly",
        height=400,
    )

    return fig


def plot_char_counts(row):
    total_chars = int(row["Total characters"])
    unique_chars = int(row["Unique characters"])

    labels = ["Total characters", "Unique characters"]
    values = [total_chars, unique_chars]

    unique_pct = unique_chars / total_chars * 100 if total_chars else 0

    text_labels = [
        f"{total_chars:,} chars<br>100%",
        f"{unique_chars:,} chars<br>{smart_round(unique_pct, 0, True)}%",
    ]

    ymax = max(values) * 1.2 if max(values) else 1

    fig = go.Figure()

    fig.add_bar(
        x=labels,
        y=values,
        text=text_labels,
        textposition="outside",
        opacity=0.85,
    )

    fig.update_layout(
        title="Character counts",
        yaxis_title="Count",
        yaxis=dict(range=[0, ymax]),
        template="plotly",
        height=400,
    )

    return fig


def plot_hsk_coverage(
    row,
    unique_coverage_key_template="HSK 1 to {lvl} unique word coverage (%)",
    unique_count_key="Unique words",
    item_label="words",
    title="Cumulative unique-word HSK coverage",
):
    hsk_labels = ["1", "2", "3", "4", "5", "6", "7–9"]

    unique_hsk_pct = [row[unique_coverage_key_template.format(lvl=lvl)] for lvl in hsk_labels]
    total_unique_words = int(row[unique_count_key])

    unique_hsk_counts = [round(total_unique_words * pct / 100) for pct in unique_hsk_pct]

    plot_labels = hsk_labels + ["All"]
    plot_pcts = unique_hsk_pct + [100.0]
    plot_counts = unique_hsk_counts + [total_unique_words]

    colors = [
        "#35c85c",  # HSK 1
        "#fdcd15",  # HSK 2
        "#ff8e28",  # HSK 3
        "#ff383e",  # HSK 4
        "#cc31e1",  # HSK 5
        "#6255f4",  # HSK 6
        "#178afd",  # HSK 7–9
        "#1c1c1c",  # All
    ]

    fig = go.Figure()

    fig.add_bar(
        x=plot_labels,
        y=plot_pcts,
        text=[
            f"{c:,} {item_label}<br>{smart_round(p, 0, True)}%"
            for c, p in zip(plot_counts, plot_pcts)
        ],
        textposition="outside",
        marker_color=colors,
        opacity=0.85,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Maximum HSK level",
        yaxis_title="Coverage (%)",
        yaxis=dict(range=[0, 119]),
        template="plotly",
        height=400,
    )

    return fig


def plot_topn_coverage(
    row,
    unique_coverage_key_template="Top {n}k unique word coverage (%)",
    unique_count_key="Unique words",
    item_label="words",
    title="Cumulative unique-word Top-N frequency coverage",
    xaxis_title="Top-N most frequent words (WordFreq)",
):
    top_levels = [1000 * i for i in range(1, 11)]
    top_labels = [f"{n//1000}k" for n in top_levels]

    unique_top_pct = [row[unique_coverage_key_template.format(n=n // 1000)] for n in top_levels]
    total_unique_words = int(row[unique_count_key])

    unique_top_counts = [round(total_unique_words * pct / 100) for pct in unique_top_pct]

    plot_labels = top_labels + ["All"]
    plot_pcts = unique_top_pct + [100.0]
    plot_counts = unique_top_counts + [total_unique_words]

    colors = [
        "#90caf9",
        "#78b8f2",
        "#60a6eb",
        "#4894e4",
        "#3884d9",
        "#307acc",
        "#2c70c4",
        "#286ac0",
        "#2467bd",
        "#2064ba",
        "#1c1c1c",
    ]

    fig = go.Figure()

    fig.add_bar(
        x=plot_labels,
        y=plot_pcts,
        text=[
            f"{c:,} {item_label}<br>{smart_round(p, 0, True)}%"
            for c, p in zip(plot_counts, plot_pcts)
        ],
        textposition="outside",
        marker_color=colors,
        opacity=0.85,
    )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Coverage (%)",
        yaxis=dict(range=[0, 119]),
        template="plotly",
        height=400,
    )

    return fig


def plot_hsk_token_coverage(
    row,
    token_coverage_key_template="HSK 1 to {lvl} token coverage (%)",
    total_count_key="Total tokens",
    item_label="tokens",
    title="Cumulative token HSK coverage",
):
    hsk_labels = ["1", "2", "3", "4", "5", "6", "7–9"]

    token_hsk_pct = [row[token_coverage_key_template.format(lvl=lvl)] for lvl in hsk_labels]
    total_tokens = int(row[total_count_key])

    token_hsk_counts = [round(total_tokens * pct / 100) for pct in token_hsk_pct]

    plot_labels = hsk_labels + ["All"]
    plot_pcts = token_hsk_pct + [100.0]
    plot_counts = token_hsk_counts + [total_tokens]

    colors = [
        "#35c85c",
        "#fdcd15",
        "#ff8e28",
        "#ff383e",
        "#cc31e1",
        "#6255f4",
        "#178afd",
        "#1c1c1c",
    ]

    fig = go.Figure()

    fig.add_bar(
        x=plot_labels,
        y=plot_pcts,
        text=[
            f"{c:,} {item_label}<br>{smart_round(p, 0, True)}%"
            for c, p in zip(plot_counts, plot_pcts)
        ],
        textposition="outside",
        marker_color=colors,
        opacity=0.85,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Maximum HSK level",
        yaxis_title="Coverage (%)",
        yaxis=dict(range=[0, 119]),
        template="plotly",
        height=400,
    )

    return fig


def plot_topn_token_coverage(
    row,
    token_coverage_key_template="Top {n}k token coverage (%)",
    total_count_key="Total tokens",
    item_label="tokens",
    title="Cumulative token Top-N frequency coverage",
    xaxis_title="Top-N most frequent words (WordFreq)",
):
    top_levels = [1000 * i for i in range(1, 11)]
    top_labels = [f"{n//1000}k" for n in top_levels]

    token_top_pct = [row[token_coverage_key_template.format(n=n // 1000)] for n in top_levels]
    total_tokens = int(row[total_count_key])

    token_top_counts = [round(total_tokens * pct / 100) for pct in token_top_pct]

    plot_labels = top_labels + ["All"]
    plot_pcts = token_top_pct + [100.0]
    plot_counts = token_top_counts + [total_tokens]

    colors = [
        "#90caf9",
        "#78b8f2",
        "#60a6eb",
        "#4894e4",
        "#3884d9",
        "#307acc",
        "#2c70c4",
        "#286ac0",
        "#2467bd",
        "#2064ba",
        "#1c1c1c",
    ]

    fig = go.Figure()

    fig.add_bar(
        x=plot_labels,
        y=plot_pcts,
        text=[
            f"{c:,} {item_label}<br>{smart_round(p, 0, True)}%"
            for c, p in zip(plot_counts, plot_pcts)
        ],
        textposition="outside",
        marker_color=colors,
        opacity=0.85,
    )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Coverage (%)",
        yaxis=dict(range=[0, 119]),
        template="plotly",
        height=400,
    )

    return fig
