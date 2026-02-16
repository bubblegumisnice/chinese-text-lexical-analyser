# -*- coding: utf-8 -*-
"""
Chinese text lexical analyser app
"""

import csv
import gc
import sys
import math
import numbers
import re
import html
import statistics
from collections import Counter

import jieba
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
import urllib.parse
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from wordfreq import zipf_frequency

# Ignore harmless Jieba syntax warnings
import warnings

warnings.filterwarnings(
    "ignore",
    message="invalid escape sequence",
    category=SyntaxWarning,
)

sys.modules.setdefault("app", sys.modules[__name__])


# =========================================================
# INLINED CONFIGURATION CONSTANTS
# =========================================================
WORD_FREQ_PATH = "wordfreq_top10000_zipf.csv"
HSK_PATH = "HSK3_2026_Pleco_pinyin_for_ambiguous_readings.txt"

WINDOW_SIZE = 1000
STEP_SIZE = 250
MAX_WINDOWS = 1000

DEFAULT_DEMO_TEXT = (
    "我很喜欢中国，尤其是它的美食、文化和历史。我不太喜欢政治、经济或者全球化。"
    "我更喜欢看仙侠小说、穿越小说和娱乐圈网文！要不要一起去吃火锅？或者试试更特别的菜，"
    "比如苗家酸汤鱼、傣族苦撒、生腌蟹或者羊杂碎？"
)

TOPN_COLORS = [
    "rgba(144,202,249,0.7)",
    "rgba(120,184,242,0.7)",
    "rgba(96,166,235,0.7)",
    "rgba(72,148,228,0.7)",
    "rgba(56,132,217,0.7)",
    "rgba(48,122,204,0.7)",
    "rgba(44,112,196,0.7)",
    "rgba(40,106,192,0.7)",
    "rgba(36,103,189,0.7)",
    "rgba(32,100,186,0.7)",
]

HSK_COLORS = {
    1: "rgba(53,200,92,0.45)",
    2: "rgba(253,205,21,0.45)",
    3: "rgba(255,142,40,0.45)",
    4: "rgba(255,56,62,0.45)",
    5: "rgba(204,49,225,0.45)",
    6: "rgba(98,85,244,0.45)",
    7: "rgba(23,138,253,0.45)",
}

MODE_EXPLANATIONS = {
    "word_vocab": "Highlights words that appear in your uploaded custom vocab list.",
    "word_hsk": "Highlights words by their HSK level from the official HSK 3.0 list.",
    "word_topn": "Highlights words by frequency band (top 1k, 2k, ..., up to 10k) based on WordFreq common word ranks.",
    "char_vocab": "Highlights characters that appear in any words in your uploaded custom vocab list. Characters not highlighted do not appear in any custom vocab words.",
    "char_hsk": "Assigns each character the lowest HSK level of any HSK-listed word that contains that character. Characters not highlighted do not appear in any HSK words.",
    "char_topn": "Assigns each character the lowest frequency band (top 1k, 2k, ..., up to 10k) of any WordFreq ranked word that contains that character. Characters not highlighted do not appear in any of the top 10k common words.",
}

HANZI_RE = re.compile(r"[\u4e00-\u9fff]")
HANZI_ONLY_RE = re.compile(r"^[\u4e00-\u9fff]+$")
SENTENCE_SPLIT_RE = re.compile(r"[。！？]+")

# Global caches
VOCAB_RESOLVE_CACHE = {}
HSK_RESOLVE_CACHE = {}
TOPN_RESOLVE_CACHE = {}

WORD_LIST_UNIQUE = "Unique words (word, occurrences in text, WordFreq Zipf)"
WORD_LIST_NOT_IN_VOCAB = "Words not in vocab (word, occurrences in text, WordFreq Zipf)"
WORD_LIST_NOT_IN_HSK = "Words not in HSK (word, occurrences in text, WordFreq Zipf)"
WORD_LIST_NOT_IN_TOP = "Words not in top 10k most common words (word, occurrences in text, WordFreq Zipf)"

CHAR_LIST_UNIQUE = "Unique characters (character, occurrences in text, WordFreq Zipf)"
CHAR_LIST_NOT_IN_VOCAB = "Characters not in vocab (character, occurrences in text, WordFreq Zipf)"
CHAR_LIST_NOT_IN_HSK = "Characters not in HSK (character, occurrences in text, WordFreq Zipf)"
CHAR_LIST_NOT_IN_TOP = "Characters not in top 10k most common words (character, occurrences in text, WordFreq Zipf)"


def format_zipf_with_ratio(zipf_value, ratio, unit):
    if zipf_value is None or isinstance(zipf_value, float) and math.isnan(zipf_value):
        zipf_text = "—"
    else:
        zipf_text = f"{zipf_value:.2f}"
    if not ratio or ratio <= 0 or math.isinf(ratio):
        return zipf_text
    ratio_value = max(int(round(ratio)), 1)
    return f"{zipf_text} (1 in {ratio_value:,} {unit})"


def wordfreq_ratio_from_zipf(zipf_value):
    if zipf_value is None or (isinstance(zipf_value, float) and math.isnan(zipf_value)):
        return None
    return 10 ** (9 - zipf_value)


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
# TOKENIZER HELPERS (INLINE)
# =========================================================
@st.cache_resource
def get_base_tokenizer():
    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()
    for word in SEGMENTATION_WORDS:
        tokenizer.add_word(word)
    return tokenizer


@st.cache_resource
def get_custom_tokenizer(vocab_signature: tuple):
    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()
    for word in SEGMENTATION_WORDS:
        tokenizer.add_word(word)
    for word in vocab_signature:
        tokenizer.add_word(word)
    return tokenizer


def get_tokenizer(custom_vocab_set=None):
    if not custom_vocab_set:
        return get_base_tokenizer()
    vocab_signature = tuple(sorted(custom_vocab_set))
    return get_custom_tokenizer(vocab_signature)


# =========================================================
# ANALYSIS HELPERS (INLINE)
# =========================================================
def extract_text_from_epub(uploaded_file):
    book = epub.read_epub(uploaded_file)

    text_chunks = []

    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text_chunks.append(soup.get_text())

    return "\n".join(text_chunks)


def extract_text_from_pdf(uploaded_file):
    text_chunks = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_chunks.append(text)

    return "\n".join(text_chunks)


def analyse_text(
    text: str,
    tokenizer,
    custom_vocab_set,
    resolve_vocab,
    resolve_hsk_level,
    resolve_topn_rank,
):

    vocab_cache = {}
    hsk_cache = {}
    topn_cache = {}
    zipf_cache = {}

    def vocab_known(w):
        if w in vocab_cache:
            return vocab_cache[w]
        known = resolve_vocab(w, custom_vocab_set)
        vocab_cache[w] = known
        return known

    def get_hsk_level(w):
        if w in hsk_cache:
            return hsk_cache[w]
        lvl = resolve_hsk_level(w)
        hsk_cache[w] = lvl
        return lvl

    def get_topn_rank(w):
        if w in topn_cache:
            return topn_cache[w]
        rank = resolve_topn_rank(w)
        topn_cache[w] = rank
        return rank

    def get_zipf(term):
        if term in zipf_cache:
            return zipf_cache[term]
        zipf_value = zipf_frequency(term, "zh")
        zipf_cache[term] = zipf_value
        return zipf_value

    def median_zipf_unique(terms):
        if not terms:
            return 0.0
        return round(statistics.median(get_zipf(term) for term in terms), 3)

    def median_zipf_weighted(counts, total):
        if total == 0:
            return 0.0

        zipf_counts = Counter()
        for term, count in counts.items():
            zipf_counts[get_zipf(term)] += count

        sorted_items = sorted(zipf_counts.items())

        def kth_value(k):
            cumulative = 0
            for zipf_value, count in sorted_items:
                cumulative += count
                if cumulative >= k:
                    return zipf_value
            return 0.0

        left = kth_value((total + 1) // 2)
        right = kth_value((total + 2) // 2)
        return round((left + right) / 2, 3)

    def cumulative_percent_from_rank_counts(rank_counts, total, cutoffs):
        coverage = {}
        sorted_ranks = sorted(rank_counts)
        idx = 0
        cumulative = 0
        for cutoff in cutoffs:
            while idx < len(sorted_ranks) and sorted_ranks[idx] <= cutoff:
                cumulative += rank_counts[sorted_ranks[idx]]
                idx += 1
            coverage[cutoff] = (cumulative / total * 100) if total else 0
        return coverage

    hanzi_chars = HANZI_RE.findall(text)
    total_chars = len(hanzi_chars)
    unique_char_set = set(hanzi_chars)
    unique_chars = len(unique_char_set)

    sentence_spans = []
    start = 0
    for match in SENTENCE_SPLIT_RE.finditer(text):
        end = match.end()
        sentence_spans.append((start, end))
        start = end
    if start < len(text):
        sentence_spans.append((start, len(text)))

    words = []
    word_positions = []
    for token, start_idx, end_idx in tokenizer.tokenize(text, HMM=False):
        if not HANZI_ONLY_RE.match(token):
            continue
        words.append(token)
        word_positions.append((start_idx, end_idx))

    total_words = len(words)
    unique_word_set = set(words)
    unique_words = len(unique_word_set)

    word_counts = Counter(words)
    char_counts = Counter(hanzi_chars)

    word_zipf_lookup = {word: get_zipf(word) for word in unique_word_set}
    char_zipf_lookup = {ch: get_zipf(ch) for ch in unique_char_set}

    def annotate_word_list(pairs):
        return [(word, count, word_zipf_lookup.get(word, 0.0)) for word, count in pairs]

    def annotate_char_list(pairs):
        return [(ch, count, char_zipf_lookup.get(ch, 0.0)) for ch, count in pairs]

    annotated_common_words = annotate_word_list(word_counts.most_common())
    annotated_common_chars = annotate_char_list(char_counts.most_common())

    median_zipf_unique_words = median_zipf_unique(unique_word_set)
    median_zipf_word_tokens = median_zipf_weighted(word_counts, total_words)

    median_zipf_unique_chars = median_zipf_unique(unique_char_set)
    median_zipf_char_tokens = median_zipf_weighted(char_counts, total_chars)

    resolved_hsk_unique = {w: get_hsk_level(w) for w in unique_word_set}
    resolved_topn_unique = {w: get_topn_rank(w) for w in unique_word_set}

    token_hsk_rank_counts = Counter()
    token_topn_rank_counts = Counter()
    for word, count in word_counts.items():
        token_hsk_rank_counts[resolved_hsk_unique[word]] += count
        token_topn_rank_counts[resolved_topn_unique[word]] += count

    if total_words == 0:
        middle_1000_extract = ""
    else:
        if total_words <= 1000:
            middle_start_word = 0
            middle_end_word = total_words
        else:
            mid = total_words // 2
            half_window = 1000 // 2
            middle_start_word = max(0, mid - half_window)
            middle_end_word = middle_start_word + 1000

        extract_start = word_positions[middle_start_word][0]
        extract_end = word_positions[middle_end_word - 1][1]
        middle_1000_extract = re.sub(r"\s+", "", text[extract_start:extract_end])

    resolved_vocab_unique = {}
    if custom_vocab_set:
        for word in unique_word_set:
            resolved_vocab_unique[word] = vocab_known(word)

    not_in_vocab_words = []
    if custom_vocab_set:
        not_in_vocab_words = [
            (word, count, word_zipf_lookup.get(word, 0.0))
            for word, count, _ in annotated_common_words
            if not resolved_vocab_unique[word]
        ]

    not_in_hsk_words = [
        (word, count, word_zipf_lookup.get(word, 0.0))
        for word, count, _ in annotated_common_words
        if resolved_hsk_unique[word] > 7
    ]
    not_in_topn_words = [
        (word, count, word_zipf_lookup.get(word, 0.0))
        for word, count, _ in annotated_common_words
        if resolved_topn_unique[word] > 10000
    ]

    resolved_char_vocab = {}
    if custom_vocab_set:
        vocab_chars = {ch for item in custom_vocab_set for ch in item}
        resolved_char_vocab = {ch: ch in vocab_chars for ch in unique_char_set}

    resolved_char_hsk = {ch: CHAR_HSK_LEVEL.get(ch, 99) for ch in unique_char_set}
    resolved_char_topn = {ch: CHAR_TOPN_RANK.get(ch, 10**9) for ch in unique_char_set}

    not_in_vocab_chars = []
    if custom_vocab_set:
        not_in_vocab_chars = [
            (ch, count, char_zipf_lookup.get(ch, 0.0))
            for ch, count, _ in annotated_common_chars
            if not resolved_char_vocab.get(ch, False)
        ]

    not_in_hsk_chars = [
        (ch, count, char_zipf_lookup.get(ch, 0.0))
        for ch, count, _ in annotated_common_chars
        if resolved_char_hsk[ch] > 7
    ]
    not_in_topn_chars = [
        (ch, count, char_zipf_lookup.get(ch, 0.0))
        for ch, count, _ in annotated_common_chars
        if resolved_char_topn[ch] > 10000
    ]

    words_per_sentence = []
    wi = 0
    for s_start, s_end in sentence_spans:
        count = 0
        while wi < len(word_positions):
            w_start, w_end = word_positions[wi]
            if w_end <= s_start:
                wi += 1
                continue
            if w_start >= s_end:
                break
            count += 1
            wi += 1
        if count > 0:
            words_per_sentence.append(count)

    avg_words_sentence = (sum(words_per_sentence) / len(words_per_sentence)) if words_per_sentence else 0

    chars_per_sentence = []
    if sentence_spans:
        sentence_idx = 0
        next_sentence_end = sentence_spans[sentence_idx][1]
        sentence_char_count = 0
        for char_idx, char in enumerate(text):
            if HANZI_ONLY_RE.match(char):
                sentence_char_count += 1
            if char_idx + 1 == next_sentence_end:
                if sentence_char_count > 0:
                    chars_per_sentence.append(sentence_char_count)
                sentence_idx += 1
                if sentence_idx >= len(sentence_spans):
                    break
                next_sentence_end = sentence_spans[sentence_idx][1]
                sentence_char_count = 0

    avg_chars_sentence = (sum(chars_per_sentence) / len(chars_per_sentence)) if chars_per_sentence else 0

    phrasing_variety = np.nan
    if total_words >= WINDOW_SIZE:
        starts_all = np.arange(0, total_words - WINDOW_SIZE + 1, STEP_SIZE, dtype=np.int64)
        starts = starts_all
        if len(starts_all) > MAX_WINDOWS:
            idx = np.linspace(0, len(starts_all) - 1, MAX_WINDOWS)
            idx = np.unique(np.rint(idx).astype(np.int64))
            starts = starts_all[idx]
        window_scores = [len(set(words[s:s + WINDOW_SIZE])) for s in starts]
        phrasing_variety = int(round(statistics.median(window_scores)))

    char_variety = np.nan
    if total_chars >= WINDOW_SIZE:
        starts_all = np.arange(0, total_chars - WINDOW_SIZE + 1, STEP_SIZE, dtype=np.int64)
        starts = starts_all
        if len(starts_all) > MAX_WINDOWS:
            idx = np.linspace(0, len(starts_all) - 1, MAX_WINDOWS)
            idx = np.unique(np.rint(idx).astype(np.int64))
            starts = starts_all[idx]
        window_scores = [len(set(hanzi_chars[s:s + WINDOW_SIZE])) for s in starts]
        char_variety = int(round(statistics.median(window_scores)))

    top_cutoffs = [1000 * i for i in range(1, 11)]
    hsk_cutoffs = list(range(1, 8))

    unique_topn_rank_counts = Counter(resolved_topn_unique.values())
    unique_topN = cumulative_percent_from_rank_counts(unique_topn_rank_counts, unique_words, top_cutoffs)
    token_topN = cumulative_percent_from_rank_counts(token_topn_rank_counts, total_words, top_cutoffs)

    unique_char_topn_rank_counts = Counter(resolved_char_topn.values())
    token_char_topn_rank_counts = Counter()
    for ch, count in char_counts.items():
        token_char_topn_rank_counts[resolved_char_topn[ch]] += count

    unique_char_topN = cumulative_percent_from_rank_counts(unique_char_topn_rank_counts, unique_chars, top_cutoffs)
    token_char_topN = cumulative_percent_from_rank_counts(token_char_topn_rank_counts, total_chars, top_cutoffs)

    unique_hsk_rank_counts = Counter(resolved_hsk_unique.values())
    unique_maxhsk = cumulative_percent_from_rank_counts(unique_hsk_rank_counts, unique_words, hsk_cutoffs)
    token_maxhsk = cumulative_percent_from_rank_counts(token_hsk_rank_counts, total_words, hsk_cutoffs)

    unique_char_hsk_rank_counts = Counter(resolved_char_hsk.values())
    token_char_hsk_rank_counts = Counter()
    for ch, count in char_counts.items():
        token_char_hsk_rank_counts[resolved_char_hsk[ch]] += count

    unique_char_maxhsk = cumulative_percent_from_rank_counts(unique_char_hsk_rank_counts, unique_chars, hsk_cutoffs)
    token_char_maxhsk = cumulative_percent_from_rank_counts(token_char_hsk_rank_counts, total_chars, hsk_cutoffs)

    word_result = {
        "Total tokens": total_words,
        "Unique words": unique_words,
        "Unique words (% of tokens)": round((unique_words / total_words * 100), 1) if total_words else 0.0,
        "Median zipf (unique words)": median_zipf_unique_words,
        "Median zipf (all tokens)": median_zipf_word_tokens,
        "Average tokens per sentence": round(avg_words_sentence, 1),
        "Median unique words per 1000-token window": phrasing_variety,
    }

    if custom_vocab_set:
        word_result["Custom vocab unique word coverage (%)"] = round((sum(resolved_vocab_unique.values()) / unique_words * 100) if unique_words else 0, 1)
        word_result["Custom vocab token coverage (%)"] = round((sum(resolved_vocab_unique[word] * count for word, count in word_counts.items()) / total_words * 100) if total_words else 0, 1)

    for cutoff in top_cutoffs:
        word_result[f"Top {cutoff//1000}k unique word coverage (%)"] = round(unique_topN[cutoff], 1)
    for cutoff in top_cutoffs:
        word_result[f"Top {cutoff//1000}k token coverage (%)"] = round(token_topN[cutoff], 1)

    for lvl in range(1, 8):
        label_u = "HSK 1 to 7–9 unique word coverage (%)" if lvl == 7 else f"HSK 1 to {lvl} unique word coverage (%)"
        word_result[label_u] = round(unique_maxhsk[lvl], 1)
    for lvl in range(1, 8):
        label_t = "HSK 1 to 7–9 token coverage (%)" if lvl == 7 else f"HSK 1 to {lvl} token coverage (%)"
        word_result[label_t] = round(token_maxhsk[lvl], 1)

    top_values_limit = None  # capture full frequency lists without truncation

    word_result[WORD_LIST_UNIQUE] = tuple(annotated_common_words[:top_values_limit])
    if custom_vocab_set:
        word_result[WORD_LIST_NOT_IN_VOCAB] = tuple(not_in_vocab_words[:top_values_limit])
    word_result[WORD_LIST_NOT_IN_HSK] = tuple(not_in_hsk_words[:top_values_limit])
    word_result[WORD_LIST_NOT_IN_TOP] = tuple(not_in_topn_words[:top_values_limit])
    word_result["Middle 1000-token extract"] = middle_1000_extract

    char_result = {
        "Total characters": total_chars,
        "Unique characters": unique_chars,
        "Unique characters (% of characters)": round((unique_chars / total_chars * 100), 1) if total_chars else 0.0,
        "Median zipf (unique characters)": median_zipf_unique_chars,
        "Median zipf (all characters)": median_zipf_char_tokens,
        "Average characters per sentence": round(avg_chars_sentence, 1),
        "Median unique characters per 1000-character window": char_variety,
    }

    if custom_vocab_set:
        char_result["Custom vocab unique character coverage (%)"] = round((sum(resolved_char_vocab.values()) / unique_chars * 100) if unique_chars else 0, 1)
        char_result["Custom vocab character coverage (%)"] = round((sum(resolved_char_vocab[ch] * count for ch, count in char_counts.items()) / total_chars * 100) if total_chars else 0, 1)

    for cutoff in top_cutoffs:
        char_result[f"Top {cutoff//1000}k unique character coverage (%)"] = round(unique_char_topN[cutoff], 1)
    for cutoff in top_cutoffs:
        char_result[f"Top {cutoff//1000}k character coverage (%)"] = round(token_char_topN[cutoff], 1)

    for lvl in range(1, 8):
        label_u = "HSK 1 to 7–9 unique character coverage (%)" if lvl == 7 else f"HSK 1 to {lvl} unique character coverage (%)"
        char_result[label_u] = round(unique_char_maxhsk[lvl], 1)
    for lvl in range(1, 8):
        label_t = "HSK 1 to 7–9 character coverage (%)" if lvl == 7 else f"HSK 1 to {lvl} character coverage (%)"
        char_result[label_t] = round(token_char_maxhsk[lvl], 1)

    char_result[CHAR_LIST_UNIQUE] = tuple(annotated_common_chars[:top_values_limit])
    if custom_vocab_set:
        char_result[CHAR_LIST_NOT_IN_VOCAB] = tuple(not_in_vocab_chars[:top_values_limit])
    char_result[CHAR_LIST_NOT_IN_HSK] = tuple(not_in_hsk_chars[:top_values_limit])
    char_result[CHAR_LIST_NOT_IN_TOP] = tuple(not_in_topn_chars[:top_values_limit])

    return word_result, char_result


# =========================================================
# RENDERING HELPERS (INLINE)
# =========================================================
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


# =========================================================
# UI HELPERS (INLINE)
# =========================================================
def sanitize_dataframe_name(name):
    sanitized = name.replace(",", "").strip()
    return sanitized or "Unnamed text"


def clear_existing_file_results(state):
    removed = False
    for fname in list(state.word_results_dict.keys()):
        if fname.startswith("(Custom vocab)"):
            continue
        state.word_results_dict.pop(fname, None)
        state.char_results_dict.pop(fname, None)
        removed = True
    return removed


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
    st.header("File for analysis")
    input_method = st.radio(
        "Input method",
        ["Upload file", "Paste text"],
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
                progress_bar = st.progress(0, text="Preparing text upload...")
                tokenizer = get_tokenizer(state.custom_vocab_set)
                progress_bar.progress(40, text="Analysing text...")
                word_stats, char_stats = analyse_text(
                    pasted_text,
                    tokenizer,
                    state.custom_vocab_set,
                    resolve_vocab,
                    resolve_hsk_level,
                    resolve_topn_rank,
                )
                replaced_previous = clear_existing_file_results(state)
                existing_names = set(state.word_results_dict.keys())
                final_name = get_available_name(effective_name, existing_names)

                progress_bar.progress(85, text=f"Saving analysis: {final_name}")
                state.word_results_dict[final_name] = word_stats
                state.char_results_dict[final_name] = char_stats
                progress_bar.progress(100, text="Upload complete!")

                suffix = " Previous analysis cleared." if replaced_previous else ""

                if not stripped_name:
                    state.upload_status_toast = (
                        "warning",
                        f"No name provided. Using default name '{final_name}'.{suffix}",
                    )
                elif final_name != effective_name:
                    state.upload_status_toast = (
                        "warning",
                        f"Name already existed. Using '{final_name}' instead.{suffix}",
                    )
                else:
                    state.upload_status_toast = (
                        "success",
                        f"Uploaded as '{final_name}'.{suffix}",
                    )

                st.rerun()

    if input_method == "Upload file":
        uploaded_file = st.file_uploader(
            "Upload a single text file (.txt, .epub, .pdf, .csv)",
            type=["txt", "epub", "pdf", "csv"],
            accept_multiple_files=False,
            key=f"files_{state.uploader_key}",
        )

        if uploaded_file:
            tokenizer = get_tokenizer(state.custom_vocab_set)
            progress_bar = st.progress(0, text=f"Reading file: {uploaded_file.name}")

            if uploaded_file.name.endswith((".txt", ".csv")):
                text = uploaded_file.read().decode("utf-8", errors="ignore")
            elif uploaded_file.name.endswith(".epub"):
                text = extract_text_from_epub(uploaded_file)
            elif uploaded_file.name.endswith(".pdf"):
                text = extract_text_from_pdf(uploaded_file)
            else:
                st.error("Unsupported file type.")
                return

            progress_bar.progress(0.4, text=f"Analysing file: {uploaded_file.name}")
            word_stats, char_stats = analyse_text(
                text,
                tokenizer,
                state.custom_vocab_set,
                resolve_vocab,
                resolve_hsk_level,
                resolve_topn_rank,
            )

            replaced_previous = clear_existing_file_results(state)
            existing_names = set(state.word_results_dict.keys())
            cleaned_name = sanitize_dataframe_name(uploaded_file.name)
            result_name = get_available_name(cleaned_name, existing_names)

            progress_bar.progress(0.85, text=f"Saving analysis: {result_name}")
            state.word_results_dict[result_name] = word_stats
            state.char_results_dict[result_name] = char_stats
            gc.collect()

            progress_bar.progress(1.0, text="Upload complete!")

            toast_message = f"Uploaded file: {uploaded_file.name}"
            if replaced_previous:
                toast_message += " (previous analysis cleared)"

            state.upload_status_toast = (
                "success",
                toast_message,
            )

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
                isinstance(item, (tuple, list)) and len(item) in (2, 3)
                for item in entries
            ):
                formatted_items = []
                for item in entries:
                    parts = [str(item[0]), str(item[1])]
                    if len(item) == 3:
                        zipf_value = item[2]
                        if isinstance(zipf_value, numbers.Real):
                            parts.append(f"{zipf_value:.2f}")
                        else:
                            parts.append(str(zipf_value))
                    formatted_items.append(":".join(parts))
                return "; ".join(formatted_items)
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
                        "WordFreq Zipf" in col,
                        "extract" in col.lower(),
                    ]
                ),
                "Top N coverage": lambda col: col.startswith("Top "),
                "HSK coverage": lambda col: col.startswith("HSK "),
                "Frequency lists": lambda col: "WordFreq Zipf" in col,
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
        if "WordFreq Zipf" in stripped:
            return False
        if "Middle 1000-token extract" in stripped:
            return False
        return True

    columns = [col for col in df.columns if should_include(col)]
    return df.loc[:, columns]


def build_frequency_table(data, item_label, total_count, occurrence_unit):
    zipf_column_label = "WordFreq Zipf (1 in __ words)"
    columns = [
        item_label,
        "Occurrences in text",
        f"Occurs every __ {occurrence_unit}",
        zipf_column_label,
    ]

    if not data:
        empty_df = pd.DataFrame(columns=columns)
        empty_df.index.name = "#"
        return empty_df

    df = pd.DataFrame(data, columns=[item_label, "Occurrences in text", "WordFreq Zipf"])

    if total_count > 0:
        occurrence_ratio = (total_count / df["Occurrences in text"]).round(0).astype(int).clip(lower=1)
        df[f"Occurs every __ {occurrence_unit}"] = occurrence_ratio.map(lambda x: f"1 in {x:,}")
    else:
        df[f"Occurs every __ {occurrence_unit}"] = "—"

    df[zipf_column_label] = [
        format_zipf_with_ratio(zipf, wordfreq_ratio_from_zipf(zipf), "words")
        for zipf in df["WordFreq Zipf"]
    ]

    df = df.drop(columns=["WordFreq Zipf"])
    df = df[
        [
            item_label,
            "Occurrences in text",
            f"Occurs every __ {occurrence_unit}",
            zipf_column_label,
        ]
    ]

    df.index = range(1, len(df) + 1)
    df.index.name = "#"
    return df


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
            .reset_index(name="Occurences")
        )

        char_df = (
            pd.Series(demo_chars)
            .value_counts()
            .rename_axis("Character")
            .reset_index(name="Occurences")
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
            st.caption(
                "WordFreq Zipf scores come from the WordFreq corpus: 7 ≈ 1 in 100 words, 6 ≈ 1 in 1,000, 5 ≈ 1 in 10,000. "
                "The ratio shown in parentheses gives an approximate real-world frequency."
            )
            
            word_sections = [("All Words", wrow.get(WORD_LIST_UNIQUE, []))]
            if has_custom_vocab:
                word_sections.append(("Not in Custom Vocab", wrow.get(WORD_LIST_NOT_IN_VOCAB, [])))
            word_sections.extend([
                ("Not in HSK Wordlist", wrow.get(WORD_LIST_NOT_IN_HSK, [])),
                ("Not in Top 10k Most Common Words", wrow.get(WORD_LIST_NOT_IN_TOP, [])),
            ])

            for tab, (label, data) in zip(st.tabs([x[0] for x in word_sections]), word_sections):
                with tab:
                    df_word = build_frequency_table(
                        data,
                        "Word",
                        total_tokens,
                        "words",
                    )

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
            st.caption(
                "WordFreq Zipf for characters treats each Hanzi as a single-character word in the WordFreq corpus. "
                "The parentheses show how often that character appears on average across written text."
            )
            
            char_sections = [("All Characters", crow.get(CHAR_LIST_UNIQUE, []))]
            if has_custom_vocab:
                char_sections.append(("Not in Custom Vocab", crow.get(CHAR_LIST_NOT_IN_VOCAB, [])))
            char_sections.extend([
                ("Not in HSK Wordlist", crow.get(CHAR_LIST_NOT_IN_HSK, [])),
                ("Not in Top 10k Most Common Words", crow.get(CHAR_LIST_NOT_IN_TOP, [])),
            ])

            for tab, (label, data) in zip(st.tabs([x[0] for x in char_sections]), char_sections):
                with tab:
                    df_char = build_frequency_table(
                        data,
                        "Character",
                        total_chars,
                        "chars",
                    )

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
