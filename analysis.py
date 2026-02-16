"""Core text analysis helpers extracted from the Streamlit app."""

from collections import Counter
import re
import statistics

import numpy as np
import pdfplumber
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
try:
    from wordfreq import zipf_frequency
except ModuleNotFoundError:
    def zipf_frequency(_term, _lang):
        return 0.0


from config import (
    WINDOW_SIZE,
    STEP_SIZE,
    MAX_WINDOWS,
    HANZI_RE,
    HANZI_ONLY_RE,
    SENTENCE_SPLIT_RE,
)

try:
    from app import (
        WORD_RANK,
        HSK_MAP,
        CHAR_HSK_LEVEL,
        CHAR_TOPN_RANK,
    )
except ModuleNotFoundError:  # When app.py is executed as __main__
    from __main__ import (
        WORD_RANK,
        HSK_MAP,
        CHAR_HSK_LEVEL,
        CHAR_TOPN_RANK,
    )


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
    most_common_all = word_counts.most_common()
    char_counts = Counter(hanzi_chars)
    char_most_common_all = char_counts.most_common()

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
        not_in_vocab_words = [(word, count) for word, count in most_common_all if not resolved_vocab_unique[word]]

    not_in_hsk_words = [(word, count) for word, count in most_common_all if resolved_hsk_unique[word] > 7]
    not_in_topn_words = [(word, count) for word, count in most_common_all if resolved_topn_unique[word] > 10000]

    resolved_char_vocab = {}
    if custom_vocab_set:
        vocab_chars = {ch for item in custom_vocab_set for ch in item}
        resolved_char_vocab = {ch: ch in vocab_chars for ch in unique_char_set}

    resolved_char_hsk = {ch: CHAR_HSK_LEVEL.get(ch, 99) for ch in unique_char_set}
    resolved_char_topn = {ch: CHAR_TOPN_RANK.get(ch, 10**9) for ch in unique_char_set}

    not_in_vocab_chars = []
    if custom_vocab_set:
        not_in_vocab_chars = [
            (ch, count)
            for ch, count in char_most_common_all
            if not resolved_char_vocab.get(ch, False)
        ]

    not_in_hsk_chars = [(ch, count) for ch, count in char_most_common_all if resolved_char_hsk[ch] > 7]
    not_in_topn_chars = [
        (ch, count)
        for ch, count in char_most_common_all
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

    word_result["Unique words (with frequencies)"] = tuple(most_common_all[:top_values_limit])
    if custom_vocab_set:
        word_result["Words not in vocab (with frequencies)"] = tuple(not_in_vocab_words[:top_values_limit])
    word_result["Words not in HSK (with frequencies)"] = tuple(not_in_hsk_words[:top_values_limit])
    word_result["Words not in top 10k most common words (with frequencies)"] = tuple(not_in_topn_words[:top_values_limit])
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

    char_result["Unique characters (with frequencies)"] = tuple(char_most_common_all[:top_values_limit])
    if custom_vocab_set:
        char_result["Characters not in vocab (with frequencies)"] = tuple(not_in_vocab_chars[:top_values_limit])
    char_result["Characters not in HSK (with frequencies)"] = tuple(not_in_hsk_chars[:top_values_limit])
    char_result["Characters not in top 10k most common words (with frequencies)"] = tuple(not_in_topn_chars[:top_values_limit])

    return word_result, char_result
