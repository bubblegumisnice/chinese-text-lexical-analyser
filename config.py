"""Shared configuration constants for the lexical analyser."""

import re

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


# Regex helpers shared by analysis modules
HANZI_RE = re.compile(r"[\u4e00-\u9fff]")
HANZI_ONLY_RE = re.compile(r"^[\u4e00-\u9fff]+$")
SENTENCE_SPLIT_RE = re.compile(r"[。！？]+")
