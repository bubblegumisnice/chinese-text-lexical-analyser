"""Jieba tokenizer helpers shared by the Streamlit app."""

import jieba
import streamlit as st

try:
    from app import SEGMENTATION_WORDS
except ModuleNotFoundError:  # When executed via `streamlit run app.py`
    from __main__ import SEGMENTATION_WORDS


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
