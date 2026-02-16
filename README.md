# Chinese Text Lexical Analyser

A Streamlit application for analysing modern Simplified Chinese text at both the **word-token level** and the **character (Hanzi) level**.

Live app:  
👉 https://chinese-text-lexical-analyser.streamlit.app/

---

# Overview

Chinese reading difficulty is multidimensional. A text can:

- Use common characters but rare word combinations  
- Use common words but long, complex sentences  
- Stay within HSK vocabulary but rely heavily on low-frequency literary terms  
- Introduce many new characters even when words look familiar  

This tool separates and measures these components explicitly.

It analyses:

- Word tokens (segmented using jieba)
- Individual characters (Hanzi)

And calculates:

- Text length
- Vocabulary size
- Sentence statistics
- Sliding-window lexical diversity
- Word frequency coverage (Top 1k–10k bands)
- HSK 3.0 coverage (Levels 1 to 7–9)
- Optional custom vocabulary coverage
- Zipf frequency profiles
- Frequency tables
- A representative 1,000-token midpoint extract with highlighting options

---

# Core Concept: Word-Level vs Character-Level Difficulty

Chinese difficulty does not operate at only one level.

Example:

- A text might use familiar characters (e.g. 人, 生, 活)
- But combine them into less common words (e.g. 人生观, 生活化)

This app reports statistics separately for:

| Words | Characters |
|-------|------------|
| Segmented lexical units | Individual Hanzi |
| Word frequency (Zipf) | Character frequency (Zipf) |
| HSK word levels | Character-level HSK mapping |
| Token coverage | Character coverage |

This allows you to distinguish:

- Vocabulary difficulty
- Glyph recognition difficulty

---

# How Word Segmentation Works

Words are segmented using **jieba**, a dictionary-based Chinese tokenizer.

Segmentation rules:

- Words from the WordFreq top 10k list are added
- Words from the official HSK 3.0 list are added
- Custom vocabulary (if uploaded) is added
- Higher-probability segmentation paths are preferred

Example:

- 中国 → 1 token
- 中国人 → 2 tokens (中国 + 人)
- But 生活化 → 1 token if recognised

Only tokens consisting entirely of Hanzi are analysed.

The following are ignored in calculations:

- English words
- Numbers
- Pinyin
- Punctuation
- Latin annotations

This makes the tool suitable for bilingual or annotated texts.

---

# Metrics Explained

## 1. Volume Metrics

### Word Level
- Total tokens
- Unique words
- Unique words (% of tokens)

### Character Level
- Total characters
- Unique characters
- Unique characters (% of total)

Interpretation:

- High unique word % → frequent vocabulary refresh
- High unique character % → many new glyphs introduced
- High word diversity but low character diversity → recombination of familiar characters
- High character diversity → broader recognition demands

---

## 2. Sentence Statistics

- Average tokens per sentence
- Average characters per sentence

Longer sentences in tokens usually indicate:

- Subordination
- Clause chaining
- Complex syntactic structure

Comparing tokens vs characters shows whether length comes from:

- Many short words
- Or fewer long compounds

---

## 3. Sliding-Window Lexical Diversity

Instead of using simple type–token ratio (which is unstable), the tool computes:

- Median unique words per 1000-token window
- Median unique characters per 1000-character window

Windows slide with overlap (step size 250).

This provides a stable measure of:

- How quickly vocabulary refreshes
- How repetitive the text is locally

Higher values indicate:

- Greater lexical variation
- Broader vocabulary usage

Lower values indicate:

- High repetition (common in graded readers)

---

## 4. Zipf Frequency Profile

Zipf frequency is the base-10 logarithm of occurrences per billion words in the WordFreq corpus.

Approximate interpretation:

| Zipf | Approximate Frequency |
|------|----------------------|
| 8    | 1 in 10 words        |
| 7    | 1 in 100 words       |
| 6    | 1 in 1,000 words     |
| 5    | 1 in 10,000 words    |
| 4    | 1 in 100,000 words   |
| 3    | 1 in 1,000,000 words |

The app reports:

- Median Zipf (unique words)
- Median Zipf (all tokens)
- Median Zipf (unique characters)
- Median Zipf (all characters)

Interpretation:

- Unique median → how rare the vocabulary set is overall
- Token median → what readers repeatedly encounter

If token median is much higher than unique median:
→ rare words appear but only occasionally

If both are similar:
→ rare words appear frequently

---

## 5. HSK 3.0 Coverage

Coverage is cumulative across levels:

- HSK 1
- HSK 1–2
- HSK 1–3
- …
- HSK 1–7–9

Reported separately for:

- Token coverage
- Unique coverage
- Word level
- Character level

Interpretation:

- Token coverage → how much of the running text falls within a level
- Unique coverage → how much of the vocabulary inventory falls within that level

Character-level HSK mapping assigns each character the **lowest HSK level of any word containing it**.

---

## 6. Top-N Frequency Coverage (WordFreq)

Cumulative frequency bands:

- Top 1k
- Top 2k
- …
- Top 10k

Reported separately for:

- Word tokens
- Unique words
- Characters
- Unique characters

Interpretation:

- Token coverage → how much of the running text uses high-frequency vocabulary
- Unique coverage → how much of the vocabulary inventory comes from common bands

This measures general-language frequency rather than syllabus alignment.

---

## 7. Custom Vocabulary Coverage (Optional)

You may upload a `.txt` file with:

- One Chinese word per line
- Hanzi only
- No pinyin
- No punctuation

Lines containing non-Hanzi characters (including Pleco headers such as `//`) are automatically ignored.

The tool reports:

- Custom vocab token coverage
- Custom vocab unique coverage
- Character-level custom coverage

This allows:

- Personal reading-level analysis
- Vocabulary tracking
- Study gap identification

---

# Frequency Tables

For both words and characters, the app displays:

- All items (sorted by frequency in the text)
- Items not in custom vocab
- Items not in HSK
- Items not in Top 10k

Each row includes:

- Occurrences in text
- Approximate occurrence rate (1 in X words)
- WordFreq Zipf score

Large tables are truncated by default (100 rows), with an option to show all.

---

# Sample Extract Tab

The app extracts a midpoint sample of:

- 1,000 word tokens (if available)

This provides a representative snapshot of:

- Vocabulary distribution
- Sentence structure
- Frequency patterns

Highlighting modes include:

- Word: custom vocab
- Word: HSK
- Word: Top-N
- Character: custom vocab
- Character: HSK
- Character: Top-N

A color legend explains each band.

---

# Supported Input Formats

- `.txt`
- `.csv`
- `.epub`
- `.pdf` (text-based only)

Scanned-image PDFs will not work.

You may also paste up to 15,000 characters directly into the app.

---

# Data Sources

## HSK Vocabulary

This app uses the official **HSK 3.0 vocabulary list (December 2025 edition)**:

- Official HSK 3.0 vocabulary list:  
  https://www.marteagency.com/pdf/%E6%96%B0%E7%89%88HSK%E8%80%83%E8%AF%95%E5%A4%A7%E7%BA%B2%E8%AF%8D%E6%B1%87.pdf

The HSK coverage metrics are cumulative across levels:

- HSK 1  
- HSK 1–2  
- HSK 1–3  
- …  
- HSK 1–7–9  

Character-level HSK mapping assigns each Hanzi the **lowest HSK level of any listed word that contains that character**.

---

## Word Frequency Data

This app uses the open-source **WordFreq** database:

- WordFreq GitHub repository:  
  https://github.com/rspeer/wordfreq

The WordFreq snapshot used reflects language usage up to approximately **2021**, meaning it is largely based on pre–large-language-model corpora and is minimally influenced by AI-generated text.

Zipf frequency scores are calculated using WordFreq’s Chinese corpus, where:

- Zipf is the base-10 logarithm of occurrences per billion words.
- Example: Zipf 7 ≈ 1 in 100 words; Zipf 3 ≈ 1 in 1,000,000 words.

### WordFreq Sources Include

WordFreq aggregates data from multiple large corpora, including:

- **Wikipedia** (encyclopedic text)
- **Subtitles** (OPUS OpenSubtitles 2018 + SUBTLEX)
- **News** (NewsCrawl 2014 + GlobalVoices)
- **Books** (Google Books Ngrams 2012)
- **Web text** (OSCAR corpus)
- **Twitter** (short-form social media)
- **Miscellaneous word frequencies**, including the free wordlist bundled with the jieba word segmenter

This produces a broad, mixed-domain estimate of real-world written language usage.

---

# Running the App Locally

## 1. Clone the repository

```bash
git clone https://github.com/your-username/chinese-text-lexical-analyser.git
cd chinese-text-lexical-analyser
```

## 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows
```

## 3. Install dependencies

If a requirements file exists:

```bash
pip install -r requirements.txt
```

Otherwise:

```bash
pip install streamlit jieba pandas numpy pdfplumber beautifulsoup4 ebooklib wordfreq
```

## 4. Run the app

```bash
streamlit run app.py
```

Open the local URL shown in your terminal (usually http://localhost:8501).

---

# Deployment

You can deploy using:

- Streamlit Community Cloud
- Hugging Face Spaces
- Railway
- Any Python server supporting Streamlit

For Streamlit Cloud:

1. Push to GitHub
2. Create a new app
3. Select `app.py` as entry point

---

# Intended Use Cases

- Graded reader difficulty comparison
- HSK syllabus alignment
- Web novel lexical profiling
- Textbook evaluation
- Corpus comparison
- Chinese reading research
- Personal vocabulary gap analysis

---

# Limitations

This tool does not directly measure:

- Grammar structures
- Discourse markers
- Syntactic trees
- Pragmatic difficulty

Reading difficulty should be interpreted as the interaction of:

- Sentence length
- Lexical diversity
- Frequency profile
- Coverage statistics

No single metric should be used in isolation.
