
# 🗳️ Global Media Framing of the Trump Assassination Attempt
### Sentiment and Topic Analysis · BERT + LDA · July 2024

> **Course:** Computational Social Science — University of Trento, Department of Sociology and Social Research
> **Author:** Lorenzo D'Amico (Student ID: 238684)

---

## 📌 Overview

This project analyzes global media coverage of the **July 2024 assassination attempt on Donald Trump** using NLP techniques. Over **3,700 news articles** were collected from domestic and international outlets via NewsAPI and GDELT, then processed with **BERT** for sentiment classification and **LDA** for topic modeling.

The central research question: *How did various media outlets cover the event, and what were the prevailing sentiments and themes?*

---

## 🔑 Key Results

| Metric | Value |
|--------|-------|
| Total articles collected | 3,700+ |
| Collection period | July – mid-August 2024 |
| Data sources | NewsAPI + GDELT |
| BERT model | `nlptown/bert-base-multilingual-uncased-sentiment` |
| Sentiment scale | 1 star (highly negative) → 5 stars (highly positive) |
| LDA topics identified | 8 |
| LDA training passes | 15 |
| Dominant sentiment | **1 star** (highly negative) across most outlets |
| Largest LDA topic | Topic 1 — U.S. Presidential Politics (29.9% of tokens) |

**Key finding:** Media coverage was predominantly negative (1–2 stars), driven by security failure narratives and political criticism. Positive sentiment (4–5 stars) was concentrated in pro-Trump outlets. LDA revealed 8 distinct thematic clusters with clear sentiment-topic interactions — e.g., security topics correlated strongly with negative sentiment, while election-speculation topics skewed neutral.

---

## 🧠 Methodology

### 1. Data Collection
- **NewsAPI** and **GDELT** APIs used to collect multilingual articles
- Source file: `combined_trump_data_cleaned.csv`
- Fields: `source`, `author`, `title`, `description`, `url`, `publishedAt`, `content`, `collectedAt`

### 2. Text Preprocessing (`NLTK`, `SpaCy`)
- HTML tag and punctuation removal
- Lowercasing
- Tokenization + Lemmatization (WordNet)
- Stopword removal
- Missing value handling (drop or fill)
- BERT truncation at 512 tokens

### 3. Sentiment Analysis — BERT
- Model: `nlptown/bert-base-multilingual-uncased-sentiment` (HuggingFace)
- Multilingual model chosen to handle international coverage
- Output: 1–5 star classification per article
- Visualized: sentiment distribution by source (top 10 outlets) and by LDA topic

### 4. Topic Modeling — LDA (`Gensim`)
- Bag-of-Words representation via `corpora.Dictionary`
- `LdaModel` trained with `num_topics=8`, `passes=15`
- Number of topics chosen empirically for interpretability
- Visualized with **pyLDAvis** (interactive intertopic distance map)
- Dominant topic assigned per document

### 5. Combined Analysis
- Dominant topic merged into main dataframe
- Sentiment distribution computed per topic
- Word frequency analysis (Top 20 most frequent terms)

---

## 📊 The 8 LDA Topics

| # | Topic Label | Token Share | Dominant Sentiment |
|---|-------------|-------------|-------------------|
| 1 | U.S. Presidential Politics | 29.9% | Negative |
| 2 | Republican Party & Events | — | Negative |
| 3 | Media & Political News | — | Negative |
| 4 | Media Reporting & Political Events | — | Mixed |
| 5 | National Events & Trump Influence | — | Negative (highest count) |
| 6 | Cybersecurity & Elections | — | Mixed/Neutral |
| 7 | Political Assassination Attempts | — | Negative |
| 8 | Presidential Security | — | Negative |

---

## 📂 Repository Structure

```
├── data/
│   └── combined_trump_data_cleaned.csv        # Cleaned dataset (3,700+ articles)
│   └── GDELT_API_dataset_generator.py             # GDELT data collection script
│   └── newsAPI_dataset_generator.py               # NewsAPI data collection script
├── report/
│   └── Project_Report.pdf                     # Full academic report
├── notebook                                   # Jupyter notebook (full pipeline)
├── requirements                               # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- GPU recommended for BERT inference (CPU works but is slow on 3,700 articles)

### Install dependencies
```bash
pip install -r requirements
```

Key libraries used:
```
pandas numpy
transformers torch          # BERT sentiment analysis
gensim pyLDAvis             # LDA topic modeling
nltk spacy                  # Text preprocessing
matplotlib seaborn          # Visualization
```

### Run the notebook
```bash
jupyter notebook notebook
```

The notebook runs the full pipeline sequentially:
1. Data loading
2. Text preprocessing
3. BERT sentiment analysis
4. LDA topic modeling + pyLDAvis visualization
5. Combined sentiment-per-topic analysis
6. Word frequency analysis

> **Note:** BERT inference on 3,700+ articles is time-intensive. The dataset includes a pre-computed `sentiment_score` column so you can skip re-running inference and jump directly to analysis cells.

---

## 🔐 API Keys (Not included in repo)

The raw data collection required:
- **NewsAPI** key → [newsapi.org](https://newsapi.org) — add your key in `newsAPI_dataset_generator.py`
- **GDELT** access → no key needed (public API)

The cleaned dataset (`combined_trump_data_cleaned.csv`) is already included in `data/` so collection does not need to be re-run.

---

## 📤 What's Uploaded to GitHub / What's Not

| File | Uploaded | Reason |
|------|----------|--------|
| `notebook` | ✅ Yes | Full analysis pipeline |
| `data/combined_trump_data_cleaned.csv` | ✅ Yes | Cleaned dataset, needed for reproducibility |
| `report/Project_Report.pdf` | ✅ Yes | Academic report |
| `GDELT_API_dataset_generator.py` | ✅ Yes | Data collection script |
| `newsAPI_dataset_generator.py` | ✅ Yes | Data collection script |
| `requirements` | ✅ Yes | Dependency list |
| `README.md` | ✅ Yes | This file |
| Raw uncleaned API dumps | ❌ No | Redundant |
| API keys / `.env` files | ❌ No | Security |
| `__pycache__/`, `.ipynb_checkpoints/` | ❌ No | Auto-generated artifacts |
| Model weights / cache | ❌ No | Downloaded automatically by HuggingFace at runtime |

---

## ⚠️ Limitations

- BERT truncates articles at 512 tokens — long articles lose tail content
- `nlptown/bert-base-multilingual-uncased-sentiment` was not fine-tuned on political news; sentiment labels are approximate
- LDA topic count (k=8) was chosen empirically, not via coherence score optimization
- Class imbalance: 1-star articles dominate, which may bias topic-level sentiment aggregations
- GDELT and NewsAPI have their own collection biases (English-language overrepresentation)

---

## 🔭 Future Work & Modern Alternatives

This project was built with BERT + LDA, a solid and reproducible baseline. The NLP landscape has however evolved significantly, and the following directions would represent meaningful upgrades:

**Topic Modeling** — [BERTopic](https://github.com/MaartenGr/BERTopic) replaces LDA with sentence embeddings + HDBSCAN clustering, eliminating the need to fix `k` manually and producing semantically richer, context-aware topics. In this setting it would likely yield cleaner separation between narratives like "Secret Service failure" and "electoral rhetoric" that LDA tends to conflate.

**Sentiment Analysis** — The 1–5 star classification used here captures intensity but not structure. Aspect-Based Sentiment Analysis (ABSA) would allow extracting separate sentiment for each named entity per article (Trump, Harris, Secret Service, etc.), which is qualitatively more informative for political text.

**LLM-as-Classifier** — Using a large language model (e.g. GPT-4o, Claude) as an interpretive layer would allow generating natural-language frame labels per article ("institutional crisis", "leadership resilience", "electoral opportunism"), going beyond numeric scores toward narrative analysis.

**The limit that remains** — Even with more powerful tools, the core finding of this project holds: media polarization is a social phenomenon, not a measurement problem. Explaining it causally would require crossing NLP data with outlet ownership, funding sources, and audience data — a direction that sits at the intersection of computational methods and media sociology.

---

## 📚 References

- Blei, D. et al. (2003). Latent Dirichlet Allocation. JMLR.
- Bird, S., Klein, E., Loper, E. (2009). Natural Language Processing with Python. O'Reilly.
- Liu, B. (2012). Sentiment Analysis and Opinion Mining. Morgan & Claypool.
- NewsAPI: [newsapi.org](https://newsapi.org) · GDELT: [gdeltproject.org](https://gdeltproject.org)
- HuggingFace model: [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
