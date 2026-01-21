import os
import time
import string
import requests
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_QUERIES = [
    "Trump shooting rally",
    "Trump assassination attempt",
    "Trump Pennsylvania rally 2024",
    "Trump July 2024 news",
    "Trump rally protest",
    "Trump Pennsylvania news",
    "Trump July 2024 incident",
    "Trump rally media coverage",
    "Trump security breach",
    "Trump rally attack",
    "Trump rally 2024 analysis",
    "Trump rally response 2024",
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "combined_trump_data_cleaned.csv")


def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def preprocess_text(text: str):
    if not isinstance(text, str):
        return []
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    sw = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in sw]
    return tokens


def fetch_gdelt_articles(query: str, maxrecords: int = 250, timeout: int = 30) -> pd.DataFrame:
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": maxrecords,
        "format": "json",
    }

    r = requests.get(GDELT_BASE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    articles = data.get("articles", [])
    if not articles:
        return pd.DataFrame()

    df = pd.DataFrame(articles)
    df["query_used"] = query
    return df


def load_existing(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def clean_and_merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, new], ignore_index=True)

    # Keep only key columns if present (prevents schema drift issues)
    keep_cols = [c for c in ["url", "title", "content", "seendate", "sourceCountry", "sourceCollection", "domain", "query_used"] if c in combined.columns]
    combined = combined[keep_cols].copy()

    # Basic quality filters
    if "url" in combined.columns:
        combined.drop_duplicates(subset="url", inplace=True)
        combined.dropna(subset=["url"], inplace=True)

    if "title" in combined.columns:
        combined.dropna(subset=["title"], inplace=True)

    # content can be missing sometimes; keep if present but not mandatory
    if "content" in combined.columns:
        combined["content"] = combined["content"].fillna("")

    return combined


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    ensure_nltk()

    queries = DEFAULT_QUERIES
    sleep_s = float(os.getenv("GDELT_SLEEP_SECONDS", "0.3"))

    existing = load_existing(OUTPUT_CSV)
    all_new = []

    for q in queries:
        try:
            df = fetch_gdelt_articles(q)
            if not df.empty:
                all_new.append(df)
        except requests.HTTPError as e:
            print(f"[HTTPError] {e} | query='{q}'")
        except Exception as e:
            print(f"[Error] {e} | query='{q}'")

        time.sleep(sleep_s)

    if not all_new:
        print("No new articles collected from GDELT.")
        return

    new_data = pd.concat(all_new, ignore_index=True)
    combined = clean_and_merge(existing, new_data)

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV} | total_rows={combined.shape[0]}")


if __name__ == "__main__":
    main()
