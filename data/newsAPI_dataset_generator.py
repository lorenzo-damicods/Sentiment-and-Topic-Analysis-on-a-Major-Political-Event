import os
import time
import requests
import pandas as pd


NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

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

OUT_DIR = "dataset"
OUT_FILE = os.path.join(OUT_DIR, "newsapi_articles.csv")


def _require_api_key() -> str:
    key = os.getenv("NEWSAPI_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Missing NEWSAPI_KEY. Set it like:\n"
            "export NEWSAPI_KEY='YOUR_KEY_HERE'"
        )
    return key


def fetch_newsapi_articles(
    query: str,
    api_key: str,
    language: str = "en",
    page_size: int = 100,
    max_pages: int = 5,
    sleep_s: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch articles from NewsAPI 'everything' endpoint.
    Notes:
      - NewsAPI may cap results depending on plan (often 100 results for free).
      - Pagination supported via 'page' param.
    """
    all_rows = []

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "apiKey": api_key,
            "language": language,
            "pageSize": page_size,
            "page": page,
            "sortBy": "publishedAt",
        }

        resp = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=30)
        if resp.status_code == 429:
            # Rate limit hit
            time.sleep(max(2.0, sleep_s))
            continue

        resp.raise_for_status()
        data = resp.json()

        articles = data.get("articles", [])
        if not articles:
            break

        for a in articles:
            all_rows.append(
                {
                    "source": (a.get("source") or {}).get("name"),
                    "author": a.get("author"),
                    "title": a.get("title"),
                    "description": a.get("description"),
                    "url": a.get("url"),
                    "published_at": a.get("publishedAt"),
                    "content": a.get("content"),
                    "query": query,
                }
            )

        time.sleep(sleep_s)

    return pd.DataFrame(all_rows)


def load_existing(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def clean_and_merge(existing: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, new_data], ignore_index=True)

    # Drop rows without essential fields
    combined = combined.dropna(subset=["url", "title"])

    # Deduplicate by URL
    combined = combined.drop_duplicates(subset=["url"])

    # Basic cleanup: strip strings
    for col in ["source", "author", "title", "description", "url", "content", "query"]:
        if col in combined.columns:
            combined[col] = combined[col].astype(str).str.strip()

    return combined


def main():
    api_key = _require_api_key()

    os.makedirs(OUT_DIR, exist_ok=True)

    existing = load_existing(OUT_FILE)

    frames = []
    for q in DEFAULT_QUERIES:
        print(f"[NewsAPI] Fetching: {q}")
        df_q = fetch_newsapi_articles(
            query=q,
            api_key=api_key,
            language="en",
            page_size=100,
            max_pages=5,
            sleep_s=1.0,
        )
        print(f"  -> fetched rows: {len(df_q)}")
        frames.append(df_q)

    new_data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined = clean_and_merge(existing, new_data)

    combined.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE} (total rows: {len(combined)})")


if __name__ == "__main__":
    main()
