from tqdm import tqdm
import requests
import pandas as pd
import uuid
from bs4 import BeautifulSoup
import time
import random

def fetch_search_results(search_term="AI", size=100, offset=0, page=1):
    """
    CNN 내부 XHR API에서 검색 결과 메타데이터를 가져와 DataFrame으로 반환합니다.
    """
    url = "https://search.prod.di.api.cnn.io/content"
    params = {
        "q":           search_term,
        "size":        size,
        "from":        offset,
        "page":        page,
        "sort":        "newest",
        "types":       "article",
        "site":        "cnn",
        "request_id":  f"stellar-search-{uuid.uuid4()}"
    }
    headers = {
        "User-Agent":       "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept":           "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Referer":          f"https://edition.cnn.com/search?q={search_term}",
    }
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data.get("result", []))

def scrape_article(url, user_agent):
    """
    단일 기사 페이지에서 제목, 저자, 날짜, 본문을 추출하여 dict로 반환합니다.
    """
    if url.startswith("/"):
        url = "https://edition.cnn.com" + url

    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.select_one("#maincontent")
    title = title_tag.get_text(strip=True) if title_tag else ""

    author_tag = soup.select_one(
        "body > div.layout__content-wrapper.layout-with-rail__content-wrapper "
        "> section.layout__top.layout-with-rail__top "
        "> div.headline.headline--has-lowertext "
        "> div.headline__footer "
        "> div.headline__sub-container "
        "> div > div.byline.vossi-byline "
        "> div.byline__names.vossi-byline__names > span"
    )
    authors = author_tag.get_text(strip=True) if author_tag else ""

    date_tag = soup.select_one(".timestamp.vossi-timestamp")
    date_text = date_tag.get_text(strip=True) if date_tag else ""

    content_tags = soup.select("div.article__content p")
    if not content_tags:
        content_tags = soup.select("article p")
    content = "\n".join(p.get_text(strip=True) for p in content_tags)

    return {
        "article_title": title,
        "authors":       authors,
        "date":          date_text,
        "content":       content
    }

def main():
    search_term = "AI"
    size = 100
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    all_records = []

    for page in tqdm(range(1, 7), desc="Pages"):
        offset = (page - 1) * 99
        df_meta = fetch_search_results(search_term, size, offset, page)
        if df_meta.empty:
            tqdm.write(f"Page {page}: no more results, stopping.")
            break

        for _, row in tqdm(df_meta.iterrows(), total=len(df_meta),
                            desc=f"Page {page} articles", leave=False):
            url = row.get("url") or row.get("path", "")
            if not url:
                continue
            rec = {
                "search_headline": row.get("headline", ""),
                "url":             url
            }
            rec.update(scrape_article(url, user_agent))
            all_records.append(rec)
            time.sleep(random.uniform(1, 3))

        time.sleep(random.uniform(1, 3))

    df = pd.DataFrame(all_records, columns=[
        "search_headline", "url", "article_title", "authors", "date", "content"
    ])
    df.to_csv("cnn_ai_articles_pages_1_to_6.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()