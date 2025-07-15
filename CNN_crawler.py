from tqdm import tqdm
import requests
import pandas as pd
import uuid
from bs4 import BeautifulSoup
import time
import random
import argparse

def fetch_search_results(search_term="AI", size=10, offset=0, page=1):
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
        "div.byline.vossi-byline span"
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
    parser = argparse.ArgumentParser(description="CNN 검색어로 기사 크롤링")
    parser.add_argument("-q", "--search_term", type=str, default="AI",
                        help="검색어 (기본: AI)")
    parser.add_argument("-s", "--size", type=int, default=10,
                        help="한 페이지당 가져올 기사 수 (기본: 10)")
    parser.add_argument("-p", "--pages", type=int, default=6,
                        help="스크래핑할 페이지 수 (기본: 6)")
    parser.add_argument("-o", "--output", type=str,
                        default="cnn_articles.csv",
                        help="저장할 CSV 파일명 (기본: cnn_articles.csv)")
    parser.add_argument("--delay-min", type=float, default=1.0,
                        help="각 요청 사이 최소 딜레이 (초, 기본: 1.0)")
    parser.add_argument("--delay-max", type=float, default=3.0,
                        help="각 요청 사이 최대 딜레이 (초, 기본: 3.0)")
    args = parser.parse_args()

    all_records = []
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

    for page in tqdm(range(1, args.pages + 1), desc="Pages"):
        offset = (page - 1) * args.size
        df_meta = fetch_search_results(
            search_term=args.search_term,
            size=args.size,
            offset=offset,
            page=page
        )
        if df_meta.empty:
            tqdm.write(f"Page {page}: no more results, stopping.")
            break

        for _, row in tqdm(df_meta.iterrows(),
                            total=len(df_meta),
                            desc=f"Page {page} articles",
                            leave=False):
            url = row.get("url") or row.get("path", "")
            if not url:
                continue

            rec = {
                "search_headline": row.get("headline", ""),
                "url":             url
            }
            rec.update(scrape_article(url, user_agent))
            all_records.append(rec)
            time.sleep(random.uniform(args.delay_min, args.delay_max))

        time.sleep(random.uniform(args.delay_min, args.delay_max))

    result_df = pd.DataFrame(all_records, columns=[
        "search_headline", "url", "article_title",
        "authors", "date", "content"
    ])
    result_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result_df)} records to {args.output}")

if __name__ == "__main__":
    main()

'''실행 예시
# 기본 옵션 (search_term="AI", size=10, pages=6)
python cnn_scraper.py

# 검색어와 페이지 수, 출력 파일 지정
python cnn_scraper.py -q "machine learning" -s 20 -p 3 -o ml_articles.csv

# 딜레이 설정
python cnn_scraper.py -q "AI ethics" --delay-min 2 --delay-max 5
'''