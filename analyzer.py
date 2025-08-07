import os
import re
import json
import requests
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse
from openai import OpenAI

client = OpenAI()

RELEVANCY_MODEL = "gpt-4"
ARTICLES_PER_QUERY = 8
NEWSAPI_DAYS_BACK = 30
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

def analyze_script(script_text):
    prompt = f"""
You are an intelligent text analysis agent.
Read the script below and return a JSON object with:

- "main_topics": A 3–5 sentence summary of the script’s main ideas.
- "keywords": 5–10 of the most important names, places, and concepts.
- "queries": 4-8 very specific search queries a journalist might use to investigate this **exact case**, not just the general topic.

Avoid generic phrasing like "unidentified victims" or "forensic techniques". Be concrete.

Return ONLY the JSON object. No extra text.

Script:
\"\"\"
{script_text}
\"\"\"
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You help extract structured information from text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())

def simplify_queries(queries):
    prompt = (
        "You are a query simplification assistant.\n\n"
        "For each question below, extract only the 2 or 3 most important keyword phrases.\n"
        "Return ONLY strictly valid JSON with this format:\n\n"
        "{\n"
        '  "results": [\n'
        '    ["keyword1 keyword2", "keyword1 keyword3"],\n'
        '    ["keyword1 keyword2 keyword3"],\n'
        '    ...\n'
        "  ]\n"
        "}\n\n"
        "Questions:\n"
    )
    for q in queries:
        prompt += f"- {q}\n"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You extract concise keyword phrases and respond with strict JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())["results"]

def batch_score_relevance(query, keywords, articles):
    key_points = "\n".join(f"- {k}" for k in keywords)
    prompt = f"""
You are an AI relevance scorer.

For each article below, assign a numeric relevance score between 0 and 100:

- 100: Directly about the topic and discusses key points in detail.
- 50: Related to the topic but does not cover any key points substantially.
- 0: Unrelated to the topic.

Be conservative with high scores: only assign 100 if the article clearly discusses the key points.

Topic:
"{query}"

Key Points:
{key_points}

Articles:
"""
    for i, art in enumerate(articles):
        prompt += f"\n{i+1}. Title: {art['title']}\nDescription: {art['desc']}"

    prompt += "\n\nReturn ONLY a JSON array of scores.\nExample:\n[100, 50, 0]"

    response = client.chat.completions.create(
        model=RELEVANCY_MODEL,
        messages=[
            {"role": "system", "content": "You score article relevance strictly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())

def estimate_legal_use(articles):
    prompt = """
You are an AI that estimates the likely copyright or usage status of online articles for content creators.
For each article, return a JSON object with:
- "label": one of:
  - "Public Domain"
  - "Fair Use Likely"
  - "License Likely Required"
- "note": one short sentence explaining why.

Return ONLY a JSON array of objects in this format:
[
  {"label": "...", "note": "..."},
  {"label": "...", "note": "..."}
]

Articles:
"""
    for i, art in enumerate(articles):
        prompt += f"\n{i+1}. Title: {art['title']}\nURL: {art['url']}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You estimate likely legal use status and explain it concisely."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())

def fetch_articles(query):
    params = {
        "engine": "google",
        "q": query,
        "hl": "en",
        "num": ARTICLES_PER_QUERY,
        "api_key": SERPAPI_KEY
    }

    results = []
    try:
        response = requests.get("https://serpapi.com/search.json", params=params)
        data = response.json()
    except Exception as e:
        print(f"SerpAPI error: {e}")
        return []

    for section in ["organic_results", "news_results", "top_stories"]:
        if section in data:
            for res in data[section]:
                results.append({
                    "title": res.get("title", ""),
                    "desc": res.get("snippet") or res.get("description") or "",
                    "url": res.get("link") or res.get("url") or "",
                    "date": res.get("date") or res.get("published") or ""
                })

    if not results and NEWSAPI_KEY:
        try:
            date_from = (datetime.now() - timedelta(days=NEWSAPI_DAYS_BACK)).strftime("%Y-%m-%d")
            newsapi_response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "language": "en",
                    "from": date_from,
                    "pageSize": ARTICLES_PER_QUERY,
                    "sortBy": "relevancy",
                    "apiKey": NEWSAPI_KEY
                }
            )
            data = newsapi_response.json()
            for article in data.get("articles", []):
                results.append({
                    "title": article.get("title", ""),
                    "desc": article.get("description") or "",
                    "url": article.get("url") or "",
                    "date": article.get("publishedAt") or ""
                })
        except Exception as e:
            print(f"NewsAPI error: {e}")

    return results

def deduplicate_articles(articles):
    seen = set()
    unique = []
    for article in articles:
        url = (article.get("url") or "").strip().lower()
        title = (article.get("title") or "").strip().lower()
        desc = (article.get("desc") or "").strip().lower()

        if url and title:
            key = ("url_title", url, title)
        elif title and desc:
            key = ("title_desc", title, desc)
        elif title:
            key = ("title_only", title)
        else:
            key = ("raw_hash", hash(json.dumps(article, sort_keys=True)))

        if key not in seen:
            seen.add(key)
            unique.append(article)

    return unique

def final_article_pass(query, articles):
    prompt = f"""
You are an AI assistant cleaning up a list of articles for a research query.

TASK:
1. Remove any duplicate or nearly duplicate articles (even if slightly reworded).
2. Reorder them so that the most relevant ones are listed first.

Query:
"{query}"

Articles:
"""
    for i, art in enumerate(articles):
        prompt += f"\n{i+1}. Title: {art['title']}\nDescription: {art['desc']}\nURL: {art['url']}"

    prompt += """
Return ONLY the cleaned and reordered list in this strict JSON format:
[
  {"title": "...", "desc": "...", "url": "..."},
  ...
]
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You remove redundant articles and order them by relevance."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    cleaned = json.loads(response.choices[0].message.content.strip())

    # Re-merge metadata
    url_map = {a["url"]: a for a in articles}
    enriched = []
    for item in cleaned:
        full = url_map.get(item["url"])
        if full:
            enriched.append(full)
    return enriched

def get_keyword_positions(script_text, keywords):
    sentences = re.split(r'[.?!]\s+', script_text)
    positions = {}
    for keyword in keywords:
        if not isinstance(keyword, str):
            continue
        for idx, sentence in enumerate(sentences):
            if keyword.lower() in sentence.lower():
                positions[keyword] = idx
                break
        else:
            positions[keyword] = 999
    return positions

def get_best_sentence_indices(sentences, articles):
    prompt = f"""
You are an AI assistant helping to align articles to a script.

**Task**:
- The script below is split into sentences.
- For each article, assign a unique index number (starting at 1) indicating which sentence best matches the article’s topic and content.
- Each article **must get a different index number**, no duplicates.
- Return ONLY a JSON array of integers.

**Script Sentences**:
{sentences}

**Articles**:
"""
    for i, art in enumerate(articles):
        prompt += f"\nArticle {i+1}:\nTitle: {art.get('title', '')}\nDescription: {art.get('desc', '')}"

    prompt += "\nReturn ONLY the JSON array of integers, no explanations."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You assign unique sentence indices to articles."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return json.loads(response.choices[0].message.content.strip())

