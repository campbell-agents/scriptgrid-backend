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
- "queries": 4–8 very specific search queries a journalist might use to investigate this **exact case**, not just the general topic.

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
    text = response.choices[0].message.content.strip()
    return json.loads(text)

def simplify_queries(queries):
    prompt = (
        "You are a query simplification assistant.\n\n"
        "For each question below, extract only the 2 or 3 most important keyword phrases.\n"
        "Return ONLY strictly valid JSON with this format:\n"
        '{ "results": [["keyword1 keyword2"], ["keyword3 keyword4"]]}'
        "\nQuestions:\n"
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

Be conservative with high scores.

Topic:
"{query}"

Key Points:
{key_points}

Articles:
"""
    for i, art in enumerate(articles):
        prompt += f"\n{i+1}. Title: {art['title']}\nDescription: {art['desc']}"

    prompt += "\n\nReturn ONLY a JSON array of scores."
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
    prompt = "You are an AI that estimates usage rights.\nReturn ONLY a JSON array:\n"
    for i, art in enumerate(articles):
        prompt += f"{i+1}. Title: {art['title']}\nURL: {art['url']}\n"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Estimate likely legal use."},
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
        data = requests.get("https://serpapi.com/search.json", params=params).json()
        for section in ["organic_results", "news_results", "top_stories"]:
            if section in data:
                for res in data[section]:
                    results.append({
                        "title": res.get("title", ""),
                        "desc": res.get("snippet") or res.get("description") or "",
                        "url": res.get("link") or res.get("url") or "",
                        "date": res.get("date") or res.get("published") or ""
                    })
    except Exception as e:
        print("Error fetching SerpAPI:", e)

    if not results and NEWSAPI_KEY:
        try:
            date_from = (datetime.now() - timedelta(days=NEWSAPI_DAYS_BACK)).strftime("%Y-%m-%d")
            news_data = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "language": "en",
                    "from": date_from,
                    "pageSize": ARTICLES_PER_QUERY,
                    "sortBy": "relevancy",
                    "apiKey": NEWSAPI_KEY
                }
            ).json()
            for article in news_data.get("articles", []):
                results.append({
                    "title": article.get("title", ""),
                    "desc": article.get("description", ""),
                    "url": article.get("url", ""),
                    "date": article.get("publishedAt", "")
                })
        except Exception as e:
            print("Error fetching NewsAPI:", e)

    return deduplicate_articles(results)  # ✅ Early dedup here

def get_best_sentence_indices(sentences, articles):
    prompt = f"""
Script Sentences:
{sentences}

Articles:
"""
    for i, art in enumerate(articles):
        prompt += f"{i+1}. {art['title']} - {art['desc']}\n"
    prompt += "Return a JSON array of unique index numbers for each article."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Assign sentence indices."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())

def deduplicate_articles(articles):
    seen = set()
    unique = []
    for article in articles:
        url = normalize_url(article.get("url", ""))
        title = normalize_title(article.get("title", ""))
        desc = normalize_title(article.get("desc", ""))

        if url and title:
            key = ("url+title", url, title)
        elif title and desc:
            key = ("title+desc", title, desc)
        else:
            key = ("fallback", title or desc)

        if key not in seen:
            seen.add(key)
            unique.append(article)
    return unique

def final_article_pass(query, articles):
    prompt = f"""
Query: "{query}"
Remove duplicates. Reorder by relevance.

Articles:
"""
    for a in articles:
        prompt += f"- {a['title']} — {a['desc']} — {a['url']}\n"

    prompt += "Return strict JSON list of articles (title, desc, url)."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You clean and reorder articles."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    cleaned = json.loads(response.choices[0].message.content.strip())
    lookup = {normalize_url(a["url"]): a for a in articles}
    return [lookup.get(normalize_url(x["url"])) for x in cleaned if normalize_url(x["url"]) in lookup]

def normalize_url(url):
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

def normalize_title(title):
    return re.sub(r'\W+', '', title.lower().strip())

