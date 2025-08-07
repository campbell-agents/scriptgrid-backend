import re
import json
import hashlib
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse
from openai import OpenAI
import os
import requests

client = OpenAI()

RELEVANCY_MODEL = "gpt-3.5-turbo"
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
- "queries": 4-8 very specific search queries a journalist might use to investigate this **exact case**, not just the general topic. Include names, dates, and locations if possible.

Avoid generic phrasing like "unidentified victims" or "forensic techniques". Be concrete.

Return ONLY the JSON object. No extra text.

Script:
\"\"\"
{script_text}
\"\"\"
"""
    response = client.chat.completions.create(
        model=RELEVANCY_MODEL,
        messages=[
            {"role": "system", "content": "You help extract structured information from text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    text_response = response.choices[0].message.content.strip()
    try:
        return json.loads(text_response)
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        print("Raw response:", text_response)
        raise

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
        model=RELEVANCY_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract concise keyword phrases for search and always respond ONLY with valid JSON as instructed."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    text_response = response.choices[0].message.content.strip()
    if not text_response.startswith("{"):
        raise ValueError("Response did not return valid JSON:\n" + text_response)

    try:
        return json.loads(text_response)["results"]
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        print("Raw response:", text_response)
        raise

def get_best_sentence_indices(script_text, articles):
    prompt = f"""
You are an AI assistant helping to align articles to a script.

**Task**:
- The script below is split into sentences.
- For each article, assign a unique index number (starting at 1) indicating which sentence best matches the article's topic and content.
- Each article **must get a different index number**, no duplicates.
- Return ONLY a JSON array of integers. For example: [1,2,3]

**Script Sentences**:
{script_text}

**Articles**:
"""
    for i, art in enumerate(articles):
        prompt += f"\nArticle {i+1}:\nTitle: {art.get('title', '')}\nDescription: {art.get('desc', '')}"

    prompt += """
Return ONLY the JSON array of integers, no explanations.
"""

    response = client.chat.completions.create(
        model=RELEVANCY_MODEL,
        messages=[
            {"role": "system", "content": "You assign unique sentence indices to articles."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    text = response.choices[0].message.content.strip()
    return json.loads(text)

def get_keyword_positions(script_text, keywords):
    sentences = re.split(r'[.?!]\s+', script_text)
    positions = {}
    for keyword in keywords:
        if not isinstance(keyword, str):
            continue
        found = False
        for idx, sentence in enumerate(sentences):
            if keyword.lower() in sentence.lower():
                positions[keyword] = idx
                found = True
                break
        if not found:
            positions[keyword] = 999
    return positions

def fetch_articles(query):
    params = {
        "engine": "google",
        "q": query,
        "hl": "en",
        "num": ARTICLES_PER_QUERY,
        "api_key": SERPAPI_KEY
    }
    try:
        response = requests.get("https://serpapi.com/search.json", params=params)
        data = response.json()
        print("\n=== RAW SERPAPI DATA ===")
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error fetching from SerpAPI: {e}")
        return []

    results = []
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
        print("No SerpAPI results. Checking NewsAPI...")
        date_from = (datetime.now() - timedelta(days=NEWSAPI_DAYS_BACK)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        newsapi_params = {
            "q": query,
            "language": "en",
            "from": date_from,
            "pageSize": ARTICLES_PER_QUERY,
            "sortBy": "relevancy",
            "apiKey": NEWSAPI_KEY
        }
        try:
            response = requests.get(url, params=newsapi_params)
            data = response.json()
            print("\n=== RAW NEWSAPI DATA ===")
            print(json.dumps(data, indent=2))
            if "articles" in data:
                for article in data["articles"]:
                    results.append({
                        "title": article.get("title", ""),
                        "desc": article.get("description") or "",
                        "url": article.get("url") or "",
                        "date": article.get("publishedAt") or ""
                    })
        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")

    results = deduplicate_articles(results)

    if results:
        try:
            simplified_keywords = [kw for group in simplify_queries([query]) for kw in group]
            scores = batch_score_relevance(query, simplified_keywords, results)
            results = [a for a, s in zip(results, scores) if s >= 80]
        except Exception as e:
            print(f"Error scoring/filtering relevance: {e}")

    return results

def normalize_title(title):
    return re.sub(r'\W+', '', title.lower().strip())

def normalize_url(url):
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

def deduplicate_articles(articles):
    seen = set()
    unique = []

    def norm(text): return re.sub(r'\W+', '', text.lower().strip()) if text else ''

    for article in articles:
        title = norm(article.get("title", ""))
        desc = norm(article.get("desc", ""))
        url = normalize_url(article.get("url", ""))
        key_variants = [
            title + url,
            title + desc,
            hashlib.sha256((title + desc).encode()).hexdigest()
        ]
        if not any(k in seen for k in key_variants):
            unique.append(article)
            for k in key_variants:
                seen.add(k)

    return unique

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

    prompt += (
        "\n\nReturn ONLY a JSON array of scores.\n"
        "Example:\n"
        "[100, 50, 0]"
    )

    response = client.chat.completions.create(
        model=RELEVANCY_MODEL,
        messages=[
            {"role": "system", "content": "You score article relevance strictly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    text_response = response.choices[0].message.content.strip()
    try:
        return json.loads(text_response)
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        print("Raw response:", text_response)
        raise

def estimate_legal_use(articles):
    prompt = """
You are an AI that estimates the likely copyright or usage status of online articles for content creators.
For each article, return a JSON object with:
- "label": one of:
  - "Public Domain"
  - "Fair Use Likely"
  - "License Likely Required"
- "note": one short sentence explaining why.

For example, if an article is from Wikipedia, it is usually public domain or Creative Commons.
If it's a recent news article, it is usually under copyright and may require permission.
If it's an excerpt or summary, fair use might apply.

Return ONLY a JSON array of objects in this format:
[
  {"label": "...", "note": "..."},
  {"label": "...", "note": "..."}
]

Articles:
"""
    for i, art in enumerate(articles):
        prompt += f"\n{i+1}. Title: {art['title']}\nURL: {art['url']}"

    prompt += (
        "\n\nReturn ONLY the JSON array, no extra text.\n"
        "Example:\n"
        '[{"label": "Public Domain", "note": "From Wikipedia, which is freely licensed."}]'
    )

    response = client.chat.completions.create(
        model=RELEVANCY_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You estimate likely legal use status and explain it concisely."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    text_response = response.choices[0].message.content.strip()
    try:
        return json.loads(text_response)
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        print("Raw response:", text_response)
        raise

