import re
import json
from datetime import datetime, timedelta
from openai import OpenAI
import os
import requests

client = OpenAI()

RELEVANCY_MODEL = "gpt-4"  # Unified for better consistency
ARTICLES_PER_QUERY = 20
NEWSAPI_DAYS_BACK = 30

# ✅ Fixed: Correct environment variable lookup
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

def analyze_script(script_text):
    prompt = f"""
You are an intelligent text analysis agent.
Read the script below and produce a JSON object with these fields:

- "main_topics": 3–5 sentences summarizing the main ideas.
- "keywords": 5–10 important keywords or named entities.
- "queries": 3 search queries a journalist could use to find more information.

Return ONLY the JSON object, no explanation or extra text.

Script:
\"\"\"
{script_text}
\"\"\"
"""
    response = client.chat.completions.create(
        model="gpt-4",
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
        model="gpt-4",
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

def batch_check_relevance(query, articles):
    prompt = f"""
Decide if each article below is strongly relevant to the topic.

Topic:
"{query}"

Articles:
"""
    for i, art in enumerate(articles):
        prompt += f"\n{i+1}. Title: {art['title']}\nDescription: {art['desc']}"

    prompt += (
        "\n\nReturn ONLY a JSON array of strings: each string must be 'Yes' or 'No'.\n"
        "Example:\n"
        '["Yes", "No", "Yes"]'
    )

    response = client.chat.completions.create(
        model=RELEVANCY_MODEL,
        messages=[
            {"role": "system", "content": "You check relevance of articles in batch."},
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

import re

import re

def get_best_sentence_indices(script_text, articles):
    """
    Uses GPT-4 to assign a unique position index to each article, starting from 1.
    Returns a list of integers.
    """
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
        model="gpt-4",
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
        # Skip keywords that are not strings
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

    return results

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
        model="gpt-4",
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
        model="gpt-4",
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
