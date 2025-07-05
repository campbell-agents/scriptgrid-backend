# api_server.py
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from analyzer.analyzer import (
    analyze_script,
    simplify_queries,
    batch_score_relevance,
    estimate_legal_use,
    get_keyword_positions,
    fetch_articles,
    get_best_sentence_indices  # ✅ make sure you import the plural one
)

app = Flask(__name__)
CORS(app)

def handle_script_analysis(script_text):
    import re

    # Analyze script
    parsed = analyze_script(script_text)

    # Ensure keywords are all strings
    parsed["keywords"] = [k for k in parsed["keywords"] if isinstance(k, str)]

    # Simplify queries
    simplified = simplify_queries(parsed["queries"])

    # Flatten any nested query lists to single strings
    flattened_queries = []
    for q in simplified:
        if isinstance(q, list):
            flattened_queries.append(" ".join(q))
        else:
            flattened_queries.append(str(q))
    simplified = flattened_queries

    # Split script into sentences for AI position matching
    sentences = re.split(r'(?<=[.?!])\s+', script_text.strip())

    all_results = []
    for query in simplified:
        # Fetch articles
        articles = fetch_articles(query)
        if not articles:
            continue

        # Score relevance
        scores = batch_score_relevance(query, parsed["keywords"], articles)

        # Filter relevant articles first
        filtered_articles = []
        for art, score in zip(articles, scores):
            if score >= 80:
                art["relevance_score"] = score
                art["query"] = query
                filtered_articles.append(art)

        if not filtered_articles:
            continue

        # Use AI to assign unique positions to each article
        positions = get_best_sentence_indices(sentences, filtered_articles)

        # Combine everything
        for art, pos in zip(filtered_articles, positions):
            all_results.append({
                "query": art["query"],
                "title": art["title"],
                "url": art["url"],
                "description": art["desc"],
                "date": art.get("date", ""),
                "script_position": pos,
                "relevance_score": art["relevance_score"]
            })

    if not all_results:
        return {
            "main_topics": parsed["main_topics"],
            "keywords": parsed["keywords"],
            "queries": parsed["queries"],
            "simplified_queries": simplified,
            "results": []
        }

    # Estimate legal use
    legal_labels = estimate_legal_use(all_results)
    for art, label in zip(all_results, legal_labels):
        art["legal_use"] = label

    # Sort results by script position and date
    sorted_results = sorted(
        all_results,
        key=lambda x: (x["script_position"], x["date"] or "")
    )

    # ✅ Assign unique sequential numbers to each article
    for idx, art in enumerate(sorted_results, start=1):
        art["result_number"] = idx

    return {
        "main_topics": parsed["main_topics"],
        "keywords": parsed["keywords"],
        "queries": parsed["queries"],
        "simplified_queries": simplified,
        "results": sorted_results
    }

@app.route("/analyze_script", methods=["POST"])
def analyze_script_endpoint():
    data = request.get_json()
    script_text = data.get("script_text", "").strip()
    if not script_text:
        return jsonify({"error": "No script_text provided"}), 400

    try:
        result = handle_script_analysis(script_text)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Optional: Keep your old endpoint if you want
@app.route("/process_script", methods=["POST"])
def process_script():
    return analyze_script_endpoint()

if __name__ == "__main__":
    app.run(debug=True, port=5050)