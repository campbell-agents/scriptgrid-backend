import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

from analyzer import (
    analyze_script,
    simplify_queries,
    batch_score_relevance,
    estimate_legal_use,
    get_keyword_positions,
    fetch_articles,
    get_best_sentence_indices,
    deduplicate_articles
)

app = Flask(__name__)
CORS(app)

def handle_script_analysis(script_text):
    import re

    # 1. Split script into sentences
    sentences = re.split(r'(?<=[.?!])\s+', script_text.strip())

    # 2. Extract main topics, keywords, and raw queries
    parsed = analyze_script(script_text)
    parsed["keywords"] = [k for k in parsed["keywords"] if isinstance(k, str)]

    # 3. Simplify queries
    simplified = simplify_queries(parsed["queries"])
    flattened = []
    for q in simplified:
        if isinstance(q, list):
            flattened.append(" ".join(q))
        else:
            flattened.append(str(q))
    simplified = flattened

    all_results = []

    # 4. For each simplified query: fetch, score, filter, position
    for query in simplified:
        articles = fetch_articles(query)
        if not articles:
            continue

        scores = batch_score_relevance(query, parsed["keywords"], articles)
        filtered = []
        for art, score in zip(articles, scores):
            if score >= 80:
                art["relevance_score"] = score
                art["query"] = query
                filtered.append(art)
        if not filtered:
            continue

        positions = get_best_sentence_indices(script_text, filtered)
        for art, pos in zip(filtered, positions):
            all_results.append({
                "query":      art["query"],
                "title":      art["title"],
                "url":        art["url"],
                "description":art["desc"],
                "date":       art.get("date", ""),
                "script_position": pos,
                "relevance_score": art["relevance_score"]
            })

    # 5. Deduplicate across all queries
    all_results = deduplicate_articles(all_results)

    # 6. If no results, return empty structure
    if not all_results:
        return {
            "main_topics":       parsed["main_topics"],
            "keywords":          parsed["keywords"],
            "queries":           parsed["queries"],
            "simplified_queries": simplified,
            "results":           []
        }

    # 7. Estimate legal use
    legal = estimate_legal_use(all_results)
    for art, label in zip(all_results, legal):
        art["legal_use"] = label

    # 8. Sort by script position then date
    sorted_results = sorted(
        all_results,
        key=lambda x: (x["script_position"], x["date"] or "")
    )

    # 9. Assign sequential result numbers
    for idx, art in enumerate(sorted_results, start=1):
        art["result_number"] = idx

    # 10. Return full payload
    return {
        "main_topics":       parsed["main_topics"],
        "keywords":          parsed["keywords"],
        "queries":           parsed["queries"],
        "simplified_queries": simplified,
        "results":           sorted_results
    }

@app.route("/analyze_script", methods=["POST"])
def analyze_script_endpoint():
    try:
        data = request.get_json()
        script_text = data.get("script_text", "").strip()
        if not script_text:
            return jsonify({"error": "No script_text provided"}), 400

        result = handle_script_analysis(script_text)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Optional alias for backward compatibility
@app.route("/process_script", methods=["POST"])
def process_script():
    return analyze_script_endpoint()

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)

