from flask import Flask, request, jsonify
from flask_cors import CORS
from analyzer import (
    analyze_script,
    simplify_queries,
    fetch_articles,
    deduplicate_articles,
    batch_score_relevance,
    estimate_legal_use,
    final_article_pass
)

app = Flask(__name__)
CORS(app)

@app.route("/analyze_script", methods=["POST", "OPTIONS"])
def analyze_script_endpoint():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json()
    script = data.get("script", "")
    if not script:
        return jsonify({"error": "No script provided"}), 400

    analysis = analyze_script(script)
    queries = analysis["queries"]
    keywords = analysis["keywords"]
    main_topics = analysis["main_topics"]

    simplified_query_sets = simplify_queries(queries)

    all_articles = []
    for simplified_keywords in simplified_query_sets:
        for keyword_string in simplified_keywords:
            articles = fetch_articles(keyword_string)
            all_articles.extend(articles)

    deduped_articles = deduplicate_articles(all_articles)

    relevance_scores = batch_score_relevance(queries[0], keywords, deduped_articles)
    for article, score in zip(deduped_articles, relevance_scores):
        article["relevance"] = score

    sorted_articles = sorted(deduped_articles, key=lambda x: x["relevance"], reverse=True)[:20]

    legal_labels = estimate_legal_use(sorted_articles)
    for article, label in zip(sorted_articles, legal_labels):
        article["source_type"] = f"{label['label']} — {label['note']}"

    cleaned = final_article_pass(queries[0], sorted_articles)

    return jsonify({
        "keywords": keywords,
        "main_topics": main_topics,
        "queries": queries,
        "results": cleaned
    })

@app.route("/process_script", methods=["POST", "OPTIONS"])
def process_script():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    return analyze_script_endpoint()

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)

