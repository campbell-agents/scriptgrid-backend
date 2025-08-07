from flask import Flask, request, jsonify
from analyzer import (
    analyze_script,
    simplify_queries,
    fetch_articles,
    batch_score_relevance,
    estimate_legal_use,
    get_best_sentence_indices,
    final_article_pass,
)

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    script_text = data.get("script", "").strip()

    if not script_text:
        return jsonify({"error": "Missing 'script' field"}), 400

    try:
        # STEP 1 — Analyze script
        analysis = analyze_script(script_text)
        queries = analysis["queries"]
        keywords = analysis["keywords"]
        main_topics = analysis["main_topics"]

        # STEP 2 — Simplify queries
        simplified_queries = simplify_queries(queries)

        all_articles = []

        # STEP 3 — Fetch and deduplicate per query
        for query_variants in simplified_queries:
            for query in query_variants:
                fetched = fetch_articles(query)
                if fetched:
                    scored = batch_score_relevance(query, keywords, fetched)
                    for i, score in enumerate(scored):
                        fetched[i]["relevance_score"] = score
                        fetched[i]["query"] = query
                    all_articles.extend(fetched)

        if not all_articles:
            return jsonify({"error": "No relevant articles found"}), 404

        # STEP 4 — Final deduplication + sorting
        all_articles = final_article_pass(" | ".join(queries), all_articles)

        # STEP 5 — Estimate usage rights
        legal_info = estimate_legal_use(all_articles)
        for article, rights in zip(all_articles, legal_info):
            article["source_type"] = rights["label"]
            article["use_note"] = rights["note"]

        # STEP 6 — Add additional metadata
        keyword_positions = get_keyword_positions(script_text, keywords)
        for article in all_articles:
            article["query"] = article.get("query", "")
            article["relevance"] = article.get("relevance_score", 0)

        # STEP 7 — Sentence mapping
        sentences = script_text.split(". ")
        best_indices = get_best_sentence_indices(sentences, all_articles)
        for article, idx in zip(all_articles, best_indices):
            article["script_sentence_index"] = idx

        return jsonify({
            "main_topics": main_topics,
            "keywords": keywords,
            "queries": queries,
            "results": all_articles
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

