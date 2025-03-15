from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load pre-trained sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

@app.route("/", methods=["GET"])
def home():
    return "Sentiment Analysis API is Running!"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        if "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"]
        result = sentiment_model(text)[0]  # Get first result

        return jsonify({"sentiment": result["label"], "score": result["score"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
