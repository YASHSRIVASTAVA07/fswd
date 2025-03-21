from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

def get_model():
    """Load model only when needed (prevents memory overload)"""
    return pipeline("sentiment-analysis")

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
        model = get_model()  # Load model only when needed
        result = model(text)[0]

        return jsonify({"sentiment": result["label"], "score": result["score"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
