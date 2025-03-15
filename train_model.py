import pandas as pd
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Download NLTK tokenizer
nltk.download('punkt')

# Sample dataset
data = {
    "text": ["I love this product", "This is the worst thing ever", "It's okay, nothing special", 
             "Absolutely amazing!", "I hate this!", "Not bad, but could be better"],
    "label": ["positive", "negative", "neutral", "positive", "negative", "neutral"]
}

df = pd.DataFrame(data)

# Convert text labels to numbers
label_map = {"positive": 1, "negative": -1, "neutral": 0}
df["label"] = df["label"].map(label_map)

# Convert text data into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model training complete. Saved as sentiment_model.pkl and vectorizer.pkl")
