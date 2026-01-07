import numpy as np
from flask import Flask, request, render_template
import joblib
import re
import string
import os

flask_app = Flask(__name__)

# --- Load Models and Vectorizer ---
# These filenames match your uploaded folder structure
tfidf = joblib.load("tfidf_vectorizer.pkl")
clf_model = joblib.load("clf_model.pkl")
reg_model = joblib.load("reg_model.pkl")

def clean_text(text):
    """Matches the 'clean_logic' from your training script"""
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Remove extra whitespace
    text = " ".join(text.split())
    return text

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    # 1. Get data from the three fields in your HTML
    # Note: These names must match the 'name' attribute in your index.html
    prob_desc = request.form.get("Problem description", "")
    input_desc = request.form.get("Input description", "")
    output_desc = request.form.get("Output description", "")

    # 2. Combine and clean text exactly like your training preprocessing
    combined_text = f"{prob_desc} {input_desc} {output_desc}"
    cleaned_input = clean_text(combined_text)

    # 3. Transform the cleaned text using the loaded TF-IDF
    text_features = tfidf.transform([cleaned_input])

    # 4. Generate Predictions
    predicted_class = clf_model.predict(text_features)[0]
    predicted_score = reg_model.predict(text_features)[0]

    # 5. Send results back to HTML
    # These variables (c_preds, s_preds) match the {{ }} tags in your HTML
    return render_template(
        "index.html",
        c_preds=f"Difficulty Level: {predicted_class}",
        s_preds=f"Calculated Score: {predicted_score:.2f}"
    )

if __name__ == "__main__":
    flask_app.run(debug=True)