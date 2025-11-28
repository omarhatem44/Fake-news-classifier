import re
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# ================== PATHS ==================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "fake_news_logreg.pkl"


# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model


# Text cleaning function (same style as training)
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ================== STREAMLIT APP ==================
def main():
    st.set_page_config(
        page_title="Fake News Classifier",
        page_icon="📰",
        layout="centered"
    )

    st.title("📰 Fake News Classifier")
    st.write(
        """
        Type or paste any news article below, and the model will predict  
        whether it is **FAKE** or **REAL**.
        """
    )

    model = load_model()

    st.subheader("✏️ Enter News Text:")
    user_text = st.text_area(
        "Paste the news article here (title + body recommended):",
        height=200,
        placeholder="Example: President announces new policy changes regarding healthcare reform..."
    )

    if st.button("🔍 Predict"):
        if not user_text.strip():
            st.warning("Please enter text before predicting.")
        else:
            cleaned = clean_text(user_text)

            if not cleaned:
                st.error("Text became empty after cleaning. Please enter clearer English text.")
                return

            # Prediction
            pred = model.predict([cleaned])[0]
            proba = model.predict_proba([cleaned])[0]
            classes = model.classes_  # ["FAKE", "REAL"]

            # Show result
            st.subheader("📌 Prediction Result")
            if pred == "FAKE":
                st.error(f"🚨 The model predicts: **{pred}**")
            else:
                st.success(f"✅ The model predicts: **{pred}**")

            # Probabilities
            st.subheader("📊 Prediction Probabilities")
            prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}
            st.write(prob_dict)

            st.caption("⚠️ This model is for educational purposes only and not intended for real-world decision-making.")


if __name__ == "__main__":
    main()
