import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ================== PATHS ==================
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "fake_news_clean.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_PATH = MODELS_DIR / "fake_news_logreg.pkl"
CM_FIG_PATH = RESULTS_DIR / "confusion_matrix_fake_news.png"
METRICS_PATH = RESULTS_DIR / "metrics_report.txt"


def load_data():
    print(f"[INFO] Loading cleaned data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Shape: {df.shape}")
    print(df.head(3))
    return df


def build_pipeline():
    """
    TF-IDF + Logistic Regression pipeline
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,      # قللناها شوية لتقليل الحجم
            ngram_range=(1, 2),      # unigrams + bigrams
            min_df=2,
            max_df=0.9
        )),
        ("logreg", LogisticRegression(
            max_iter=200,
            n_jobs=1,               # <= مهم: نوقف الـ parallel
            class_weight="balanced",
            solver="liblinear"      # solver خفيف ومناسب لـ text
        ))
    ])
    return pipeline



def plot_confusion_matrix(cm, classes, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix - Fake News Classifier'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix figure to: {save_path}")


def main():
    # 1) load data
    df = load_data()
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    # 2) train/test split
    print("[INFO] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 3) build pipeline
    pipeline = build_pipeline()

    # 4) train
    print("[INFO] Training model...")
    pipeline.fit(X_train, y_train)
    print("[INFO] Training finished.")

    # 5) evaluate
    print("[INFO] Evaluating on test set...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n===== TEST METRICS =====")
    print(f"Accuracy: {acc:.4f}")

    cls_report = classification_report(y_test, y_pred, digits=4)
    print("\nClassification report:")
    print(cls_report)

    cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
    print("\nConfusion matrix (rows=True, cols=Pred):")
    print(cm)

    # 6) save metrics to file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(cls_report)
        f.write("\nConfusion matrix (labels = [FAKE, REAL]):\n")
        f.write(str(cm))
    print(f"[INFO] Saved metrics report to: {METRICS_PATH}")

    # 7) plot confusion matrix
    plot_confusion_matrix(cm, classes=["FAKE", "REAL"], save_path=CM_FIG_PATH)

    # 8) save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[INFO] Saved model to: {MODEL_PATH}")

    print("\n[INFO] All done.")


if __name__ == "__main__":
    main()
