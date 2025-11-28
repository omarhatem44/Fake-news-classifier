import os
from pathlib import Path

import pandas as pd

# ================== PATHS ==================
# نفترض إنك بتشغّل السكربت من جذر المشروع fake-news-classifier
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FAKE_PATH = RAW_DIR / r"D:\projects for my CV\Fake News Classifier NLP\data\Fake.csv"
TRUE_PATH = RAW_DIR / r"D:\projects for my CV\Fake News Classifier NLP\data\True.csv"

MERGED_PATH = PROCESSED_DIR / r"D:\projects for my CV\Fake News Classifier NLP\data\fake_news_merged.csv"


def load_and_merge():
    print(f"[INFO] Loading data...")
    print(f"  - Fake: {FAKE_PATH}")
    print(f"  - True: {TRUE_PATH}")

    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    print(f"[INFO] Fake shape : {fake_df.shape}")
    print(f"[INFO] True shape : {true_df.shape}")

    # نتأكد من الأعمدة
    print("\n[INFO] Fake columns:", fake_df.columns.tolist())
    print("[INFO] True columns:", true_df.columns.tolist())

    # نضيف label
    fake_df["label"] = "FAKE"
    true_df["label"] = "REAL"

    # دمج
    df = pd.concat([fake_df, true_df], ignore_index=True)
    print(f"\n[INFO] Merged shape: {df.shape} rows, {df.shape[1]} columns")

    return df


def basic_eda(df: pd.DataFrame):
    print("\n===== BASIC INFO =====")
    print(df.info())

    print("\n===== CLASS DISTRIBUTION (label) =====")
    print(df["label"].value_counts())

    print("\n===== SUBJECT DISTRIBUTION (top 10) =====")
    print(df["subject"].value_counts().head(10))

    # طول النص (text)
    df["text_length"] = df["text"].astype(str).str.split().str.len()

    print("\n===== TEXT LENGTH (words) =====")
    print(df["text_length"].describe())

    print("\n===== SAMPLE ROWS =====")
    print(df[["title", "label"]].head(5))


def save_merged(df: pd.DataFrame):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Saving merged data to: {MERGED_PATH}")
    # مش محتاجين عمود text_length في الفايل النهائي
    df_to_save = df.drop(columns=["text_length"], errors="ignore")
    df_to_save.to_csv(MERGED_PATH, index=False)
    print("[INFO] Done.")


def main():
    df = load_and_merge()
    basic_eda(df)
    save_merged(df)


if __name__ == "__main__":
    main()
