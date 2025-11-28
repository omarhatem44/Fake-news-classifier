import re
from pathlib import Path

import pandas as pd

# ================== PATHS ==================
BASE_DIR = Path(__file__).resolve().parents[1]

MERGED_PATH = BASE_DIR / "data" / "fake_news_merged.csv"
CLEAN_PATH = BASE_DIR / "data" / "fake_news_clean.csv"


def clean_text(text: str) -> str:
    """تنضيف بسيط للنص الإنجليزي."""
    if not isinstance(text, str):
        return ""

    # lower case
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # remove html tags
    text = re.sub(r"<.*?>", " ", text)

    # keep only letters (remove numbers & punctuation)
    text = re.sub(r"[^a-z\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    print(f"[INFO] Loading merged data from: {MERGED_PATH}")
    df = pd.read_csv(MERGED_PATH)
    print(f"[INFO] Shape: {df.shape}")

    # نكوّن عمود واحد للنص: title + text
    print("[INFO] Creating combined text column...")
    df["raw_text"] = (df["title"].astype(str) + " " + df["text"].astype(str))

    # نطبّق التنضيف
    print("[INFO] Cleaning text...")
    df["clean_text"] = df["raw_text"].apply(clean_text)

    # نطبع شوية أمثلة
    print("\n===== SAMPLE CLEANED TEXT =====")
    for i in range(3):
        print(f"\n[Sample {i+1}]")
        print("LABEL :", df.loc[i, "label"])
        print("ORIG  :", df.loc[i, "raw_text"][:200], "...")
        print("CLEAN :", df.loc[i, "clean_text"][:200], "...")

    # نحتفظ بس الأعمدة المهمة للموديل
    df_out = df[["clean_text", "label"]].rename(
        columns={"clean_text": "text"}
    )

    print(f"\n[INFO] Final clean shape: {df_out.shape}")

    print(f"[INFO] Saving cleaned data to: {CLEAN_PATH}")
    df_out.to_csv(CLEAN_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
