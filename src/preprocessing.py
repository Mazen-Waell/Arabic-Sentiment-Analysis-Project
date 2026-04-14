import re
import os
import pandas as pd

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
# Remove non-Arabic characters (keep only Arabic letters and spaces)
    text=re.sub(r"[^\u0600-\u06FF\s]", "", text)
 # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(df):
    df=df.copy()
    df["Cleaned_Text"] = df["text"].apply(clean_text)
    df=df[df["Cleaned_Text"].str.len() > 10]
    df["label"] = df["label"].replace("Mixed", "Neutral")
    return df


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("../data/raw/ar_reviews_100k.tsv", sep="\t")
    # Preprocess the data
    cleaned_df = preprocess_data(df)
    # Save the cleaned data
    os.makedirs("../data/processed", exist_ok=True)
    cleaned_df.to_csv("../data/processed/clean.csv", index=False, encoding="utf-8-sig")





