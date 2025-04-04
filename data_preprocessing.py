import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

# Download stopwords & tokenizer
nltk.download("stopwords")
nltk.download("punkt")

# Initialize tools
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Load the datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add a "label" column: 1 for fake news, 0 for real
fake_df["label"] = 1
true_df["label"] = 0

# Merge both of the datasets
df = pd.concat([fake_df, true_df], axis=0)

# Shuffle the dataset to mix fake and real news
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check if 'title' and 'text' columns exist
if 'title' not in df.columns or 'text' not in df.columns:
    print("Error: Columns 'title' or 'text' are missing from the dataset.")
else:
    # Function to clean text
    def preprocess_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\W', ' ', text)  # Remove special characters
        words = word_tokenize(text)  # Tokenize words
        words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords & stem words
        return " ".join(words)

    # Combine title & text and apply preprocessing to the entire dataset
    df["cleaned_text"] = df["title"].astype(str) + " " + df["text"].astype(str)  # Combine title & text
    df["cleaned_text"] = df["cleaned_text"].apply(preprocess_text)

    # Save the preprocessed dataset
    df.to_csv("preprocessed_news.csv", index=False)

    # Display sample output
    print("\nPreprocessed Data Sample:")
    print(df[["cleaned_text", "label"]].head())

    # Optionally, print dataset info
    print("\nDataset Info:")
    print(df.info())

    # Save the cleaned dataset
    df.to_csv("cleaned_news.csv", index=False)