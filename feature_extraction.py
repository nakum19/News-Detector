import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # Using joblib for saving models

# Load the preprocessed dataset
df = pd.read_csv("cleaned_news.csv")

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_df=0.7)

# Transform the cleaned text into TF-IDF features
X = vectorizer.fit_transform(df["cleaned_text"])

# Add a print statement to confirm that feature extraction is happening
print("✅ TF-IDF features extracted successfully!")

# Save the vectorizer for future use
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("✅ TF-IDF vectorizer saved successfully!")

# Save the features and labels
joblib.dump(X, "features.pkl")  # Save features
joblib.dump(df["label"], "labels.pkl")  # Save labels

# Add a final confirmation message
print("✅ Features and labels saved successfully!")