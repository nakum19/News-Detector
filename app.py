print("Hello, Fake News Detector!")

import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
model = joblib.load('final_fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Assuming you saved your vectorizer

# Title of the app
st.title("Fake News Detector")

# Input from the user
news_input = st.text_area("Enter the news article text:")

def predict_fake_news(news_text):
    # Transform the input text using the vectorizer
    news_tfidf = vectorizer.transform([news_text])
    
    # Predict using the trained model
    prediction = model.predict(news_tfidf)
    
    if prediction == 1:
        return "The news is real."
    else:
        return "The news is fake."

# When the user presses the button, show the result
if st.button('Predict'):
    if news_input:
        result = predict_fake_news(news_input)
        st.write(result)
    else:
        st.write("Please enter some text.")