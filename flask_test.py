
# # print("Before importing libraries...")
# import pandas as pd
# # print("Pandas imported successfully.")

# import nltk
# # print("NLTK imported successfully.")

# from flask import Flask
# # print("Flask imported successfully.")

# print("Test libraries loaded successfully!")
# app = Flask(__name__)
# # print("Flask app initialized.")

# if __name__ == "__main__":
#     app.run(debug=True)

# import pandas as pd
# import nltk
# from flask import Flask

# print("Test libraries loaded successfully!")

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return "Hello, Flask is running!"

# if __name__ == "__main__":
#     print("Starting Flask...")  # Added debug print
#     app.run(debug=True, port=5001)


from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']
    # Placeholder for model prediction
    prediction = "Fake" if "fake" in news_text.lower() else "Real"
    return f"Prediction: {prediction}"

if __name__ == '__main__':
    app.run(debug=True, port=5001)