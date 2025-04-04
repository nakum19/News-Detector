A machine learning-based project that detects whether a news article is **fake** or **real** using Natural Language Processing (NLP) techniques.


🚀 How to Run the Project

✅ Prerequisites

Make sure you have the following installed:

- Python 3.8+
- `pip`
- Virtual environment support (optional but recommended)

🔧 1. Clone the repository

```bash
git clone https://github.com/your-username/Fake_News_Detector.git
cd Fake_News_Detector
```

### 📦 2. Set up a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

📥 3. Install dependencies

```bash
pip install -r requirements.txt
```


🧪 4. Run the pipeline step-by-step

**Step 1: Preprocess the dataset**
```bash
python data_preprocessing.py
```

**Step 2: Extract TF-IDF features**
```bash
python feature_extraction.py
```

**Step 3: Train the model**
```bash
python model_training.py
```

**Step 4: Evaluate the model**
```bash
python model_evaluation.py
```

## Download Required Files

These files are too large for GitHub and need to be downloaded manually.

cleaned_news.csv : https://drive.google.com/file/d/1TMmMk4qF1E2_MyZuhm_H3jMDw_n3NC2o/view?usp=sharing

preprocessed_news.csv: https://drive.google.com/file/d/19x21FKSw7gd0Hwwfcsspr48JTAmYR9j8/view?usp=sharing

After downloading, place these files in your project directory before running the app.

💡 Project Goal

This project was created as part of a hackathon to tackle the spread of misinformation using accessible machine learning techniques. It demonstrates how NLP can be used in practical, real-world applications.


Feel free to fork or contribute!
