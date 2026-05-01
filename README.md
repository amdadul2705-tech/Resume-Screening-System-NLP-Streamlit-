### Resume Screening System (NLP + ML).

### Overview
This project is an AI-powered Resume Screening System that ranks resumes based on a given job description using NLP techniques.

### Features
- Upload multiple resumes (PDF)
- Extract and clean text
- Compare with job description
- Rank candidates using cosine similarity

### Tech Stack
- Python
- Scikit-learn
- Streamlit
- NLP (TF-IDF)

### How It Works
1. Convert text into TF-IDF vectors
2. Compute cosine similarity
3. Rank resumes based on similarity score

### Run Locally
pip install -r requirements.txt
streamlit run app.py
