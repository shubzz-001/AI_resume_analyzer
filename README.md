# AI Resume Analyzer & Job Recommendation System

An AI-powered application that analyzes resumes using NLP and Machine Learning to
extract skills, predict suitable job roles, recommend jobs, calculate resume scores,
and identify skill gaps.

## 🚀 Features
- Resume parsing (PDF & DOCX)
- NLP-based text preprocessing
- Skill extraction using keyword & phrase matching
- Job role prediction using ML (TF-IDF + Logistic Regression)
- Content-based job recommendation using cosine similarity
- Resume scoring system
- Skill gap analysis
- Interactive Streamlit dashboard

## 🛠 Tech Stack
- Python
- Streamlit
- spaCy, NLTK
- scikit-learn
- pandas, numpy


## ▶️ How to Run
```bash
git clone <repo-url>
cd ai_resume_analyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python ml/train_model.py
streamlit run app.py

