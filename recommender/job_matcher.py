import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

with open("ml/job_role_model.pkl", "rb") as f:
    model = pickle.load(f)

def recommend_jobs(cleaned_resume_text, top_n=3) :
    df = pd.read_csv("data/job_descriptions.csv")

    job_texts = df["job_description"].tolist()
    resume_vec = model.named_steps["tfidf"].transform([cleaned_resume_text])
    job_vecs = model.named_steps["tfidf"].transform(job_texts)

    similarities = cosine_similarity(resume_vec, job_vecs)[0]

    df["match_score"] = similarities * 100
    return df.sort_values(by="match_score", ascending=False).head(top_n)
