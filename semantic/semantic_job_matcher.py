import pandas as pd
from semantic.embeddings import embed, cosine_sim

def semantic_recommend_jobs(resume_text, top_n=3) :
    df = pd.read_csv("data/job_descriptions.csv")

    resume_vec = embed([resume_text])[0]
    job_vecs = embed(df["job_description"].tolist())

    scores = []
    for v in job_vecs :
        scores.append(cosine_sim(resume_vec, v) * 100)

    df["semantic_match"] = [round(s, 1) for s in scores]
    return df.sort_values("semantic_match", ascending=False).head(top_n)