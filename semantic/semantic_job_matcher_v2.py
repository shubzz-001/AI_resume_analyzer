import pandas as pd
from semantic.embeddings import embed, cosine_sim

def semantic_recommend_jobs_v2(
        resume_text,
        top_n=3,
        threshold=0.5,
) :
    """
        Returns top-N jobs with:
        - semantic_match %
        - explanation (why matched)
    """

    df = pd.read_csv("data/job_descriptions.csv")

    resume_vec = embed([resume_text])[0]
    job_texts = df["job_description"].tolist()
    job_vecs = embed(job_texts)

    results = []

    for idx, vec in enumerate(job_vecs) :
        sim = cosine_sim(resume_vec, vec)

        if sim >= threshold :
            results.append({
                "job_title": df.iloc[idx]["job_title"],
                "semantic_match": round(sim * 100, 1),
                "explanation": generate_explanation(
                    resume_text,
                    job_texts[idx]
                )
            })

    results = sorted(
        results,
        key=lambda x: x["semantic_match"],
        reverse=True
    )

    return results

def generate_explanation(resume_text, job_texts) :
    """
        Simple explanation logic
    """

    resume_words = set(resume_text.lower().split())
    job_words = set(job_texts.lower().split())

    overlap = resume_words & job_words
    top_terms = list(overlap)[:5]

    if top_terms :
        return "Matched Keywords: "+", ".join(top_terms)

    return "Strong Semantic similarity based on overall context"