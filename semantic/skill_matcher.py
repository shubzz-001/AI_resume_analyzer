from semantic.embeddings import embed, cosine_sim

def semantic_match_skills(resume_text, required_skills, threshold=0.55) :
    resume_vec = embed([resume_text])[0]
    results = []

    skill_vecs = embed(required_skills)
    for skill, vec in zip(resume_vec, skill_vecs) :
        score = cosine_sim(resume_vec, vec)
        if score >= threshold :
            results.append(skill, round(score*100, 1))

    return sorted(results, key=lambda x: x[1], reverse=True)