from semantic.embeddings import embed, cosine_sim

def semantic_ats_score(resume_skills, weighted_skills) :
    """
    weighted_skills: dict {skill: weight}
    """

    resume_vec = embed([resume_skills])[0]
    total_weight = sum(weighted_skills.values())
    gained = 0

    skill_texts = list(weighted_skills.keys())
    skill_vec = embed(skill_texts)

    explanations = []
    for skill, vec in zip(skill_texts, skill_vec):
        sim = cosine_sim(resume_vec, vec)
        if sim >= 0.55 :
            gained += weighted_skills[skill]
            explanations.append(skill, round(sim*100,1))

    score = int((gained / total_weight) * 100) if total_weight else 0
    return score, explanations
