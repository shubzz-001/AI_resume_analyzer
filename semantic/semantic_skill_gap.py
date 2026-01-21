from semantic.embeddings import embed, cosine_sim

def semantic_skill_gap(
        resume_text,
        required_skills,
        threshold=0.55
) :
    """
        required_skills: list[str]
        returns:
          matched_skills: list[(skill, similarity%)]
          missing_skills: list[str]
    """

    resume_vec = embed([resume_text])[0]
    skill_vecs = embed(required_skills)

    matched = []
    missing = []

    for skill, vec in zip(required_skills, skill_vecs) :
        sim = cosine_sim(resume_vec, vec)
        if sim >= threshold :
            matched.append((skill, round(sim*100, 1)))
        else :
            missing.append(skill)

    return matched, missing