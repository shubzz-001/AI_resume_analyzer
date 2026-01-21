from utils.ats_score import calculate_ats_score
from semantic.semantic_ats import semantic_ats_score

def hybrid_ats_score(
        resume_text,
        resume_skills,
        job_role,
        keyword_weight=0.4,
        semantic_weight=0.6
) :
    """
        Returns:
          final_score (int),
          breakdown (dict)
    """

    # --- Keyword-based ATS ---
    keyword_score, _, _, _, _ = calculate_ats_score(
        job_role,
        resume_skills,
        resume_text
    )

    # --- Semantic ATS ---
    # Extract weighted core skills from ATS data
    from utils.ats_score import json

    with open("data/ats_job_skills.json", "r") as f:
        ats_data = json.load(f)

    weighted_skills = ats_data[job_role]["core"]
    semantic_score, semantic_explanations = semantic_ats_score(
        resume_text,
        weighted_skills
    )

    # --- Combine ---
    final_score = int(
        (keyword_score * keyword_weight) +
        (semantic_score * semantic_weight)
    )

    breakdown = {
        "keyword_score": keyword_score,
        "semantic_score": semantic_score,
        "final_score": final_score,
        "semantic_matches": semantic_explanations
    }

    return final_score, breakdown