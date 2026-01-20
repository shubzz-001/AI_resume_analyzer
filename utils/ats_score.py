import json

def calculate_ats_score(
    job_role,
    resume_skills,
    cleaned_text,
    ats_file="data/ats_job_skills.json"
):
    with open(ats_file, "r") as f:
        ats_data = json.load(f)

    role_data = ats_data.get(job_role)
    if not role_data:
        return 0, 0, [], [], []

    resume_skills = set(resume_skills)

    core_skills = role_data["core"]
    optional_skills = role_data["optional"]

    total_weight = sum(core_skills.values()) + sum(optional_skills.values())
    matched_weight = 0

    missing_core = []
    missing_optional = []

    # ---- Core Skills ----
    for skill, weight in core_skills.items():
        if skill in resume_skills:
            matched_weight += weight
        else:
            missing_core.append(skill)

    # ---- Optional Skills ----
    for skill, weight in optional_skills.items():
        if skill in resume_skills:
            matched_weight += weight
        else:
            missing_optional.append(skill)

    ats_score = int((matched_weight / total_weight) * 100)

    # ---- Keyword Coverage ----
    keywords = list(core_skills.keys()) + list(optional_skills.keys())
    keyword_hits = sum(1 for k in keywords if k in cleaned_text)
    coverage = int((keyword_hits / len(keywords)) * 100)

    return ats_score, coverage, missing_core, missing_optional, keywords
