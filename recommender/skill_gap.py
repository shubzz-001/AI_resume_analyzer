import json

def find_skill_gaps(predicted_role, candidate_skills) :
    with open("data/job_required_skills.json", "r") as f :
        role_skills = json.load(f)

    required = set(role_skills.get(predicted_role, []))
    present = set(candidate_skills)

    missing = required - present
    return sorted(missing)