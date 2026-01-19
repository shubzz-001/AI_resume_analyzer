
def calculate_resume_score(skills, cleaned_text) :
    score = 0

    # 1- Skill Coverage(max 50)
    skill_score = min(len(skills)*5,50)
    score += skill_score

    # 2- Resume Length Quality (max 20)
    word_count = len(cleaned_text.split())
    if word_count > 300:
        score += 20
    elif word_count > 150:
        score += 10

    # 3- Experience Indicators(max 20)
    experience_keywords = ["experience", "project", "internship", "worked", "developed"]
    exp_hits = sum(1 for k in experience_keywords if k in cleaned_text)
    score += min(exp_hits*5, 20)

    # 4- Certification / achievement keywords (max 10)
    achievement_keywords = ["achievement", "certification", "certified", "award"]
    ach_hits = sum(1 for k in achievement_keywords if k in cleaned_text)
    score += min(ach_hits*5, 10)

    return min(score, 100)