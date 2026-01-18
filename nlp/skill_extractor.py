
def load_skills(path="data/skills_list.txt") :
    with open(path, "r") as f:
        return [skill.strip().lower() for skill in f.readlines()]

def extract_skills(text, skills_db) :
    found = set()
    text = text.lower()

    for skill in skills_db :
        if skill in text :
            found.add(skill)

    return sorted(found)