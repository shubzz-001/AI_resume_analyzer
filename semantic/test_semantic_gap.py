from semantic.semantic_skill_gap import semantic_skill_gap

resume_text = """
Built predictive models using Python to analyze data
and generate insights.
"""

required_skills = [
    "machine learning",
    "deep learning",
    "sql"
]

matched, missing = semantic_skill_gap(resume_text, required_skills)

print("Matched:", matched)
print("Missing:", missing)