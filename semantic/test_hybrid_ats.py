from semantic.hybrid_ats import hybrid_ats_score

resume_text = """
Built predictive models using Python for data analysis.
"""

resume_skills = ["python", "data analysis"]

job_role = "ML Engineer"

score, details = hybrid_ats_score(
    resume_text,
    resume_skills,
    job_role
)

print("Final ATS Score:", score)
print("Breakdown:", details)