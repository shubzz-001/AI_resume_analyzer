
def generate_improvement_tips(
        ats_score,
        missing_core,
        missing_optional,
        missing_sections
) :
    tips = []

    if ats_score < 60:
        tips.append(
            "Increase ATS score by adding missing core skills and aligning resume keywords with the job description."
        )

    if missing_core:
        tips.append(
            f"Add these high-priority skills to your resume: {', '.join(missing_core)}."
        )

    if missing_optional:
        tips.append(
            f"Optional skills like {', '.join(missing_optional)} can improve shortlisting chances."
        )

    if missing_sections:
        tips.append(
            f"Add missing resume sections: {', '.join(missing_sections)}."
        )

    tips.append(
        "Use measurable impact in bullet points (e.g., improved performance by 20%)."
    )

    return tips