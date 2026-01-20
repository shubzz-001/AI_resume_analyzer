
def generate_ats_feedback(
        ats_score,
        missing_core,
        missing_optional,
        missing_sections
) :
    feedback = []

    if ats_score < 50 :
        feedback.append("Resume has low ATS compatibility. Consider adding core skills.")
    elif ats_score < 75 :
        feedback.append("Resume is moderately ATS-friendly but needs improvement.")
    else :
        feedback.append("Resume is highly ATS compatible.")

    if missing_core :
        feedback.append(
            f"Add high-priority skills : {','.join(missing_core)}"
        )

    if missing_optional :
        feedback.append(
            f"Consider adding optional skills : {','.join(missing_optional)}"
        )

    if missing_sections :
        feedback.append(
            f"Missing Important sections : {','.join(missing_sections)}"
        )

    return feedback
