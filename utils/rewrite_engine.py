from utils.impact_verbs import IMPACT_VERBS

def suggest_rewrites(job_role) :
    if job_role == "Data Analyst" :
        verbs = IMPACT_VERBS["data"]
    elif job_role == "ML Engineer" :
        verbs = IMPACT_VERBS["ml"]
    elif job_role == "Backend Engineer" :
        verbs = IMPACT_VERBS["backend"]
    else :
        verbs = IMPACT_VERBS["general"]

    templates = [
        f"{verbs[0].capitalize()} datasets using Python and SQL to deliver actionable insights.",
        f"{verbs[1].capitalize()} end-to-end solutions improving performance and reliability.",
        f"{verbs[2].capitalize()} features aligned with business requirements."
    ]

    return templates