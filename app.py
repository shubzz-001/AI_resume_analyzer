import streamlit as st

# ---------- Core Pipeline ----------
from resume_parser.parser import extract_text
from nlp.cleaner import clean_text
from nlp.skill_extractor import load_skills, extract_skills
from ml.predictor import predict_job_role

# ---------- Phase 1 ----------
from recommender.job_matcher import recommend_jobs
from utils.scoring import calculate_resume_score
from utils.ats_score import calculate_ats_score
from utils.section_checker import check_resume_sections
from utils.ats_feedback import generate_ats_feedback
from utils.rewrite_engine import suggest_rewrites
from utils.improvement_tips import generate_improvement_tips

# ---------- Semantic NLP ----------
from semantic.hybrid_ats import hybrid_ats_score
from semantic.semantic_job_matcher_v2 import semantic_recommend_jobs_v2


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer ")
st.caption(
    "ATS + Semantic NLP powered resume analysis with job-specific scoring, recommendations"
)
st.divider()


# ================= UPLOAD =================
uploaded_file = st.file_uploader(
    "Upload Resume (PDF or DOCX)",
    type=["pdf", "docx"],
    help="Text-based resumes work best"
)

skills_db = load_skills()


# ================= MAIN LOGIC =================
if uploaded_file is not None:
    with st.spinner("🔍 Analyzing resume..."):
        resume_text = extract_text(uploaded_file)

    if resume_text and len(resume_text.strip()) > 50:
        st.toast("Resume analyzed successfully!", icon="✅")

        # ---------- NLP ----------
        cleaned_text = clean_text(resume_text)

        # ---------- Skills ----------
        skills = extract_skills(cleaned_text, skills_db)

        # ---------- ML Prediction ----------
        predicted_role, confidence = predict_job_role(cleaned_text)

        # ---------- Resume Quality ----------
        resume_score = calculate_resume_score(skills, cleaned_text)

        # ================= TOP DASHBOARD =================
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 ML Predicted Job Role")
            st.success(f"{predicted_role} ({confidence}%)")

        with col2:
            st.subheader("📊 Resume Quality Score")
            st.progress(resume_score / 100)
            st.metric("Score", f"{resume_score}/100")

        st.divider()

        # ================= ATS MODE =================
        st.subheader("🤖 ATS Analysis Mode")

        ats_mode = st.radio(
            "Select ATS Evaluation Mode",
            ["Keyword ATS", "Semantic ATS (AI)", "Hybrid ATS (Recommended)"],
            horizontal=True
        )

        target_role = st.selectbox(
            "Target Job Role",
            ["Data Analyst", "ML Engineer", "Backend Developer", "Software Engineer"]
        )

        # ---------- ATS LOGIC ----------
        if ats_mode == "Keyword ATS":
            ats_score, coverage, missing_core, missing_optional, _ = calculate_ats_score(
                target_role, skills, cleaned_text
            )
            semantic_explain = []

        elif ats_mode == "Semantic ATS (AI)":
            ats_score, details = hybrid_ats_score(
                cleaned_text, skills, target_role, keyword_weight=0.0, semantic_weight=1.0
            )
            coverage = "Semantic"
            missing_core = []
            missing_optional = []
            semantic_explain = details["semantic_matches"]

        else:  # Hybrid
            ats_score, details = hybrid_ats_score(
                cleaned_text, skills, target_role
            )
            coverage = "Hybrid"
            missing_core = []
            missing_optional = []
            semantic_explain = details["semantic_matches"]

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("📈 ATS Score")
            st.progress(ats_score / 100)
            st.metric("ATS Score", f"{ats_score}/100")

        with col4:
            st.subheader("🔍 Evaluation Type")
            st.info(str(coverage))

        st.divider()

        # ================= SEMANTIC EXPLANATIONS =================
        if semantic_explain:
            st.subheader("🧠 Semantic Match Explanations")
            for skill, sim in semantic_explain:
                st.caption(f"Matched `{skill}` ({sim}%)")

        # ================= SECTION CHECK =================
        missing_sections = check_resume_sections(cleaned_text)

        st.subheader("📄 Resume Section Completeness")
        if missing_sections:
            for section in missing_sections:
                st.warning(f"Missing section: {section.capitalize()}")
        else:
            st.success("All important resume sections are present")

        st.divider()

        # ================= ATS FEEDBACK =================
        feedback = generate_ats_feedback(
            ats_score,
            missing_core,
            missing_optional,
            missing_sections
        )

        st.subheader("🧠 ATS Feedback")
        for msg in feedback:
            st.info(msg)

        st.divider()

        # ================= IMPROVEMENT TIPS =================
        st.subheader("✍️ Resume Improvement Suggestions")

        improvement_tips = generate_improvement_tips(
            ats_score,
            missing_core,
            missing_optional,
            missing_sections
        )

        for tip in improvement_tips:
            st.info(tip)

        st.divider()

        # ================= BULLET REWRITES =================
        st.subheader("🔁 ATS-Friendly Bullet Point Examples")
        for bullet in suggest_rewrites(target_role):
            st.markdown(f"- {bullet}")

        st.divider()

        # ================= SKILLS =================
        st.subheader("🛠 Extracted Skills")
        if skills:
            st.markdown(" ".join([f"`{s}`" for s in skills]))
        else:
            st.warning("No skills detected")

        st.divider()

        # ================= JOB RECOMMENDATIONS =================
        st.subheader("💼 Job Recommendations (Semantic)")

        sem_jobs = semantic_recommend_jobs_v2(cleaned_text)

        if sem_jobs:
            st.dataframe(sem_jobs, use_container_width=True)
        else:
            st.warning("No strong semantic job matches found")

        st.divider()

        # ================= RAW TEXT =================
        with st.expander("📄 View Cleaned Resume Text"):
            st.text_area("Cleaned Resume Text", cleaned_text, height=250)

    else:
        st.error("❌ Could not extract text from resume. Upload a text-based PDF or DOCX.")
