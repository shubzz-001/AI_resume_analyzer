
import streamlit as st

from resume_parser.parser import extract_text
from nlp.cleaner import clean_text
from nlp.skill_extractor import load_skills, extract_skills
from ml.predictor import predict_job_role
from recommender.job_matcher import recommend_jobs
from recommender.skill_gap import find_skill_gaps
from utils.scoring import calculate_resume_score


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer & Job Recommender")
st.caption(
    "Upload your resume to get job predictions, recommendations, resume score, and skill gap analysis"
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

        # ---- NLP Cleaning ----
        cleaned_text = clean_text(resume_text)

        # ---- Skill Extraction ----
        skills = extract_skills(cleaned_text, skills_db)

        # ---- Job Role Prediction ----
        job_role, confidence = predict_job_role(cleaned_text)

        # ---- Resume Score ----
        resume_score = calculate_resume_score(skills, cleaned_text)

        # ---- Job Recommendations ----
        recommendations = recommend_jobs(cleaned_text)

        # ---- Skill Gaps ----
        skill_gaps = find_skill_gaps(job_role, skills)

        # ================= DASHBOARD =================

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Predicted Job Role")
            st.success(f"{job_role} ({confidence}%)")

        with col2:
            st.subheader("📊 Resume Score")
            st.progress(resume_score / 100)
            st.metric("Score", f"{resume_score}/100")

        st.divider()

        # ================= SKILLS =================
        st.subheader("🛠 Extracted Skills")
        if skills:
            st.markdown(" ".join([f"`{skill}`" for skill in skills]))
        else:
            st.warning("No skills detected.")

        st.divider()

        # ================= JOB RECOMMENDATIONS =================
        st.subheader("💼 Job Recommendations")
        st.dataframe(
            recommendations[["job_title", "match_score"]]
            .rename(columns={
                "job_title": "Job Role",
                "match_score": "Match %"
            }),
            use_container_width=True
        )

        st.divider()

        # ================= SKILL GAPS =================
        st.subheader("📉 Skill Gaps")
        if skill_gaps:
            for skill in skill_gaps:
                st.warning(f"Missing: {skill}")
        else:
            st.success("Great! No major skill gaps found 🎉")

        st.divider()

        # ================= CLEANED TEXT (OPTIONAL VIEW) =================
        with st.expander("📄 View Cleaned Resume Text"):
            st.text_area(
                "Cleaned Resume Text",
                cleaned_text,
                height=250
            )

    else:
        st.error(
            "❌ Could not extract text from resume. "
            "Please upload a text-based PDF or DOCX."
        )
