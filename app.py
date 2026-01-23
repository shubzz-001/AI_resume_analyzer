import streamlit as st
import re

from resume_parser.parser import extract_text
from nlp.cleaner import clean_text
from nlp.skill_extractor import load_skills, extract_skills
from ml.predictor import predict_job_role

from utils.scoring import calculate_resume_score
from utils.ats_score import calculate_ats_score
from utils.section_checker import check_resume_sections
from utils.ats_feedback import generate_ats_feedback
from utils.rewrite_engine import suggest_rewrites
from utils.improvement_tips import generate_improvement_tips

from semantic.hybrid_ats import hybrid_ats_score
from semantic.semantic_job_matcher_v2 import semantic_recommend_jobs_v2


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer & Resume Coach")
st.caption("Profile-based resume analysis with ATS and Semantic AI")
st.divider()


# ================= HELPERS =================
def extract_profile_info(text):
    email = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phone = re.findall(r"\+?\d[\d\s\-]{8,15}", text)

    return {
        "email": email[0] if email else "Not found",
        "phone": phone[0] if phone else "Not found"
    }


def generate_resume_summary(skills, predicted_role, ats_score):
    skill_part = ", ".join(skills[:6]) if skills else "relevant technical skills"

    return (
        f"The candidate appears to be well-aligned with the role of {predicted_role}. "
        f"The resume highlights experience with {skill_part}. "
        f"Based on ATS analysis, the resume achieved a score of {ats_score}%, "
        f"indicating {'strong' if ats_score >= 70 else 'moderate'} alignment with hiring standards."
    )


def generate_structured_analysis(skills, missing_sections, ats_score):
    analysis = []

    if skills:
        analysis.append(
            f"The resume demonstrates technical competency in areas such as "
            f"{', '.join(skills[:5])}."
        )

    if missing_sections:
        analysis.append(
            f"However, important sections like {', '.join(missing_sections)} are missing, "
            "which may impact ATS performance."
        )
    else:
        analysis.append(
            "The resume contains all essential sections including experience, projects, and education."
        )

    analysis.append(
        f"Overall ATS score is {ats_score}%, suggesting that the resume is "
        f"{'well optimized' if ats_score >= 70 else 'partially optimized'}."
    )

    return analysis


# ================= UPLOAD =================
uploaded_file = st.file_uploader(
    "Upload Resume (PDF or DOCX)",
    type=["pdf", "docx"]
)

skills_db = load_skills()


if uploaded_file:
    with st.spinner("Analyzing resume..."):
        resume_text = extract_text(uploaded_file)

    if resume_text and len(resume_text.strip()) > 50:
        cleaned_text = clean_text(resume_text)
        skills = extract_skills(cleaned_text, skills_db)

        predicted_role, confidence = predict_job_role(cleaned_text)
        resume_score = calculate_resume_score(skills, cleaned_text)

        profile = extract_profile_info(cleaned_text)

        # ================= PROFILE =================
        st.subheader("👤 Candidate Profile")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Role", predicted_role)
        c2.metric("Email", profile["email"])
        c3.metric("Phone", profile["phone"])

        st.info(generate_resume_summary(skills, predicted_role, resume_score))
        st.divider()

        # ================= ROLE SELECTION =================
        st.subheader("🎯 Target Job Role")
        target_role = st.selectbox(
            "You may change the target role if applying for a different position",
            ["Data Analyst", "ML Engineer", "Backend Developer", "Software Engineer"],
            index=["Data Analyst", "ML Engineer", "Backend Developer", "Software Engineer"].index(predicted_role)
            if predicted_role in ["Data Analyst", "ML Engineer", "Backend Developer", "Software Engineer"]
            else 0
        )

        ats_score, details = hybrid_ats_score(cleaned_text, skills, target_role)

        # ================= SNAPSHOT =================
        s1, s2, s3 = st.columns(3)
        s1.metric("ATS Score", f"{ats_score}/100")
        s2.metric("Resume Score", f"{resume_score}/100")
        s3.metric("Confidence", f"{confidence}%")

        st.divider()

        # ================= TABS =================
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Analysis", "🤖 ATS Feedback", "💼 Jobs", "✍️ Resume Coach"]
        )

        # ---- Analysis ----
        with tab1:
            missing_sections = check_resume_sections(cleaned_text)
            analysis_points = generate_structured_analysis(skills, missing_sections, ats_score)

            for point in analysis_points:
                st.write("• " + point)

        # ---- ATS Feedback ----
        with tab2:
            feedback = generate_ats_feedback(
                ats_score, [], [], missing_sections
            )
            for f in feedback:
                st.info(f)

        # ---- Jobs ----
        with tab3:
            jobs = semantic_recommend_jobs_v2(cleaned_text)
            st.dataframe(jobs, use_container_width=True)

        # ---- Resume Coach ----
        with tab4:
            for tip in generate_improvement_tips(
                ats_score, [], [], missing_sections
            ):
                st.info(tip)

            st.subheader("Example Bullet Points")
            for bullet in suggest_rewrites(target_role):
                st.markdown(f"- {bullet}")

    else:
        st.error("Could not extract text from resume.")
