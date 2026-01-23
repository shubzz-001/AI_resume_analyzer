import streamlit as st

from resume_parser.parser import extract_text
from nlp.cleaner import clean_text
from nlp.skill_extractor import load_skills, extract_skills
from ml.predictor import predict_job_role

from utils.scoring import calculate_resume_score
from utils.section_checker import check_resume_sections
from utils.ats_feedback import generate_ats_feedback
from utils.rewrite_engine import suggest_rewrites
from utils.improvement_tips import generate_improvement_tips
from utils.profile_extractor import extract_profile_info

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

        # ================= PROFILE EXTRACTION =================
        profile = extract_profile_info(resume_text)

        st.subheader("👤 Candidate Profile")

        p1, p2, p3 = st.columns(3)
        p1.metric("Name", profile.get("name", "Not found"))
        p2.metric("Email", profile.get("email", "Not found"))
        p3.metric("Phone", profile.get("phone", "Not found"))

        l1, l2 = st.columns(2)
        with l1:
            st.write("🔗 **LinkedIn**")
            st.write(profile.get("linkedin", "Not found"))

        with l2:
            st.write("🧑‍💻 **GitHub**")
            st.write(profile.get("github", "Not found"))

        st.info(
            f"The candidate appears well-suited for the role of **{predicted_role}**. "
            f"The resume highlights skills such as {', '.join(skills[:5]) if skills else 'relevant technologies'}, "
            f"and achieved an ATS score of **{resume_score}%**, indicating "
            f"{'strong' if resume_score >= 70 else 'moderate'} alignment with industry expectations."
        )

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
        s3.metric("Prediction Confidence", f"{confidence}%")

        st.divider()

        # ================= TABS =================
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Analysis", "🤖 ATS Feedback", "💼 Jobs", "✍️ Resume Coach"]
        )

        # ---------- Analysis ----------
        with tab1:
            missing_sections = check_resume_sections(cleaned_text)
            st.subheader("Resume Analysis Summary")

            st.write(
                f"The resume demonstrates proficiency in **{', '.join(skills[:5]) if skills else 'key technical skills'}**. "
                f"It achieved an ATS score of **{ats_score}%**, indicating that the resume is "
                f"{'well optimized' if ats_score >= 70 else 'partially optimized'}."
            )

            if missing_sections:
                st.warning(
                    f"The following important sections are missing: {', '.join(missing_sections)}."
                )
            else:
                st.success("All essential resume sections are present.")

        # ---------- ATS Feedback ----------
        with tab2:
            feedback = generate_ats_feedback(
                ats_score, [], [], missing_sections
            )
            for f in feedback:
                st.info(f)

        # ---------- Jobs ----------
        with tab3:
            jobs = semantic_recommend_jobs_v2(cleaned_text)
            if jobs:
                st.dataframe(jobs, use_container_width=True)
            else:
                st.warning("No strong semantic job matches found.")

        # ---------- Resume Coach ----------
        with tab4:
            for tip in generate_improvement_tips(
                ats_score, [], [], missing_sections
            ):
                st.info(tip)

            st.subheader("Example ATS-Friendly Bullet Points")
            for bullet in suggest_rewrites(target_role):
                st.markdown(f"- {bullet}")

    else:
        st.error("❌ Could not extract text from resume.")
