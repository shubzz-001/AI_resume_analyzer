
import streamlit as st
from resume_parser.parser import extract_text
from nlp.cleaner import clean_text
from nlp.skill_extractor import load_skills, extract_skills
from ml.predictor import predict_job_role
from recommnder.job_matcher import recommend_jobs
from utils.scoring import calculate_resume_score

# Page config
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer")

# Upload resume
uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

# Load skills database
skills_db = load_skills()

if uploaded_file is not None:
    # Step 1: Extract text
    resume_text = extract_text(uploaded_file)

    if resume_text and len(resume_text.strip()) > 50:
        st.success("Resume text extracted successfully!")

        # Step 2: Clean text
        cleaned_text = clean_text(resume_text)

        st.subheader("🧹 Cleaned Resume Text")
        st.text_area("Cleaned Resume Text", cleaned_text, height=250)

        # Step 3: Extract skills
        skills = extract_skills(cleaned_text, skills_db)

        st.subheader("🛠 Extracted Skills")
        if skills:
            st.write(", ".join(skills))
        else:
            st.warning("No skills found.")

        # Step 4: Predict job role
        job_role, confidence = predict_job_role(cleaned_text)

        st.subheader("🎯 Predicted Job Role")
        st.success(f"{job_role} ({confidence}%)")

        # Step 5: Job Recommendations

        st.subheader("💼 Job Recommendation")

        recommendations = recommend_jobs(cleaned_text)

        st.dataframe(
            recommendations[["job_title", "match_score"]]
            .rename(columns={"job_title": "Job Role", "match_score": "Match %"})
        )

        # Step 6: Resume Scores
        resume_score = calculate_resume_score(skills, cleaned_text)
        st.subheader("📊 Resume Score")
        st.progress(resume_score/100)
        st.success(f"Resume Score: {resume_score}/100")

    else:
        st.error("❌ Could not extract text from resume. Please upload a text-based PDF or DOCX.")
