
import streamlit as st
from resume_parser.parser import extract_text
from nlp.cleaner import clean_text

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf","docx"])

if uploaded_file is not None:
    resume_text = extract_text(uploaded_file)

    if resume_text and len(resume_text.strip()) > 50 :
        cleaned_text = clean_text(resume_text)

        st.subheader("🧹 Cleaned Resume Text")
        st.text_area("Clean Text", cleaned_text, height=250)
    else :
        st.error("Could not Extract Text from Resume")
