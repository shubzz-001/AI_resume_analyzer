
import streamlit as st
from resume_parser.parser import extract_text

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf","docx"])

if uploaded_file :
    resume_text = extract_text(uploaded_file)

    if resume_text :
        st.success("Resume text Extracted Successfully")
        st.text_area("Extracted Resume Text", resume_text, height=300)
    else :
        st.error("Could not Extract Text from Resume")
