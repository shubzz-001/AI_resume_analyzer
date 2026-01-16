import pdfplumber
from docx import Document

def extract_text(file) :
    text = ""

    if file.name.endswith(".pdf") :
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

    elif file.name.endswith(".docx") :
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text.strip()