import pdfplumber
from docx import Document
from io import BytesIO


def extract_text(file) :
    text = ""

    file_bytes = file.read()
    file_stream = BytesIO(file_bytes)


    if file.name.endswith(".pdf") :
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

    elif file.name.endswith(".docx") :
        doc = Document(file_stream)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text.strip()