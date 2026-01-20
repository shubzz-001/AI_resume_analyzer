def check_resume_sections(cleaned_text) :
    sections = {
        "experience": ["experience", "work experience", "employment"],
        "projects": ["project", "projects"],
        "education": ["education", "degree", "university"]
    }

    missing_sections = []

    for section, keywords in sections.items() :
        if not any(k in cleaned_text for k in keywords) :
            missing_sections.append(section)

    return missing_sections