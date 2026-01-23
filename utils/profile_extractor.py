import re

def extract_profile_info(text) :
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    name = "Not Found"
    for line in lines:
        if len(line.split()) <= 4 and line.replace(" ","").isalpha() :
            name = line.title()
            break

    email_match = re.findall(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        text
    )
    email = email_match[0] if email_match else "Not Found"

    # ---------- PHONE ------------
    phone_match = re.findall(
        r"\+?\d[\d\s\-]{8,15}",
        text
    )
    phone = phone_match[0] if phone_match else "Not found"

    # ---------- LINKEDIN ----------
    linkedin_match = re.findall(
        r"(https?://(www\.)?linkedin\.com/[^\s]+)",
        text
    )
    linkedin = linkedin_match[0][0] if linkedin_match else "Not found"

    # ---------- GITHUB ----------
    github_match = re.findall(
        r"(https?://(www\.)?github\.com/[^\s]+)",
        text
    )
    github = github_match[0][0] if github_match else "Not found"

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github
    }