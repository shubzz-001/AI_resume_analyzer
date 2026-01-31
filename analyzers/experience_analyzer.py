from typing import Dict, List
import re
import logging

from nlp.entity_extractor import extract_years_of_experience, extract_companies

logger = logging.getLogger(__name__)


def analyze_experience(resume_text: str) -> Dict[str, any]:
    """
    Analyze work experience from resume.
    """
    # Extract years of experience
    years_exp = extract_years_of_experience(resume_text)

    # Extract companies
    companies = extract_companies(resume_text)

    # Estimate experience level
    experience_level = get_experience_level(years_exp)

    # Detect job titles
    job_titles = extract_job_titles(resume_text)

    return {
        'years_of_experience': years_exp,
        'experience_level': experience_level,
        'companies': companies,
        'company_count': len(companies),
        'job_titles': job_titles,
        'title_count': len(job_titles),
        'assessment': generate_experience_assessment(years_exp, len(companies))
    }


def get_experience_level(years: int = None) -> str:
    """Determine experience level."""
    if years is None:
        return "Unknown"
    elif years >= 10:
        return "Senior"
    elif years >= 5:
        return "Mid-level"
    elif years >= 2:
        return "Junior"
    else:
        return "Entry-level"


def extract_job_titles(text: str) -> List[str]:
    """Extract job titles from resume."""
    common_titles = [
        'engineer', 'developer', 'analyst', 'manager', 'lead',
        'senior', 'junior', 'intern', 'consultant', 'specialist',
        'architect', 'designer', 'scientist', 'researcher'
    ]

    found_titles = []
    text_lower = text.lower()

    for title in common_titles:
        if title in text_lower:
            found_titles.append(title)

    return list(set(found_titles))


def generate_experience_assessment(years: int = None, company_count: int = 0) -> str:
    """Generate experience assessment."""
    if years is None:
        return "Experience duration unclear from resume"

    assessment = f"{years}+ years of experience"

    if company_count > 3:
        assessment += " with diverse company experience"
    elif company_count > 1:
        assessment += " across multiple organizations"

    return assessment