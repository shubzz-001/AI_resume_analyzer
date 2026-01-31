from typing import Dict, List
import re
import logging

logger = logging.getLogger(__name__)


def analyze_format(resume_text: str) -> Dict[str, any]:
    """
    Analyze resume format and structure.
    """
    # Check sections
    sections_present = check_sections(resume_text)

    # Analyze length
    length_analysis = analyze_length(resume_text)

    # Check formatting
    formatting_score = check_formatting(resume_text)

    return {
        'sections_present': sections_present,
        'missing_sections': get_missing_sections(sections_present),
        'length_analysis': length_analysis,
        'formatting_score': formatting_score,
        'is_well_formatted': formatting_score >= 70,
        'recommendations': generate_format_recommendations(
            sections_present,
            length_analysis,
            formatting_score
        )
    }


def check_sections(text: str) -> Dict[str, bool]:
    """Check for standard resume sections."""
    text_lower = text.lower()

    sections = {
        'contact': any(k in text_lower for k in ['email', 'phone', '@']),
        'summary': any(k in text_lower for k in ['summary', 'objective', 'profile']),
        'experience': any(k in text_lower for k in ['experience', 'work history', 'employment']),
        'education': any(k in text_lower for k in ['education', 'degree', 'university']),
        'skills': any(k in text_lower for k in ['skills', 'technologies', 'proficiencies']),
        'projects': 'project' in text_lower
    }

    return sections


def get_missing_sections(sections: Dict[str, bool]) -> List[str]:
    """Get list of missing critical sections."""
    critical = ['contact', 'experience', 'education', 'skills']
    return [s for s in critical if not sections.get(s, False)]


def analyze_length(text: str) -> Dict[str, any]:
    """Analyze resume length."""
    words = text.split()
    word_count = len(words)

    return {
        'word_count': word_count,
        'is_appropriate_length': 200 <= word_count <= 1500,
        'assessment': (
            'Too short' if word_count < 200 else
            'Too long' if word_count > 1500 else
            'Appropriate length'
        )
    }


def check_formatting(text: str) -> int:
    """Check formatting quality (0-100)."""
    score = 100

    # Penalize for excessive caps
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    if caps_ratio > 0.15:
        score -= 10

    # Penalize for very long lines (indicates poor formatting)
    lines = text.split('\n')
    long_lines = sum(1 for line in lines if len(line) > 100)
    if long_lines > len(lines) * 0.3:
        score -= 15

    return max(0, score)


def generate_format_recommendations(
        sections: Dict[str, bool],
        length: Dict,
        formatting_score: int
) -> List[str]:
    """Generate format recommendations."""
    recs = []

    # Missing sections
    missing = get_missing_sections(sections)
    if missing:
        recs.append(f"Add missing sections: {', '.join(missing)}")

    # Length issues
    if not length['is_appropriate_length']:
        recs.append(f"Adjust resume length: {length['assessment']}")

    # Formatting issues
    if formatting_score < 70:
        recs.append("Improve formatting for better readability")

    return recs