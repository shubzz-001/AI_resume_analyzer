from typing import Dict
import logging

from nlp.text_analyzer import get_content_quality_score, TextAnalyzer

logger = logging.getLogger(__name__)


def analyze_content_quality(resume_text: str) -> Dict[str, any]:
    """
    Analyze resume content quality.
    """
    # Get comprehensive quality score
    quality_data = get_content_quality_score(resume_text)

    # Get suggestions
    analyzer = TextAnalyzer()
    suggestions = analyzer.get_suggestions(resume_text)

    return {
        'overall_score': quality_data['overall_score'],
        'action_verbs': quality_data['action_verbs'],
        'quantification_score': quality_data['quantification_score'],
        'readability_score': quality_data['readability_score'],
        'weak_verbs_count': quality_data['weak_verbs_count'],
        'passive_voice_count': quality_data['passive_voice_count'],
        'first_person_count': quality_data['first_person_count'],
        'bullet_points': quality_data['bullet_points'],
        'quality_level': get_quality_level(quality_data['overall_score']),
        'suggestions': suggestions,
        'strengths': identify_strengths(quality_data),
        'weaknesses': identify_weaknesses(quality_data)
    }


def get_quality_level(score: float) -> str:
    """Get quality level from score."""
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Fair"
    else:
        return "Needs Improvement"


def identify_strengths(quality_data: Dict) -> list:
    """Identify content strengths."""
    strengths = []

    if quality_data['action_verbs']['total'] >= 10:
        strengths.append("Strong use of action verbs")

    if quality_data['quantification_score'] >= 40:
        strengths.append("Good quantification of achievements")

    if quality_data['readability_score'] >= 70:
        strengths.append("Clear and readable content")

    if quality_data['bullet_points'].get('count', 0) > 0:
        if quality_data['bullet_points']['action_verb_percentage'] >= 70:
            strengths.append("Effective bullet point structure")

    return strengths


def identify_weaknesses(quality_data: Dict) -> List[str]:
    """Identify content weaknesses."""
    weaknesses = []

    if quality_data['weak_verbs_count'] > 5:
        weaknesses.append("Too many weak verbs")

    if quality_data['quantification_score'] < 30:
        weaknesses.append("Insufficient quantification")

    if quality_data['passive_voice_count'] > 5:
        weaknesses.append("Excessive passive voice")

    if quality_data['first_person_count'] > 0:
        weaknesses.append("Contains first-person pronouns")

    return weaknesses