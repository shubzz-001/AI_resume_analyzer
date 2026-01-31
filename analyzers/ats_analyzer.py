from typing import Dict, List, Tuple
import logging

from semantic.hybrid_ats import hybrid_ats_score, get_ats_improvement_suggestions
from config import EXCELLENT_SCORE, GOOD_SCORE, FAIR_SCORE

logger = logging.getLogger(__name__)


def analyze_ats_compatibility(
        resume_text: str,
        resume_skills: List[str],
        job_role: str
) -> Dict[str, any]:
    """
    Comprehensive ATS compatibility analysis.
    """
    try:
        # Calculate hybrid ATS score
        ats_score, breakdown = hybrid_ats_score(
            resume_text,
            resume_skills,
            job_role
        )

        # Get improvement suggestions
        suggestions = get_ats_improvement_suggestions(breakdown)

        # Determine compatibility level
        compatibility = get_compatibility_level(ats_score)

        # Get keyword details
        keyword_details = breakdown.get('keyword_details', {})

        return {
            'ats_score': ats_score,
            'compatibility_level': compatibility,
            'keyword_score': breakdown.get('keyword_score', 0),
            'semantic_score': breakdown.get('semantic_score', 0),
            'matched_core_skills': keyword_details.get('matched_core', []),
            'matched_optional_skills': keyword_details.get('matched_optional', []),
            'missing_core_skills': keyword_details.get('missing_core', []),
            'missing_optional_skills': keyword_details.get('missing_optional', []),
            'semantic_matches': breakdown.get('semantic_matches', []),
            'suggestions': suggestions,
            'job_role': job_role,
            'pass_probability': calculate_pass_probability(ats_score)
        }

    except Exception as e:
        logger.error(f"Error in ATS compatibility analysis: {str(e)}")
        return {
            'ats_score': 0,
            'compatibility_level': 'Unknown',
            'error': str(e)
        }


def get_compatibility_level(score: int) -> str:
    """
    Get compatibility level from score.
    """
    if score >= EXCELLENT_SCORE:
        return "Excellent"
    elif score >= GOOD_SCORE:
        return "Good"
    elif score >= FAIR_SCORE:
        return "Fair"
    else:
        return "Poor"


def calculate_pass_probability(score: int) -> float:
    """
    Calculate probability of passing ATS.
    """
    if score >= 80:
        return 95.0
    elif score >= 70:
        return 85.0
    elif score >= 60:
        return 70.0
    elif score >= 50:
        return 50.0
    elif score >= 40:
        return 30.0
    else:
        return 15.0


def analyze_keyword_density(
        resume_text: str,
        keywords: List[str]
) -> Dict[str, any]:
    """
    Analyze keyword density in resume.
    """
    text_lower = resume_text.lower()
    word_count = len(resume_text.split())

    keyword_counts = {}
    total_keyword_occurrences = 0

    for keyword in keywords:
        count = text_lower.count(keyword.lower())
        if count > 0:
            keyword_counts[keyword] = count
            total_keyword_occurrences += count

    density = (total_keyword_occurrences / word_count * 100) if word_count > 0 else 0

    return {
        'keyword_counts': keyword_counts,
        'total_occurrences': total_keyword_occurrences,
        'word_count': word_count,
        'density_percentage': round(density, 2),
        'coverage': round((len(keyword_counts) / len(keywords) * 100), 2) if keywords else 0
    }


def check_ats_format_compliance(resume_text: str) -> Dict[str, any]:
    """
    Check if resume format is ATS-friendly.
    """
    issues = []
    warnings = []

    # Check for special characters that might cause issues
    special_chars = ['•', '◆', '■', '●', '▪']
    for char in special_chars:
        if char in resume_text:
            warnings.append(f"Found special character '{char}' - may not parse correctly")

    # Check for tables/complex formatting indicators
    if '\t' in resume_text:
        warnings.append("Detected tab characters - may indicate table formatting")

    # Check resume length
    word_count = len(resume_text.split())
    if word_count < 200:
        issues.append("Resume is too short (< 200 words)")
    elif word_count > 2000:
        warnings.append("Resume is very long (> 2000 words)")

    # Compliance score
    compliance_score = 100 - (len(issues) * 15) - (len(warnings) * 5)
    compliance_score = max(0, compliance_score)

    return {
        'compliance_score': compliance_score,
        'issues': issues,
        'warnings': warnings,
        'is_compliant': len(issues) == 0,
        'recommendation': (
            'Format is ATS-friendly' if len(issues) == 0
            else 'Fix format issues before applying'
        )
    }


def generate_ats_report(
        resume_text: str,
        resume_skills: List[str],
        job_role: str
) -> Dict[str, any]:
    """
    Generate comprehensive ATS report.
    """
    # Main ATS analysis
    ats_analysis = analyze_ats_compatibility(resume_text, resume_skills, job_role)

    # Format compliance
    format_analysis = check_ats_format_compliance(resume_text)

    # Overall assessment
    overall_score = int(
        ats_analysis['ats_score'] * 0.7 +
        format_analysis['compliance_score'] * 0.3
    )

    return {
        'overall_score': overall_score,
        'ats_analysis': ats_analysis,
        'format_analysis': format_analysis,
        'final_recommendation': get_final_recommendation(overall_score),
        'priority_actions': get_priority_actions(ats_analysis, format_analysis)
    }


def get_final_recommendation(score: int) -> str:
    """Get final recommendation based on overall score."""
    if score >= 80:
        return "Your resume is highly optimized for ATS. Ready to apply!"
    elif score >= 60:
        return "Your resume is ATS-compatible with minor improvements needed."
    elif score >= 40:
        return "Your resume needs significant improvements for ATS."
    else:
        return "Your resume requires major revisions for ATS compatibility."


def get_priority_actions(
        ats_analysis: Dict,
        format_analysis: Dict
) -> List[str]:
    """Get priority actions from analysis."""
    actions = []

    # Format issues first
    if format_analysis.get('issues'):
        actions.extend(format_analysis['issues'])

    # Critical skill gaps
    missing_core = ats_analysis.get('missing_core_skills', [])
    if missing_core:
        actions.append(f"Add critical skills: {', '.join(missing_core[:3])}")

    # Low scores
    if ats_analysis.get('ats_score', 0) < 50:
        actions.append("Tailor resume more closely to job description")

    return actions[:5]  # Top 5 priority actions


class ATSAnalyzer:
    """Class-based ATS analyzer for advanced usage."""

    def __init__(self):
        """Initialize ATS analyzer."""
        pass

    def analyze(
            self,
            resume_text: str,
            resume_skills: List[str],
            job_role: str
    ) -> Dict[str, any]:
        """Perform comprehensive ATS analysis."""
        return analyze_ats_compatibility(resume_text, resume_skills, job_role)

    def generate_report(
            self,
            resume_text: str,
            resume_skills: List[str],
            job_role: str
    ) -> Dict[str, any]:
        """Generate full ATS report."""
        return generate_ats_report(resume_text, resume_skills, job_role)