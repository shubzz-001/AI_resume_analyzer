import json
from typing import Dict, List, Tuple
import logging

from semantic.semantic_ats import semantic_ats_score
from config import (
    KEYWORD_WEIGHT,
    SEMANTIC_WEIGHT,
    ATS_JOB_SKILLS_FILE
)

logger = logging.getLogger(__name__)


def load_ats_data(job_role: str) -> Dict[str, any]:
    """
    Load ATS data for specific job role.
    """
    if not ATS_JOB_SKILLS_FILE.exists():
        raise FileNotFoundError(f"ATS data file not found: {ATS_JOB_SKILLS_FILE}")

    try:
        with open(ATS_JOB_SKILLS_FILE, 'r') as f:
            ats_data = json.load(f)

        if job_role not in ats_data:
            available_roles = list(ats_data.keys())
            raise ValueError(
                f"Job role '{job_role}' not found. "
                f"Available roles: {', '.join(available_roles)}"
            )

        return ats_data[job_role]

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid ATS data file: {str(e)}")


def calculate_keyword_ats_score(
        job_role: str,
        resume_skills: List[str],
        resume_text: str
) -> Tuple[int, Dict[str, any]]:
    """
    Calculate keyword-based ATS score.
    """
    try:
        # Load ATS data
        role_data = load_ats_data(job_role)

        # Get skill requirements
        core_skills = role_data.get('core', {})
        optional_skills = role_data.get('optional', {})

        # Calculate total weight
        total_weight = sum(core_skills.values()) + sum(optional_skills.values())

        if total_weight == 0:
            logger.warning(f"No skill weights defined for {job_role}")
            return 0, {}

        # Convert resume skills to set for fast lookup
        resume_skills_set = set(s.lower() for s in resume_skills)
        resume_text_lower = resume_text.lower()

        # Calculate matched weight
        matched_weight = 0
        matched_core = []
        matched_optional = []
        missing_core = []
        missing_optional = []

        # Check core skills
        for skill, weight in core_skills.items():
            if skill in resume_skills_set or skill in resume_text_lower:
                matched_weight += weight
                matched_core.append(skill)
            else:
                missing_core.append(skill)

        # Check optional skills
        for skill, weight in optional_skills.items():
            if skill in resume_skills_set or skill in resume_text_lower:
                matched_weight += weight
                matched_optional.append(skill)
            else:
                missing_optional.append(skill)

        # Calculate score
        score = int((matched_weight / total_weight) * 100)

        details = {
            'matched_core': matched_core,
            'matched_optional': matched_optional,
            'missing_core': missing_core,
            'missing_optional': missing_optional,
            'total_core_skills': len(core_skills),
            'total_optional_skills': len(optional_skills),
            'matched_weight': matched_weight,
            'total_weight': total_weight
        }

        logger.info(f"Keyword ATS score for {job_role}: {score}/100")

        return score, details

    except Exception as e:
        logger.error(f"Error calculating keyword ATS score: {str(e)}")
        raise RuntimeError(f"Keyword ATS scoring failed: {str(e)}")


def hybrid_ats_score(
        resume_text: str,
        resume_skills: List[str],
        job_role: str,
        keyword_weight: float = KEYWORD_WEIGHT,
        semantic_weight: float = SEMANTIC_WEIGHT
) -> Tuple[int, Dict[str, any]]:
    """
    Calculate hybrid ATS score combining keyword and semantic approaches.
    """
    # Validate inputs
    if not resume_text or not isinstance(resume_text, str):
        raise ValueError("Resume text must be a non-empty string")

    if job_role not in get_available_job_roles():
        raise ValueError(f"Unsupported job role: {job_role}")

    # Validate weights
    if abs((keyword_weight + semantic_weight) - 1.0) > 0.01:
        raise ValueError("Keyword weight and semantic weight must sum to 1.0")

    try:
        # Calculate keyword-based score
        keyword_score, keyword_details = calculate_keyword_ats_score(
            job_role,
            resume_skills,
            resume_text
        )

        # Calculate semantic score
        role_data = load_ats_data(job_role)
        weighted_skills = role_data.get('core', {})

        semantic_score, semantic_matches = semantic_ats_score(
            resume_text,
            weighted_skills
        )

        # Combine scores
        final_score = int(
            (keyword_score * keyword_weight) +
            (semantic_score * semantic_weight)
        )

        # Create comprehensive breakdown
        breakdown = {
            'final_score': final_score,
            'keyword_score': keyword_score,
            'semantic_score': semantic_score,
            'weights': {
                'keyword': keyword_weight,
                'semantic': semantic_weight
            },
            'keyword_details': keyword_details,
            'semantic_matches': [
                {'skill': skill, 'similarity': sim}
                for skill, sim in semantic_matches
            ],
            'job_role': job_role,
            'recommendation': get_score_recommendation(final_score)
        }

        logger.info(
            f"Hybrid ATS score for {job_role}: {final_score}/100 "
            f"(Keyword: {keyword_score}, Semantic: {semantic_score})"
        )

        return final_score, breakdown

    except Exception as e:
        logger.error(f"Error calculating hybrid ATS score: {str(e)}")
        raise RuntimeError(f"Hybrid ATS scoring failed: {str(e)}")


def get_score_recommendation(score: int) -> str:
    """
    Get recommendation based on ATS score.
    """
    if score >= 80:
        return "Excellent! Your resume is highly optimized for ATS systems."
    elif score >= 60:
        return "Good! Your resume should pass most ATS systems. Consider adding more relevant skills."
    elif score >= 40:
        return "Fair. Your resume needs improvement to reliably pass ATS systems."
    else:
        return "Low ATS compatibility. Focus on adding core skills and relevant keywords."


def get_available_job_roles() -> List[str]:
    """
    Get list of available job roles from ATS data.
    """
    try:
        with open(ATS_JOB_SKILLS_FILE, 'r') as f:
            ats_data = json.load(f)
        return list(ats_data.keys())
    except Exception as e:
        logger.error(f"Error loading job roles: {str(e)}")
        return []


def compare_ats_scores(
        resume_text: str,
        resume_skills: List[str],
        job_roles: List[str]
) -> Dict[str, Dict[str, any]]:
    """
    Compare ATS scores across multiple job roles.
    """
    results = {}

    for role in job_roles:
        try:
            score, breakdown = hybrid_ats_score(
                resume_text,
                resume_skills,
                role
            )
            results[role] = {
                'score': score,
                'breakdown': breakdown
            }
        except Exception as e:
            logger.warning(f"Could not calculate ATS score for {role}: {str(e)}")
            results[role] = {
                'score': 0,
                'error': str(e)
            }

    # Sort by score
    results = dict(sorted(results.items(), key=lambda x: x[1].get('score', 0), reverse=True))

    return results


def get_ats_improvement_suggestions(
        breakdown: Dict[str, any]
) -> List[str]:
    """
    Generate suggestions to improve ATS score.
    """
    suggestions = []

    keyword_details = breakdown.get('keyword_details', {})
    final_score = breakdown.get('final_score', 0)

    # Missing core skills
    missing_core = keyword_details.get('missing_core', [])
    if missing_core:
        suggestions.append(
            f"Add these high-priority skills: {', '.join(missing_core[:5])}"
        )

    # Missing optional skills
    missing_optional = keyword_details.get('missing_optional', [])
    if missing_optional:
        suggestions.append(
            f"Consider adding these skills: {', '.join(missing_optional[:3])}"
        )

    # Low semantic score
    semantic_score = breakdown.get('semantic_score', 0)
    if semantic_score < 50:
        suggestions.append(
            "Rephrase your experience to better match job requirements"
        )

    # Low keyword score
    keyword_score = breakdown.get('keyword_score', 0)
    if keyword_score < 50:
        suggestions.append(
            "Include more specific technical keywords from the job description"
        )

    # Overall low score
    if final_score < 60:
        suggestions.append(
            "Tailor your resume more closely to the specific job role"
        )

    return suggestions


class HybridATSScorer:
    """
    Class-based hybrid ATS scorer for advanced usage.
    """

    def __init__(
            self,
            keyword_weight: float = KEYWORD_WEIGHT,
            semantic_weight: float = SEMANTIC_WEIGHT
    ):

        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight

    def score(
            self,
            resume_text: str,
            resume_skills: List[str],
            job_role: str
    ) -> Tuple[int, Dict[str, any]]:
        """
        Calculate hybrid ATS score.
        """
        return hybrid_ats_score(
            resume_text,
            resume_skills,
            job_role,
            self.keyword_weight,
            self.semantic_weight
        )

    def compare_roles(
            self,
            resume_text: str,
            resume_skills: List[str],
            job_roles: List[str]
    ) -> Dict[str, Dict]:
        """
        Compare scores across roles.
        """
        return compare_ats_scores(resume_text, resume_skills, job_roles)

    def get_suggestions(self, breakdown: Dict[str, any]) -> List[str]:
        """
        Get improvement suggestions.
        """
        return get_ats_improvement_suggestions(breakdown)