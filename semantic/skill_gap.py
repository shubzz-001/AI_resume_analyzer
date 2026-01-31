from typing import List, Tuple, Dict
import logging

from semantic.embeddings import embed_text, cosine_similarity
from config import SEMANTIC_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


def semantic_skill_gap(
        resume_text: str,
        required_skills: List[str],
        threshold: float = SEMANTIC_SIMILARITY_THRESHOLD
) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    Identify skill gaps using semantic similarity.
    """
    if not resume_text or not required_skills:
        return [], required_skills

    try:
        # Generate embeddings
        resume_vec = embed_text(resume_text)
        skill_vecs = embed_text(required_skills)

        matched = []
        missing = []

        # FIXED BUG: Use required_skills in zip, not resume_vec
        for skill, vec in zip(required_skills, skill_vecs):
            sim = cosine_similarity(resume_vec, vec)

            if sim >= threshold:
                matched.append((skill, round(sim * 100, 1)))
            else:
                missing.append(skill)

        logger.info(
            f"Skill gap analysis: {len(matched)} matched, "
            f"{len(missing)} missing out of {len(required_skills)} total"
        )

        return matched, missing

    except Exception as e:
        logger.error(f"Error in semantic skill gap analysis: {str(e)}")
        return [], required_skills


def semantic_match_skills(
        resume_text: str,
        required_skills: List[str],
        threshold: float = SEMANTIC_SIMILARITY_THRESHOLD
) -> List[Tuple[str, float]]:
    """
    Match skills semantically and return sorted by similarity.
    FIXED: Bug in original skill_matcher.py - corrected zip logic.
    """
    if not resume_text or not required_skills:
        return []

    try:
        resume_vec = embed_text(resume_text)
        skill_vecs = embed_text(required_skills)

        results = []

        # FIXED BUG: Use required_skills, not resume_vec
        for skill, vec in zip(required_skills, skill_vecs):
            score = cosine_similarity(resume_vec, vec)
            if score >= threshold:
                results.append((skill, round(score * 100, 1)))

        # Sort by similarity descending
        return sorted(results, key=lambda x: x[1], reverse=True)

    except Exception as e:
        logger.error(f"Error matching skills: {str(e)}")
        return []


def analyze_skill_coverage(
        resume_text: str,
        required_skills: List[str]
) -> Dict[str, any]:
    """
    Comprehensive skill coverage analysis.
    """
    matched, missing = semantic_skill_gap(resume_text, required_skills)

    coverage_percentage = (len(matched) / len(required_skills) * 100) if required_skills else 0

    return {
        'coverage_percentage': round(coverage_percentage, 2),
        'matched_skills': matched,
        'missing_skills': missing,
        'total_required': len(required_skills),
        'matched_count': len(matched),
        'missing_count': len(missing),
        'recommendation': (
            'Excellent coverage!' if coverage_percentage >= 80 else
            'Good coverage' if coverage_percentage >= 60 else
            'Needs improvement'
        )
    }