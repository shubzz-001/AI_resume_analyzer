from typing import Dict, List, Tuple
import logging

from semantic.embeddings import embed_text, cosine_similarity
from config import SEMANTIC_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


def semantic_ats_score(
        resume_text: str,
        weighted_skills: Dict[str, int],
        threshold: float = SEMANTIC_SIMILARITY_THRESHOLD
) -> Tuple[int, List[Tuple[str, float]]]:
    """
    Calculate ATS score using semantic similarity.
    """
    # Validate inputs
    if not resume_text or not isinstance(resume_text, str):
        raise ValueError("Resume text must be a non-empty string")

    if not weighted_skills:
        logger.warning("No weighted skills provided")
        return 0, []

    try:
        # Generate resume embedding
        resume_vec = embed_text(resume_text)

        # Calculate total weight
        total_weight = sum(weighted_skills.values())

        if total_weight == 0:
            logger.warning("Total weight is zero")
            return 0, []

        # Calculate gained weight from matching skills
        gained = 0
        explanations = []

        # Get skill embeddings
        skill_texts = list(weighted_skills.keys())
        skill_vecs = embed_text(skill_texts)

        # Calculate similarities
        for skill, vec in zip(skill_texts, skill_vecs):
            sim = cosine_similarity(resume_vec, vec)

            # Check if similarity meets threshold
            if sim >= threshold:
                gained += weighted_skills[skill]
                # FIXED BUG: Use tuple instead of two separate arguments
                explanations.append((skill, round(sim * 100, 1)))
                logger.debug(f"Skill '{skill}' matched with {sim:.2%} similarity")

        # Calculate final score
        score = int((gained / total_weight) * 100) if total_weight > 0 else 0

        logger.info(f"Semantic ATS score: {score}/100 with {len(explanations)} matching skills")

        return score, explanations

    except Exception as e:
        logger.error(f"Error calculating semantic ATS score: {str(e)}")
        raise RuntimeError(f"Semantic ATS scoring failed: {str(e)}")


def calculate_skill_match(
        resume_text: str,
        required_skills: List[str],
        threshold: float = SEMANTIC_SIMILARITY_THRESHOLD
) -> Dict[str, any]:
    """
    Calculate how well resume matches required skills.
    """
    if not resume_text or not required_skills:
        return {
            'match_percentage': 0,
            'matched_skills': [],
            'missing_skills': required_skills,
            'total_skills': len(required_skills)
        }

    try:
        # Generate embeddings
        resume_vec = embed_text(resume_text)
        skill_vecs = embed_text(required_skills)

        matched = []
        missing = []

        # Check each skill
        for skill, vec in zip(required_skills, skill_vecs):
            sim = cosine_similarity(resume_vec, vec)

            if sim >= threshold:
                matched.append({
                    'skill': skill,
                    'similarity': round(sim * 100, 1)
                })
            else:
                missing.append(skill)

        match_percentage = (len(matched) / len(required_skills)) * 100 if required_skills else 0

        return {
            'match_percentage': round(match_percentage, 2),
            'matched_skills': matched,
            'missing_skills': missing,
            'total_skills': len(required_skills),
            'matched_count': len(matched),
            'missing_count': len(missing)
        }

    except Exception as e:
        logger.error(f"Error calculating skill match: {str(e)}")
        return {
            'match_percentage': 0,
            'matched_skills': [],
            'missing_skills': required_skills,
            'total_skills': len(required_skills),
            'error': str(e)
        }


def get_semantic_score_breakdown(
        resume_text: str,
        job_requirements: Dict[str, any]
) -> Dict[str, any]:
    """
    Get detailed semantic scoring breakdown.
    """
    results = {
        'total_score': 0,
        'core_skills': {},
        'optional_skills': {},
        'recommendations': []
    }

    try:
        # Core skills scoring
        if 'core' in job_requirements:
            core_score, core_matches = semantic_ats_score(
                resume_text,
                job_requirements['core']
            )
            results['core_skills'] = {
                'score': core_score,
                'matches': core_matches,
                'weight': 0.7  # 70% weight
            }

        # Optional skills scoring
        if 'optional' in job_requirements:
            optional_score, optional_matches = semantic_ats_score(
                resume_text,
                job_requirements['optional']
            )
            results['optional_skills'] = {
                'score': optional_score,
                'matches': optional_matches,
                'weight': 0.3  # 30% weight
            }

        # Calculate total score
        total = 0
        if results['core_skills']:
            total += results['core_skills']['score'] * 0.7
        if results['optional_skills']:
            total += results['optional_skills']['score'] * 0.3

        results['total_score'] = int(total)

        # Generate recommendations
        if results['core_skills'] and results['core_skills']['score'] < 70:
            results['recommendations'].append(
                "Focus on adding more core skills to improve your score"
            )

        if results['optional_skills'] and results['optional_skills']['score'] < 50:
            results['recommendations'].append(
                "Consider adding optional skills to stand out"
            )

    except Exception as e:
        logger.error(f"Error in semantic score breakdown: {str(e)}")
        results['error'] = str(e)

    return results


def compare_semantic_match(
        text1: str,
        text2: str
) -> Dict[str, any]:
    """
    Compare semantic similarity between two texts.
    """
    try:
        emb1 = embed_text(text1)
        emb2 = embed_text(text2)

        similarity = cosine_similarity(emb1, emb2)

        return {
            'similarity': round(similarity * 100, 2),
            'similarity_score': similarity,
            'match_level': (
                'Strong' if similarity >= 0.7 else
                'Moderate' if similarity >= 0.5 else
                'Weak'
            )
        }

    except Exception as e:
        logger.error(f"Error comparing texts: {str(e)}")
        return {
            'similarity': 0,
            'error': str(e)
        }


class SemanticATSScorer:
    """
    Class-based semantic ATS scorer for advanced usage.
    """

    def __init__(self, threshold: float = SEMANTIC_SIMILARITY_THRESHOLD):
        """
        Initialize semantic ATS scorer.
        """
        self.threshold = threshold

    def score(
            self,
            resume_text: str,
            weighted_skills: Dict[str, int]
    ) -> Tuple[int, List[Tuple[str, float]]]:
        """
        Calculate semantic ATS score.
        """
        return semantic_ats_score(resume_text, weighted_skills, self.threshold)

    def match_skills(
            self,
            resume_text: str,
            required_skills: List[str]
    ) -> Dict[str, any]:
        """
        Match resume against required skills.
        """
        return calculate_skill_match(resume_text, required_skills, self.threshold)

    def detailed_analysis(
            self,
            resume_text: str,
            job_requirements: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Detailed semantic analysis.
        """
        return get_semantic_score_breakdown(resume_text, job_requirements)