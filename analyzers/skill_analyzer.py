from typing import Dict, List, Tuple
import logging

from nlp.skill_extractor import categorize_skills, suggest_related_skills
from semantic.skill_gap import semantic_skill_gap, analyze_skill_coverage

logger = logging.getLogger(__name__)


def analyze_skills(
        resume_skills: List[str],
        resume_text: str,
        job_role: str = None
) -> Dict[str, any]:
    """
    Comprehensive skill analysis.
    """
    # Categorize skills
    categorized = categorize_skills(resume_skills)

    # Get suggestions
    related_suggestions = suggest_related_skills(resume_skills)

    # Calculate skill strength
    skill_strength = calculate_skill_strength(len(resume_skills))

    analysis = {
        'total_skills': len(resume_skills),
        'skills_list': resume_skills,
        'categorized_skills': categorized,
        'technical_count': len(categorized.get('technical', [])),
        'soft_count': len(categorized.get('soft', [])),
        'domain_count': len(categorized.get('domain', [])),
        'skill_strength': skill_strength,
        'related_suggestions': related_suggestions[:10],
        'recommendation': get_skill_recommendation(len(resume_skills))
    }

    return analysis


def calculate_skill_strength(skill_count: int) -> str:
    """
    Calculate skill strength level.
    """
    if skill_count >= 15:
        return "Strong"
    elif skill_count >= 10:
        return "Good"
    elif skill_count >= 5:
        return "Fair"
    else:
        return "Weak"


def get_skill_recommendation(skill_count: int) -> str:
    """Get recommendation based on skill count."""
    if skill_count >= 15:
        return "Excellent skill coverage!"
    elif skill_count >= 10:
        return "Good skill set. Consider adding specialized skills."
    elif skill_count >= 5:
        return "Add more relevant technical skills."
    else:
        return "Significantly expand your skill set."


def analyze_skill_gaps(
        resume_text: str,
        resume_skills: List[str],
        required_skills: List[str]
) -> Dict[str, any]:
    """
    Analyze skill gaps for specific requirements.
    """
    # Semantic gap analysis
    matched, missing = semantic_skill_gap(resume_text, required_skills)

    # Get coverage analysis
    coverage = analyze_skill_coverage(resume_text, required_skills)

    return {
        'coverage_percentage': coverage['coverage_percentage'],
        'matched_skills': matched,
        'missing_skills': missing,
        'matched_count': len(matched),
        'missing_count': len(missing),
        'priority_skills': missing[:5],  # Top 5 to add
        'recommendation': coverage['recommendation']
    }


def compare_skill_sets(
        candidate_skills: List[str],
        ideal_skills: List[str]
) -> Dict[str, any]:
    """
    Compare candidate skills against ideal skill set.
    """
    candidate_set = set(s.lower() for s in candidate_skills)
    ideal_set = set(s.lower() for s in ideal_skills)

    matched = candidate_set & ideal_set
    missing = ideal_set - candidate_set
    extra = candidate_set - ideal_set

    match_percentage = (len(matched) / len(ideal_set) * 100) if ideal_set else 0

    return {
        'match_percentage': round(match_percentage, 2),
        'matched_skills': sorted(list(matched)),
        'missing_skills': sorted(list(missing)),
        'additional_skills': sorted(list(extra)),
        'matched_count': len(matched),
        'missing_count': len(missing),
        'assessment': (
            'Excellent match' if match_percentage >= 80 else
            'Good match' if match_percentage >= 60 else
            'Partial match' if match_percentage >= 40 else
            'Poor match'
        )
    }


def rank_skills_by_importance(
        skills: List[str],
        job_role: str
) -> List[Tuple[str, int]]:
    """
    Rank skills by importance for job role.
    """
    # Import role keywords
    from ml.model_config import ROLE_KEYWORDS

    role_skills = ROLE_KEYWORDS.get(job_role, [])
    role_skills_set = set(s.lower() for s in role_skills)

    ranked = []
    for skill in skills:
        # Higher score if in role keywords
        importance = 10 if skill.lower() in role_skills_set else 5
        ranked.append((skill, importance))

    # Sort by importance
    return sorted(ranked, key=lambda x: x[1], reverse=True)


def generate_skill_report(
        resume_text: str,
        resume_skills: List[str],
        job_role: str,
        required_skills: List[str] = None
) -> Dict[str, any]:
    """
    Generate comprehensive skill report.
    """
    # Basic analysis
    analysis = analyze_skills(resume_skills, resume_text, job_role)

    # Gap analysis if required skills provided
    gap_analysis = None
    if required_skills:
        gap_analysis = analyze_skill_gaps(
            resume_text,
            resume_skills,
            required_skills
        )

    # Rank skills
    ranked_skills = rank_skills_by_importance(resume_skills, job_role)

    return {
        'skill_analysis': analysis,
        'gap_analysis': gap_analysis,
        'ranked_skills': ranked_skills[:10],  # Top 10
        'summary': generate_skill_summary(analysis, gap_analysis)
    }


def generate_skill_summary(
        analysis: Dict,
        gap_analysis: Dict = None
) -> str:
    """Generate skill summary text."""
    total = analysis['total_skills']
    strength = analysis['skill_strength']

    summary = f"Found {total} skills with {strength.lower()} overall coverage. "

    if gap_analysis:
        coverage = gap_analysis['coverage_percentage']
        summary += f"Matches {coverage:.0f}% of required skills. "

        if gap_analysis['missing_count'] > 0:
            summary += f"Consider adding: {', '.join(gap_analysis['priority_skills'][:3])}."

    return summary


class SkillAnalyzer:
    """Class-based skill analyzer."""

    def __init__(self):
        """Initialize skill analyzer."""
        pass

    def analyze(
            self,
            resume_skills: List[str],
            resume_text: str,
            job_role: str = None
    ) -> Dict[str, any]:
        """Analyze skills."""
        return analyze_skills(resume_skills, resume_text, job_role)

    def find_gaps(
            self,
            resume_text: str,
            resume_skills: List[str],
            required_skills: List[str]
    ) -> Dict[str, any]:
        """Find skill gaps."""
        return analyze_skill_gaps(resume_text, resume_skills, required_skills)

    def generate_report(
            self,
            resume_text: str,
            resume_skills: List[str],
            job_role: str,
            required_skills: List[str] = None
    ) -> Dict[str, any]:
        """Generate complete skill report."""
        return generate_skill_report(
            resume_text,
            resume_skills,
            job_role,
            required_skills
        )