from analyzers.ats_analyzer import (
    analyze_ats_compatibility,
    check_ats_format_compliance,
    generate_ats_report,
    ATSAnalyzer
)

from analyzers.skill_analyzer import (
    analyze_skills,
    analyze_skill_gaps,
    compare_skill_sets,
    generate_skill_report,
    SkillAnalyzer
)

from analyzers.experience_analyzer import (
    analyze_experience,
    get_experience_level
)

from analyzers.format_analyzer import (
    analyze_format,
    check_sections
)

from analyzers.content_quality import (
    analyze_content_quality
)

__all__ = [
    # ATS Analyzer
    'analyze_ats_compatibility',
    'check_ats_format_compliance',
    'generate_ats_report',
    'ATSAnalyzer',

    # Skill Analyzer
    'analyze_skills',
    'analyze_skill_gaps',
    'compare_skill_sets',
    'generate_skill_report',
    'SkillAnalyzer',

    # Experience Analyzer
    'analyze_experience',
    'get_experience_level',

    # Format Analyzer
    'analyze_format',
    'check_sections',

    # Content Quality
    'analyze_content_quality'
]


# Convenience function for complete analysis
def complete_resume_analysis(
        resume_text: str,
        resume_skills: list,
        job_role: str,
        required_skills: list = None
) -> dict:
    """
    Perform complete resume analysis.
    """
    return {
        'ats_analysis': analyze_ats_compatibility(resume_text, resume_skills, job_role),
        'skill_analysis': analyze_skills(resume_skills, resume_text, job_role),
        'experience_analysis': analyze_experience(resume_text),
        'format_analysis': analyze_format(resume_text),
        'content_quality': analyze_content_quality(resume_text)
    }