from typing import List, Dict

from semantic.embeddings import (
    embed_text,
    embed,
    cosine_similarity,
    cosine_sim,
    batch_cosine_similarity,
    semantic_search,
    get_text_similarity,
    get_embedding_dimension,
    clear_embedding_cache,
    EmbeddingGenerator
)

from semantic.cache import (
    get_cached_embedding,
    set_cached_embedding,
    clear_cache,
    get_cache_size,
    get_cache_stats,
    cache_info,
    EmbeddingCache
)

from semantic.semantic_ats import (
    semantic_ats_score,
    calculate_skill_match,
    get_semantic_score_breakdown,
    compare_semantic_match,
    SemanticATSScorer
)

from semantic.hybrid_ats import (
    hybrid_ats_score,
    calculate_keyword_ats_score,
    get_available_job_roles,
    compare_ats_scores,
    get_ats_improvement_suggestions,
    HybridATSScorer
)

from semantic.semantic_matcher import (
    semantic_recommend_jobs,
    semantic_recommend_jobs_v2
)

from semantic.skill_gap import (
    semantic_skill_gap,
    semantic_match_skills,
    analyze_skill_coverage
)

__all__ = [
    # Embeddings
    'embed_text',
    'embed',
    'cosine_similarity',
    'cosine_sim',
    'batch_cosine_similarity',
    'semantic_search',
    'get_text_similarity',
    'get_embedding_dimension',
    'clear_embedding_cache',
    'EmbeddingGenerator',

    # Cache
    'get_cached_embedding',
    'set_cached_embedding',
    'clear_cache',
    'get_cache_size',
    'get_cache_stats',
    'cache_info',
    'EmbeddingCache',

    # Semantic ATS
    'semantic_ats_score',
    'calculate_skill_match',
    'get_semantic_score_breakdown',
    'compare_semantic_match',
    'SemanticATSScorer',

    # Hybrid ATS
    'hybrid_ats_score',
    'calculate_keyword_ats_score',
    'get_available_job_roles',
    'compare_ats_scores',
    'get_ats_improvement_suggestions',
    'HybridATSScorer',

    # Job Matching
    'semantic_recommend_jobs',
    'semantic_recommend_jobs_v2',

    # Skill Gap
    'semantic_skill_gap',
    'semantic_match_skills',
    'analyze_skill_coverage'
]


# Convenience functions

def quick_ats_analysis(resume_text: str, skills: List[str], job_role: str) -> dict:
    """
    Quick ATS analysis with hybrid scoring.
    """
    score, breakdown = hybrid_ats_score(resume_text, skills, job_role)
    suggestions = get_ats_improvement_suggestions(breakdown)

    return {
        'score': score,
        'breakdown': breakdown,
        'suggestions': suggestions
    }


def find_best_matching_jobs(resume_text: str, top_n: int = 5) -> List[dict]:
    """
    Find best matching jobs for resume.
    """
    return semantic_recommend_jobs_v2(resume_text, top_n)


def analyze_skills_for_role(
    resume_text: str,
    job_role: str
) -> dict:
    """
    Analyze skills for specific job role.
    """
    # This would require loading required skills for the role
    # Placeholder implementation
    return {
        'role': job_role,
        'analysis': 'Skill analysis for role'
    }