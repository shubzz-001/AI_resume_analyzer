from nlp.cleaner import (
    clean_text,
    clean_for_display,
    normalize_text,
    remove_noise,
    extract_sentences,
    get_token_count,
    get_word_frequency,
    TextCleaner
)

from nlp.skill_extractor import (
    load_skills,
    extract_skills,
    categorize_skills,
    extract_skill_context,
    get_skill_frequency,
    suggest_related_skills,
    SkillExtractor
)

from nlp.entity_extractor import (
    extract_emails,
    extract_phones,
    extract_linkedin,
    extract_github,
    extract_urls,
    extract_name,
    extract_dates,
    extract_years_of_experience,
    extract_education_degree,
    extract_companies,
    extract_locations,
    extract_all_entities,
    EntityExtractor
)

from nlp.text_analyzer import (
    count_action_verbs,
    detect_weak_verbs,
    detect_quantification,
    calculate_quantification_score,
    analyze_sentence_length,
    calculate_readability_score,
    detect_passive_voice,
    detect_first_person,
    analyze_bullet_points,
    get_content_quality_score,
    TextAnalyzer
)

__all__ = [
    # Cleaner
    'clean_text',
    'clean_for_display',
    'normalize_text',
    'remove_noise',
    'extract_sentences',
    'get_token_count',
    'get_word_frequency',
    'TextCleaner',

    # Skill Extractor
    'load_skills',
    'extract_skills',
    'categorize_skills',
    'extract_skill_context',
    'get_skill_frequency',
    'suggest_related_skills',
    'SkillExtractor',

    # Entity Extractor
    'extract_emails',
    'extract_phones',
    'extract_linkedin',
    'extract_github',
    'extract_urls',
    'extract_name',
    'extract_dates',
    'extract_years_of_experience',
    'extract_education_degree',
    'extract_companies',
    'extract_locations',
    'extract_all_entities',
    'EntityExtractor',

    # Text Analyzer
    'count_action_verbs',
    'detect_weak_verbs',
    'detect_quantification',
    'calculate_quantification_score',
    'analyze_sentence_length',
    'calculate_readability_score',
    'detect_passive_voice',
    'detect_first_person',
    'analyze_bullet_points',
    'get_content_quality_score',
    'TextAnalyzer',
]


# Convenience functions for common workflows

def process_resume_text(text: str) -> dict:

    return {
        'cleaned_text': clean_text(text),
        'skills': extract_skills(clean_text(text)),
        'entities': extract_all_entities(text),
        'quality_score': get_content_quality_score(text)
    }


def extract_profile_info(text: str) -> dict:

    extractor = EntityExtractor()
    return extractor.extract_contact_info(text)


def analyze_text_quality(text: str) -> dict:

    analyzer = TextAnalyzer()
    quality = analyzer.analyze(text)
    suggestions = analyzer.get_suggestions(text)

    quality['suggestions'] = suggestions
    return quality


def get_skills_with_categories(text: str) -> dict:

    cleaned = clean_text(text)
    skills = extract_skills(cleaned)
    categorized = categorize_skills(skills)

    return {
        'all_skills': skills,
        'categorized': categorized,
        'count': len(skills)
    }