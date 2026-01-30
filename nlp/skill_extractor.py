import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
import logging
import re

from config import TECHNICAL_SKILLS_FILE, SOFT_SKILLS_FILE, DOMAIN_SKILLS_FILE, SKILLS_LIST_FILE

logger = logging.getLogger(__name__)

# Cache for loaded skills
_skills_cache = {}


def load_skills(skills_type: str = "all") -> List[str]:

    global _skills_cache

    # Return from cache if available
    if skills_type in _skills_cache:
        return _skills_cache[skills_type]

    skills = []

    try:
        if skills_type in ["technical", "all"]:
            skills.extend(_load_skills_from_file(TECHNICAL_SKILLS_FILE))

        if skills_type in ["soft", "all"]:
            skills.extend(_load_skills_from_file(SOFT_SKILLS_FILE))

        if skills_type in ["domain", "all"]:
            skills.extend(_load_skills_from_file(DOMAIN_SKILLS_FILE))

        # Fallback to simple text file if JSON files don't exist
        if not skills and SKILLS_LIST_FILE.exists():
            skills = _load_skills_from_txt(SKILLS_LIST_FILE)

        # Remove duplicates and normalize
        skills = list(set(skill.lower().strip() for skill in skills if skill))

        logger.info(f"Loaded {len(skills)} skills of type '{skills_type}'")

        # Cache the results
        _skills_cache[skills_type] = skills

        return skills

    except Exception as e:
        logger.error(f"Error loading skills: {str(e)}")
        return []


def _load_skills_from_file(file_path: Path) -> List[str]:

    if not file_path.exists():
        logger.warning(f"Skills file not found: {file_path}")
        return []

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Extract all values if dict
            skills = []
            for value in data.values():
                if isinstance(value, list):
                    skills.extend(value)
                elif isinstance(value, str):
                    skills.append(value)
            return skills

        return []

    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        return []


def _load_skills_from_txt(file_path: Path) -> List[str]:

    try:
        with open(file_path, 'r') as f:
            return [line.strip().lower() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        return []


def extract_skills(text: str, skills_db: List[str] = None, fuzzy: bool = True) -> List[str]:

    if not text:
        return []

    # Load skills if not provided
    if skills_db is None:
        skills_db = load_skills("all")

    if not skills_db:
        logger.warning("No skills database available")
        return []

    text_lower = text.lower()
    found_skills = set()

    # Exact matching
    for skill in skills_db:
        # Use word boundary for exact match
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)

    # Fuzzy matching for common variations
    if fuzzy:
        found_skills.update(_fuzzy_skill_match(text_lower, skills_db))

    result = sorted(list(found_skills))
    logger.debug(f"Found {len(result)} skills in text")

    return result


def _fuzzy_skill_match(text: str, skills_db: List[str]) -> Set[str]:

    matched = set()

    # Common variations
    variations = {
        'javascript': ['js', 'javascript'],
        'typescript': ['ts', 'typescript'],
        'python': ['python', 'python3', 'py'],
        'c++': ['c++', 'cpp', 'cplusplus'],
        'c#': ['c#', 'csharp'],
        'node.js': ['nodejs', 'node.js', 'node js'],
        'react.js': ['react', 'reactjs', 'react.js'],
        'vue.js': ['vue', 'vuejs', 'vue.js'],
        'angular': ['angular', 'angularjs'],
        'postgresql': ['postgres', 'postgresql'],
        'mysql': ['mysql', 'my sql'],
        'mongodb': ['mongo', 'mongodb'],
        'docker': ['docker', 'containerization'],
        'kubernetes': ['k8s', 'kubernetes'],
        'machine learning': ['ml', 'machine learning'],
        'artificial intelligence': ['ai', 'artificial intelligence'],
        'natural language processing': ['nlp', 'natural language processing']
    }

    for canonical, variants in variations.items():
        if canonical in skills_db:
            for variant in variants:
                pattern = r'\b' + re.escape(variant) + r'\b'
                if re.search(pattern, text):
                    matched.add(canonical)
                    break

    return matched


def categorize_skills(skills: List[str]) -> Dict[str, List[str]]:

    categorized = {
        "technical": [],
        "soft": [],
        "domain": []
    }

    try:
        technical_skills = load_skills("technical")
        soft_skills = load_skills("soft")
        domain_skills = load_skills("domain")

        for skill in skills:
            skill_lower = skill.lower()

            if skill_lower in technical_skills:
                categorized["technical"].append(skill)
            elif skill_lower in soft_skills:
                categorized["soft"].append(skill)
            elif skill_lower in domain_skills:
                categorized["domain"].append(skill)
            else:
                # Default to technical if unknown
                categorized["technical"].append(skill)

    except Exception as e:
        logger.error(f"Error categorizing skills: {str(e)}")
        # Put all in technical if error
        categorized["technical"] = skills

    return categorized


def extract_skill_context(text: str, skill: str, context_window: int = 50) -> List[str]:

    contexts = []
    text_lower = text.lower()
    skill_lower = skill.lower()

    # Find all occurrences
    pattern = r'\b' + re.escape(skill_lower) + r'\b'

    for match in re.finditer(pattern, text_lower):
        start = max(0, match.start() - context_window)
        end = min(len(text), match.end() + context_window)
        context = text[start:end].strip()
        contexts.append(context)

    return contexts


def get_skill_frequency(text: str, skills_db: List[str] = None) -> Dict[str, int]:

    if skills_db is None:
        skills_db = load_skills("all")

    text_lower = text.lower()
    frequency = {}

    for skill in skills_db:
        pattern = r'\b' + re.escape(skill) + r'\b'
        count = len(re.findall(pattern, text_lower))
        if count > 0:
            frequency[skill] = count

    # Sort by frequency
    return dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))


def suggest_related_skills(skills: List[str]) -> List[str]:

    # Skill relationships
    relationships = {
        'python': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'django', 'flask'],
        'java': ['spring boot', 'hibernate', 'maven', 'junit'],
        'javascript': ['react', 'node.js', 'express', 'typescript'],
        'react': ['redux', 'next.js', 'react hooks'],
        'machine learning': ['deep learning', 'neural networks', 'feature engineering'],
        'sql': ['postgresql', 'mysql', 'database design'],
        'docker': ['kubernetes', 'ci/cd', 'devops'],
        'aws': ['cloud computing', 'ec2', 's3', 'lambda']
    }

    suggestions = set()

    for skill in skills:
        skill_lower = skill.lower()
        if skill_lower in relationships:
            suggestions.update(relationships[skill_lower])

    # Remove skills already present
    suggestions = suggestions - set(s.lower() for s in skills)

    return sorted(list(suggestions))


class SkillExtractor:

    def __init__(self, skills_type: str = "all"):

        self.skills_db = load_skills(skills_type)
        self.skills_type = skills_type

    def extract(self, text: str, fuzzy: bool = True) -> List[str]:

        return extract_skills(text, self.skills_db, fuzzy)

    def extract_with_frequency(self, text: str) -> Dict[str, int]:

        return get_skill_frequency(text, self.skills_db)

    def extract_with_context(self, text: str, context_window: int = 50) -> Dict[str, List[str]]:

        skills = self.extract(text)
        result = {}

        for skill in skills:
            contexts = extract_skill_context(text, skill, context_window)
            if contexts:
                result[skill] = contexts

        return result


# Backward compatibility
def load_skills_db(path: str = None) -> List[str]:

    if path:
        return _load_skills_from_txt(Path(path))
    return load_skills("all")