import re
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Action verbs categorized by context
ACTION_VERBS = {
    'leadership': [
        'led', 'managed', 'directed', 'supervised', 'coordinated',
        'oversaw', 'guided', 'mentored', 'trained', 'facilitated'
    ],
    'achievement': [
        'achieved', 'accomplished', 'delivered', 'exceeded', 'surpassed',
        'outperformed', 'completed', 'earned', 'won', 'attained'
    ],
    'technical': [
        'developed', 'programmed', 'built', 'engineered', 'designed',
        'implemented', 'deployed', 'optimized', 'automated', 'integrated'
    ],
    'analytical': [
        'analyzed', 'evaluated', 'assessed', 'investigated', 'researched',
        'examined', 'measured', 'tested', 'validated', 'verified'
    ],
    'creative': [
        'created', 'designed', 'innovated', 'invented', 'originated',
        'conceptualized', 'pioneered', 'established', 'launched', 'introduced'
    ],
    'improvement': [
        'improved', 'enhanced', 'optimized', 'streamlined', 'upgraded',
        'modernized', 'refined', 'restructured', 'transformed', 'revitalized'
    ]
}

# Weak verbs to avoid
WEAK_VERBS = [
    'responsible for', 'worked on', 'helped with', 'did', 'made',
    'handled', 'dealt with', 'was', 'had', 'got'
]

# Quantification patterns
QUANTIFICATION_PATTERNS = [
    r'\d+%',  # Percentages
    r'\$\d+[KkMmBb]?',  # Money
    r'\d+[KkMmBb]?\+?\s*(?:users|customers|clients)',  # Users/customers
    r'\d+x\s+(?:faster|improvement|increase)',  # Multipliers
    r'by\s+\d+%',  # By X percent
    r'saved\s+\$?\d+',  # Saved money/time
    r'reduced\s+\w+\s+by\s+\d+',  # Reduced by
    r'increased\s+\w+\s+by\s+\d+'  # Increased by
]


def count_action_verbs(text: str) -> Dict[str, int]:

    text_lower = text.lower()
    counts = {}

    for category, verbs in ACTION_VERBS.items():
        count = sum(1 for verb in verbs if re.search(r'\b' + verb + r'\b', text_lower))
        counts[category] = count

    counts['total'] = sum(counts.values())

    logger.debug(f"Found {counts['total']} action verbs")
    return counts


def detect_weak_verbs(text: str) -> List[Tuple[str, int]]:

    text_lower = text.lower()
    weak_found = []

    for weak_verb in WEAK_VERBS:
        count = len(re.findall(r'\b' + re.escape(weak_verb) + r'\b', text_lower))
        if count > 0:
            weak_found.append((weak_verb, count))

    logger.debug(f"Found {len(weak_found)} types of weak verbs")
    return weak_found


def detect_quantification(text: str) -> List[str]:

    quantified = []

    for pattern in QUANTIFICATION_PATTERNS:
        matches = re.findall(r'[^.!?]*' + pattern + r'[^.!?]*[.!?]', text, re.IGNORECASE)
        quantified.extend(matches)

    logger.debug(f"Found {len(quantified)} quantified statements")
    return quantified


def calculate_quantification_score(text: str) -> float:

    quantified = detect_quantification(text)

    # Count total bullet points/sentences
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    # Calculate percentage
    score = (len(quantified) / len(sentences)) * 100

    # Cap at 100
    return min(score, 100.0)


def analyze_sentence_length(text: str) -> Dict[str, float]:

    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return {'average': 0, 'min': 0, 'max': 0, 'count': 0}

    lengths = [len(s.split()) for s in sentences]

    return {
        'average': sum(lengths) / len(lengths),
        'min': min(lengths),
        'max': max(lengths),
        'count': len(sentences)
    }


def calculate_readability_score(text: str) -> float:

    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    words = text.split()

    if not sentences or not words:
        return 0.0

    avg_sentence_length = len(words) / len(sentences)

    # Simplified scoring (ideal is 15-20 words per sentence)
    if 15 <= avg_sentence_length <= 20:
        score = 100
    elif avg_sentence_length < 15:
        score = 80 + (avg_sentence_length / 15) * 20
    else:
        score = max(0, 100 - (avg_sentence_length - 20) * 3)

    return min(100, max(0, score))


def detect_passive_voice(text: str) -> List[str]:

    # Simple pattern for passive voice (to be + past participle)
    passive_pattern = r'\b(?:am|is|are|was|were|be|been|being)\s+\w+ed\b'

    sentences = re.split(r'[.!?]', text)
    passive_sentences = []

    for sentence in sentences:
        if re.search(passive_pattern, sentence, re.IGNORECASE):
            passive_sentences.append(sentence.strip())

    logger.debug(f"Found {len(passive_sentences)} passive voice sentences")
    return passive_sentences


def detect_first_person(text: str) -> int:

    first_person = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
    text_lower = text.lower()

    count = sum(len(re.findall(r'\b' + pronoun + r'\b', text_lower)) for pronoun in first_person)

    logger.debug(f"Found {count} first-person pronouns")
    return count


def analyze_bullet_points(text: str) -> Dict[str, any]:

    # Detect bullet points (lines starting with -, •, *, or numbers)
    bullet_pattern = r'(?:^|\n)\s*(?:[-•*]|\d+\.)\s+(.+?)(?=\n|$)'
    bullets = re.findall(bullet_pattern, text, re.MULTILINE)

    if not bullets:
        return {
            'count': 0,
            'avg_length': 0,
            'with_action_verbs': 0,
            'with_quantification': 0
        }

    # Analyze each bullet
    action_verb_count = 0
    quantified_count = 0
    lengths = []

    for bullet in bullets:
        lengths.append(len(bullet.split()))

        # Check for action verbs
        for verbs in ACTION_VERBS.values():
            if any(re.search(r'\b' + verb + r'\b', bullet.lower()) for verb in verbs):
                action_verb_count += 1
                break

        # Check for quantification
        if any(re.search(pattern, bullet) for pattern in QUANTIFICATION_PATTERNS):
            quantified_count += 1

    return {
        'count': len(bullets),
        'avg_length': sum(lengths) / len(lengths) if lengths else 0,
        'with_action_verbs': action_verb_count,
        'with_quantification': quantified_count,
        'action_verb_percentage': (action_verb_count / len(bullets)) * 100 if bullets else 0,
        'quantification_percentage': (quantified_count / len(bullets)) * 100 if bullets else 0
    }


def get_content_quality_score(text: str) -> Dict[str, any]:

    # Calculate individual metrics
    action_verbs = count_action_verbs(text)
    weak_verbs = detect_weak_verbs(text)
    quantification_score = calculate_quantification_score(text)
    readability = calculate_readability_score(text)
    passive_voice = len(detect_passive_voice(text))
    first_person = detect_first_person(text)
    bullet_analysis = analyze_bullet_points(text)

    # Calculate component scores
    action_verb_score = min((action_verbs['total'] / 10) * 100, 100)  # 10+ action verbs = 100
    weak_verb_penalty = min(len(weak_verbs) * 5, 30)  # Max 30 point penalty
    passive_penalty = min(passive_voice * 3, 20)  # Max 20 point penalty
    first_person_penalty = min(first_person * 5, 20)  # Max 20 point penalty

    # Overall score (weighted average)
    overall_score = (
                            action_verb_score * 0.25 +
                            quantification_score * 0.25 +
                            readability * 0.20 +
                            (bullet_analysis['action_verb_percentage'] if bullet_analysis['count'] > 0 else 50) * 0.15 +
                            (bullet_analysis['quantification_percentage'] if bullet_analysis[
                                                                                 'count'] > 0 else 50) * 0.15
                    ) - weak_verb_penalty - passive_penalty - first_person_penalty

    overall_score = max(0, min(100, overall_score))

    return {
        'overall_score': round(overall_score, 2),
        'action_verbs': action_verbs,
        'weak_verbs_count': len(weak_verbs),
        'quantification_score': round(quantification_score, 2),
        'readability_score': round(readability, 2),
        'passive_voice_count': passive_voice,
        'first_person_count': first_person,
        'bullet_points': bullet_analysis
    }


class TextAnalyzer:

    def __init__(self):
        pass

    def analyze(self, text: str) -> Dict[str, any]:
        return get_content_quality_score(text)

    def get_suggestions(self, text: str) -> List[str]:

        suggestions = []
        quality = get_content_quality_score(text)

        if quality['action_verbs']['total'] < 5:
            suggestions.append("Add more strong action verbs to describe your achievements")

        if quality['weak_verbs_count'] > 3:
            suggestions.append("Replace weak phrases like 'responsible for' with strong action verbs")

        if quality['quantification_score'] < 30:
            suggestions.append("Add numbers and metrics to quantify your achievements (e.g., 'Increased sales by 25%')")

        if quality['passive_voice_count'] > 5:
            suggestions.append("Use active voice instead of passive voice for stronger impact")

        if quality['first_person_count'] > 0:
            suggestions.append("Remove first-person pronouns (I, me, my) - use action verbs directly")

        if quality['bullet_points']['count'] > 0:
            if quality['bullet_points']['action_verb_percentage'] < 70:
                suggestions.append("Start more bullet points with action verbs")

            if quality['bullet_points']['quantification_percentage'] < 40:
                suggestions.append("Quantify more of your achievements with specific numbers")

        return suggestions