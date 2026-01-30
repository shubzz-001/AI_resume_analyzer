import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import spacy

logger = logging.getLogger(__name__)

# Regex patterns
PATTERNS = {
    'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'phone': r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
    'linkedin': r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?',
    'github': r'(?:https?://)?(?:www\.)?github\.com/[\w-]+/?',
    'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    'date': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b',
    'years_exp': r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)',
}


def extract_emails(text: str) -> List[str]:

    if not text:
        return []

    emails = re.findall(PATTERNS['email'], text)
    # Clean and deduplicate
    emails = list(set(email.lower() for email in emails))

    logger.debug(f"Found {len(emails)} email(s)")
    return emails


def extract_phones(text: str) -> List[str]:

    if not text:
        return []

    phones = re.findall(PATTERNS['phone'], text)
    # Clean and deduplicate
    phones = list(set(phone.strip() for phone in phones))

    # Filter out numbers that are too short or clearly not phone numbers
    phones = [p for p in phones if len(re.sub(r'\D', '', p)) >= 10]

    logger.debug(f"Found {len(phones)} phone number(s)")
    return phones


def extract_linkedin(text: str) -> Optional[str]:

    if not text:
        return None

    matches = re.findall(PATTERNS['linkedin'], text, re.IGNORECASE)

    if matches:
        # Return first match, ensure it has https
        url = matches[0]
        if not url.startswith('http'):
            url = 'https://' + url
        return url

    return None


def extract_github(text: str) -> Optional[str]:

    if not text:
        return None

    matches = re.findall(PATTERNS['github'], text, re.IGNORECASE)

    if matches:
        # Return first match, ensure it has https
        url = matches[0]
        if not url.startswith('http'):
            url = 'https://' + url
        return url

    return None


def extract_urls(text: str) -> List[str]:

    if not text:
        return []

    urls = re.findall(PATTERNS['url'], text)
    return list(set(urls))


def extract_name(text: str) -> Optional[str]:

    if not text:
        return None

    lines = text.strip().split('\n')

    # Look in first 5 lines
    for line in lines[:5]:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip lines with email/phone
        if '@' in line or re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', line):
            continue

        # Check if line looks like a name
        words = line.split()

        # Name should be 2-4 words
        if 2 <= len(words) <= 4:
            # All words should start with capital letter
            if all(w[0].isupper() for w in words if w):
                # Should be mostly alphabetic
                if all(w.replace('-', '').replace("'", '').isalpha() for w in words):
                    logger.debug(f"Extracted name: {line}")
                    return line.title()

    # Try with spaCy NER
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:500])  # Only first 500 chars

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                logger.debug(f"Extracted name via NER: {ent.text}")
                return ent.text
    except Exception as e:
        logger.debug(f"Could not use spaCy for name extraction: {str(e)}")

    return None


def extract_dates(text: str) -> List[str]:

    if not text:
        return []

    dates = re.findall(PATTERNS['date'], text, re.IGNORECASE)
    return list(set(dates))


def extract_years_of_experience(text: str) -> Optional[int]:

    if not text:
        return None

    text_lower = text.lower()
    matches = re.findall(PATTERNS['years_exp'], text_lower, re.IGNORECASE)

    if matches:
        # Return the maximum years found
        years = [int(m) for m in matches]
        return max(years)

    return None


def extract_education_degree(text: str) -> List[str]:

    degrees = []
    text_lower = text.lower()

    degree_patterns = [
        r'\b(?:bachelor|b\.?s\.?|b\.?a\.?|b\.?tech|b\.?e\.?)\s+(?:of|in)?\s+[\w\s]+',
        r'\b(?:master|m\.?s\.?|m\.?a\.?|m\.?tech|m\.?e\.?|mba)\s+(?:of|in)?\s+[\w\s]+',
        r'\b(?:phd|ph\.?d\.?|doctorate)\s+(?:of|in)?\s+[\w\s]+',
        r'\b(?:associate|a\.?s\.?|a\.?a\.?)\s+(?:of|in)?\s+[\w\s]+'
    ]

    for pattern in degree_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        degrees.extend(matches)

    # Clean up
    degrees = [d.strip() for d in degrees if d.strip()]
    return list(set(degrees))


def extract_companies(text: str) -> List[str]:

    companies = []

    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ == "ORG":
                companies.append(ent.text)

        logger.debug(f"Found {len(companies)} companies")

    except Exception as e:
        logger.warning(f"Could not extract companies: {str(e)}")

    return list(set(companies))


def extract_locations(text: str) -> List[str]:

    locations = []

    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                locations.append(ent.text)

        logger.debug(f"Found {len(locations)} locations")

    except Exception as e:
        logger.warning(f"Could not extract locations: {str(e)}")

    return list(set(locations))


def extract_all_entities(text: str) -> Dict[str, any]:

    entities = {
        'name': extract_name(text),
        'emails': extract_emails(text),
        'phones': extract_phones(text),
        'linkedin': extract_linkedin(text),
        'github': extract_github(text),
        'urls': extract_urls(text),
        'dates': extract_dates(text),
        'years_experience': extract_years_of_experience(text),
        'degrees': extract_education_degree(text),
        'companies': extract_companies(text),
        'locations': extract_locations(text)
    }

    return entities


class EntityExtractor:

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not available for entity extraction")
            self.nlp = None

    def extract_contact_info(self, text: str) -> Dict[str, any]:

        return {
            'name': extract_name(text),
            'email': extract_emails(text)[0] if extract_emails(text) else None,
            'phone': extract_phones(text)[0] if extract_phones(text) else None,
            'linkedin': extract_linkedin(text),
            'github': extract_github(text)
        }

    def extract_profile(self, text: str) -> Dict[str, any]:

        return extract_all_entities(text)

    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:

        if not self.nlp:
            return {}

        doc = self.nlp(text)

        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)

        # Deduplicate
        entities = {k: list(set(v)) for k, v in entities.items()}

        return entities