import spacy
import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Load spaCy model (singleton pattern)
_nlp_model = None


def get_nlp_model():

    global _nlp_model

    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise RuntimeError("spaCy model 'en_core_web_sm' not found. Please install it.")

    return _nlp_model


def clean_text(text: str, lowercase: bool = True, remove_stopwords: bool = False) -> str:

    if not text or not isinstance(text, str):
        logger.warning("Empty or invalid text provided")
        return ""

    try:
        # Basic cleaning first
        text = _basic_clean(text)

        # Process with spaCy
        nlp = get_nlp_model()
        doc = nlp(text)

        # Extract tokens based on options
        tokens = []
        for token in doc:
            # Skip if not alphabetic
            if not token.is_alpha:
                continue

            # Skip stop words if requested
            if remove_stopwords and token.is_stop:
                continue

            # Use lemma (base form)
            word = token.lemma_

            # Apply lowercase if requested
            if lowercase:
                word = word.lower()

            tokens.append(word)

        cleaned = " ".join(tokens)
        logger.debug(f"Cleaned text: {len(text)} -> {len(cleaned)} characters")

        return cleaned

    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        # Fallback to basic cleaning
        return _basic_clean(text)


def _basic_clean(text: str) -> str:

    if not text:
        return ""

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Remove email addresses (preserve for extraction)
    # text = re.sub(r'\S+@\S+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s\-\+\#\.\@]', ' ', text)

    # Remove digits only if standalone
    text = re.sub(r'\b\d+\b', '', text)

    return text.strip()


def clean_for_display(text: str) -> str:

    if not text:
        return ""

    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove multiple punctuation
    text = re.sub(r'([.!?]){2,}', r'\1', text)

    return text.strip()


def normalize_text(text: str) -> str:

    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    return text.strip()


def remove_noise(text: str) -> str:

    noise_patterns = [
        r'page \d+ of \d+',
        r'references available upon request',
        r'curriculum vitae',
        r'resume of',
        r'confidential',
        r'\bcv\b'
    ]

    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text.strip()


def extract_sentences(text: str) -> list:

    if not text:
        return []

    try:
        nlp = get_nlp_model()
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception as e:
        logger.error(f"Error extracting sentences: {str(e)}")
        # Fallback to simple split
        return [s.strip() for s in text.split('.') if s.strip()]


def get_token_count(text: str) -> int:

    if not text:
        return 0

    try:
        nlp = get_nlp_model()
        doc = nlp(text)
        return len([token for token in doc if token.is_alpha])
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return len(text.split())


def get_word_frequency(text: str, top_n: int = 20) -> dict:

    if not text:
        return {}

    try:
        nlp = get_nlp_model()
        doc = nlp(text.lower())

        # Count words (lemmas)
        word_freq = {}
        for token in doc:
            if token.is_alpha and not token.is_stop and len(token.text) > 2:
                lemma = token.lemma_
                word_freq[lemma] = word_freq.get(lemma, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        return dict(sorted_words[:top_n])

    except Exception as e:
        logger.error(f"Error getting word frequency: {str(e)}")
        return {}


class TextCleaner:
    """
    Class-based text cleaner.
    """

    def __init__(self, lowercase: bool = True, remove_stopwords: bool = False):

        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.nlp = get_nlp_model()

    def clean(self, text: str) -> str:

        return clean_text(text, self.lowercase, self.remove_stopwords)

    def clean_batch(self, texts: list) -> list:

        return [self.clean(text) for text in texts]


# Convenience function for backward compatibility
def clean_resume_text(text: str) -> str:

    return clean_text(text, lowercase=True, remove_stopwords=False)