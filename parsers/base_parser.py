from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    def __init__(self):
        self.supported_extensions = []

    @abstractmethod
    def parse(self, file_path: Path) -> str:

        pass

    def validate_file(self, file_path: Path) -> bool:

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        if not file_path.is_file():
            logger.error(f"Not a file: {file_path}")
            return False

        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            logger.error(f"Unsupported extension: {extension}")
            return False

        return True

    def clean_text(self, text: str) -> str:

        if not text:
            return ""

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove null characters
        text = text.replace('\x00', '')

        return text.strip()

    def get_file_info(self, file_path: Path) -> dict:

        return {
            "name": file_path.name,
            "extension": file_path.suffix,
            "size_bytes": file_path.stat().st_size,
            "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
        }