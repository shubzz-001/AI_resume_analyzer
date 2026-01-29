from pathlib import Path
from typing import Optional
import logging
import chardet

from parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)


class TXTParser(BaseParser):
    """Parser for plain text documents."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.txt']

    def parse(self, file_path: Path) -> str:

        if not self.validate_file(file_path):
            raise FileNotFoundError(f"Invalid TXT file: {file_path}")

        try:
            text = self._read_text_file(file_path)

            if not text or len(text.strip()) < 10:
                raise ValueError("TXT file appears to be empty")

            return self.clean_text(text)

        except Exception as e:
            logger.error(f"Error parsing TXT {file_path}: {str(e)}")
            raise ValueError(f"Failed to parse TXT: {str(e)}")

    def _read_text_file(self, file_path: Path) -> str:

        # Try common encodings first
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                logger.debug(f"Successfully read with {encoding} encoding")
                return text
            except (UnicodeDecodeError, UnicodeError):
                continue

        # If all fail, try to auto-detection
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()

            detected = chardet.detect(raw_data)
            encoding = detected['encoding']

            logger.info(f"Auto-detected encoding: {encoding}")

            text = raw_data.decode(encoding)
            return text

        except Exception as e:
            logger.error(f"All encoding attempts failed: {str(e)}")
            raise ValueError("Could not determine file encoding")

    def parse_from_bytes(self, file_bytes: bytes, filename: str = "resume.txt") -> str:

        try:
            # Try UTF-8 first
            try:
                text = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # Try auto-detection
                detected = chardet.detect(file_bytes)
                encoding = detected['encoding']
                text = file_bytes.decode(encoding)

            if not text or len(text.strip()) < 10:
                raise ValueError("TXT file appears to be empty")

            return self.clean_text(text)

        except Exception as e:
            logger.error(f"Error parsing TXT bytes from {filename}: {str(e)}")
            raise ValueError(f"Failed to parse TXT: {str(e)}")

    def get_line_count(self, file_path: Path) -> int:

        try:
            text = self._read_text_file(file_path)
            return len(text.split('\n'))
        except Exception as e:
            logger.error(f"Error getting line count: {str(e)}")
            return 0

    def get_word_count(self, file_path: Path) -> int:

        try:
            text = self._read_text_file(file_path)
            return len(text.split())
        except Exception as e:
            logger.error(f"Error getting word count: {str(e)}")
            return 0

    def extract_metadata(self, file_path: Path) -> dict:

        metadata = self.get_file_info(file_path)

        try:
            text = self._read_text_file(file_path)

            metadata.update({
                "line_count": len(text.split('\n')),
                "word_count": len(text.split()),
                "char_count": len(text),
                "encoding": self._detect_encoding(file_path)
            })

        except Exception as e:
            logger.warning(f"Could not extract TXT metadata: {str(e)}")

        return metadata

    def _detect_encoding(self, file_path: Path) -> str:

        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()

            detected = chardet.detect(raw_data)
            return detected['encoding']

        except Exception:
            return "unknown"


# Convenience function for direct use
def extract_text_from_txt(file_path: Path) -> str:

    parser = TXTParser()
    return parser.parse(file_path)