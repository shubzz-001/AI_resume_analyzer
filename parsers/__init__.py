from pathlib import Path
from typing import Union, Optional
import logging

from parsers.pdf_parser import PDFParser
from parsers.docx_parser import DOCXParser
from parsers.txt_parser import TXTParser
from config import SUPPORTED_RESUME_FORMATS, MAX_FILE_SIZE_MB

logger = logging.getLogger(__name__)


class ResumeParser:

    def __init__(self):
        self.parsers = {
            '.pdf': PDFParser(),
            '.docx': DOCXParser(),
            '.txt': TXTParser()
        }

    def parse(self, file_path: Union[str, Path]) -> str:

        file_path = Path(file_path)

        # Validate file
        self._validate_file(file_path)

        # Get appropriate parser
        extension = file_path.suffix.lower()
        parser = self.parsers.get(extension)

        if not parser:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {', '.join(SUPPORTED_RESUME_FORMATS)}"
            )

        # Parse document
        try:
            logger.info(f"Parsing {file_path.name} using {parser.__class__.__name__}")
            text = parser.parse(file_path)
            logger.info(f"Successfully extracted {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}: {str(e)}")
            raise

    def parse_from_bytes(
            self,
            file_bytes: bytes,
            filename: str,
            file_extension: Optional[str] = None
    ) -> str:

        # Determine extension
        if not file_extension:
            file_extension = Path(filename).suffix.lower()

        # Validate extension
        if file_extension not in self.parsers:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {', '.join(SUPPORTED_RESUME_FORMATS)}"
            )

        # Get appropriate parser
        parser = self.parsers[file_extension]

        try:
            logger.info(f"Parsing {filename} from bytes using {parser.__class__.__name__}")
            text = parser.parse_from_bytes(file_bytes, filename)
            logger.info(f"Successfully extracted {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Failed to parse {filename}: {str(e)}")
            raise ValueError(f"Failed to parse document: {str(e)}")

    def _validate_file(self, file_path: Path) -> None:

        # Check existence
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        # Check size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File too large: {size_mb:.2f}MB. "
                f"Maximum size: {MAX_FILE_SIZE_MB}MB"
            )

        # Check if empty
        if file_path.stat().st_size == 0:
            raise ValueError("File is empty")

    def get_supported_formats(self) -> list:

        return list(self.parsers.keys())

    def is_supported(self, filename: str) -> bool:

        extension = Path(filename).suffix.lower()
        return extension in self.parsers

    def extract_metadata(self, file_path: Union[str, Path]) -> dict:

        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        parser = self.parsers.get(extension)
        if not parser:
            return {}

        try:
            return parser.extract_metadata(file_path)
        except Exception as e:
            logger.warning(f"Could not extract metadata: {str(e)}")
            return {}


# Convenience function for direct use
def extract_text(file_path: Union[str, Path]) -> str:

    parser = ResumeParser()
    return parser.parse(file_path)


# Convenience function for uploaded files
def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:

    parser = ResumeParser()
    return parser.parse_from_bytes(file_bytes, filename)