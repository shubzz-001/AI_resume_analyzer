import pdfplumber
from pathlib import Path
from typing import Optional
from io import BytesIO
import logging

from parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """Parser for PDF documents."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.pdf']

    def parse(self, file_path: Path) -> str:

        if not self.validate_file(file_path):
            raise FileNotFoundError(f"Invalid PDF file: {file_path}")

        try:
            text = self._extract_text_from_pdf(file_path)

            if not text or len(text.strip()) < 10:
                raise ValueError("PDF appears to be empty or contains no extractable text")

            return self.clean_text(text)

        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {str(e)}")
            raise ValueError(f"Failed to parse PDF: {str(e)}")

    def _extract_text_from_pdf(self, file_path: Path) -> str:

        text_parts = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()

                    if page_text:
                        text_parts.append(page_text)
                        logger.debug(f"Extracted {len(page_text)} chars from page {page_num}")
                    else:
                        logger.warning(f"No text found on page {page_num}")

                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {str(e)}")
                    continue

        return "\n".join(text_parts)

    def parse_from_bytes(self, file_bytes: bytes, filename: str = "resume.pdf") -> str:

        try:
            file_stream = BytesIO(file_bytes)
            text_parts = []

            with pdfplumber.open(file_stream) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {filename}: {str(e)}")
                        continue

            text = "\n".join(text_parts)

            if not text or len(text.strip()) < 10:
                raise ValueError("PDF appears to be empty or contains no extractable text")

            return self.clean_text(text)

        except Exception as e:
            logger.error(f"Error parsing PDF bytes from {filename}: {str(e)}")
            raise ValueError(f"Failed to parse PDF: {str(e)}")

    def get_page_count(self, file_path: Path) -> int:

        try:
            with pdfplumber.open(file_path) as pdf:
                return len(pdf.pages)
        except Exception as e:
            logger.error(f"Error getting page count: {str(e)}")
            return 0

    def extract_metadata(self, file_path: Path) -> dict:

        metadata = self.get_file_info(file_path)

        try:
            with pdfplumber.open(file_path) as pdf:
                metadata.update({
                    "page_count": len(pdf.pages),
                    "pdf_metadata": pdf.metadata
                })
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {str(e)}")

        return metadata


# Convenience function for direct use
def extract_text_from_pdf(file_path: Path) -> str:

    parser = PDFParser()
    return parser.parse(file_path)