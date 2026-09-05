"""PDF resume extraction with safe empty-field fallback behavior."""

from pathlib import Path

import fitz

from app.nlp.text_cleaner import clean_text


class ResumeProcessingError(Exception):
    """Raised when a resume cannot be opened or extracted."""


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract text from all PDF pages without logging personal content."""

    try:
        with fitz.open(file_path) as document:
            text = "\n".join(page.get_text() for page in document)
    except (OSError, RuntimeError, fitz.FileDataError) as error:
        raise ResumeProcessingError("The uploaded PDF could not be read.") from error
    cleaned = clean_text(text)
    if not cleaned:
        raise ResumeProcessingError("The uploaded PDF did not contain readable text.")
    return cleaned


def extract_text_from_pdf_bytes(content: bytes) -> str:
    """Extract PDF text from uploaded bytes without creating a temp file."""

    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            text = "\n".join(page.get_text() for page in document)
    except (OSError, RuntimeError, fitz.FileDataError) as error:
        raise ResumeProcessingError("The uploaded PDF could not be read.") from error
    cleaned = clean_text(text)
    if not cleaned:
        raise ResumeProcessingError("The uploaded PDF did not contain readable text.")
    return cleaned
