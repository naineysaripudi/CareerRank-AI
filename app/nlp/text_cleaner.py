"""Small, deterministic text-cleaning helpers."""

import re


def clean_text(text: str) -> str:
    """Normalize extracted text while preserving useful punctuation."""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
