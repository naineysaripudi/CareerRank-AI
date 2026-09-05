"""Controlled-vocabulary skill extraction with alias normalization."""

import re

from app.nlp.skill_taxonomy import SKILL_ALIASES, SKILL_TERMS


def extract_skills(text: str) -> list[str]:
    """Find known skills without inventing skills outside the taxonomy."""

    normalized_text = text.casefold()
    found: dict[str, str] = {}
    for term in SKILL_TERMS:
        pattern = rf"(?<![a-z0-9+#]){re.escape(term)}(?![a-z0-9+#])"
        if re.search(pattern, normalized_text):
            canonical = SKILL_ALIASES[term]
            found[canonical.casefold()] = canonical
    return sorted(found.values())
