"""Skill-gap analysis based on controlled required/preferred skills."""

from app.nlp.skill_extractor import extract_skills


def analyze_skill_gap(candidate_skills: list[str], job: dict[str, object]) -> dict[str, list[str]]:
    candidate = {skill.casefold() for skill in candidate_skills}
    required = extract_skills(str(job.get("required_skills", "")))
    preferred = extract_skills(str(job.get("preferred_skills", "")))
    required_missing = [skill for skill in required if skill.casefold() not in candidate]
    preferred_missing = [skill for skill in preferred if skill.casefold() not in candidate]
    return {
        "strong_skills": [skill for skill in required + preferred if skill.casefold() in candidate],
        "missing_required_skills": required_missing,
        "missing_preferred_skills": preferred_missing,
        "high_priority_skills": required_missing,
    }
