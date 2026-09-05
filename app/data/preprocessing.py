"""Reusable preprocessing helpers for job text and structured fields."""

import re

import pandas as pd


def normalize_text(value: object) -> str:
    """Collapse whitespace and return a predictable lowercase text value."""

    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip().lower()
    return text


def parse_skill_list(value: object) -> list[str]:
    """Convert comma or semicolon separated skills into normalized values."""

    text = normalize_text(value)
    if not text:
        return []
    skills = re.split(r"[,;|]", text)
    return list(dict.fromkeys(skill.strip() for skill in skills if skill.strip()))


def build_job_search_text(job: pd.Series) -> str:
    """Build the text representation that will later be embedded for retrieval."""

    fields = [
        job.get("job_title", ""),
        job.get("required_skills", ""),
        job.get("preferred_skills", ""),
        job.get("job_description", ""),
        job.get("location", ""),
        job.get("experience_level", ""),
    ]
    return " ".join(normalize_text(field) for field in fields if normalize_text(field))


def prepare_jobs(jobs: pd.DataFrame) -> pd.DataFrame:
    """Clean job records and add fields used by later pipeline stages."""

    prepared = jobs.copy()
    text_columns = ["job_title", "company", "location", "employment_type", "experience_level", "education", "source"]
    for column in text_columns:
        prepared[column] = prepared[column].fillna("").map(normalize_text)

    for column in ["required_skills", "preferred_skills", "job_description"]:
        prepared[column] = prepared[column].fillna("").astype(str).str.strip()

    prepared["salary_min"] = pd.to_numeric(prepared["salary_min"], errors="coerce").fillna(0)
    prepared["salary_max"] = pd.to_numeric(prepared["salary_max"], errors="coerce").fillna(0)
    prepared["search_text"] = prepared.apply(build_job_search_text, axis=1)
    return prepared
