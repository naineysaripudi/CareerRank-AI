"""Load and validate the synthetic or user-provided job dataset."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "job_id",
    "job_title",
    "company",
    "location",
    "employment_type",
    "experience_level",
    "salary_min",
    "salary_max",
    "required_skills",
    "preferred_skills",
    "job_description",
    "education",
    "source",
}


def load_jobs(path: str | Path) -> pd.DataFrame:
    """Read jobs from CSV and fail early when the schema is incomplete."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Job dataset not found: {csv_path}")

    jobs = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS.difference(jobs.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Job dataset is missing required columns: {missing}")

    jobs = jobs.copy()
    jobs["job_id"] = jobs["job_id"].astype(str)
    jobs = jobs.drop_duplicates(subset="job_id").reset_index(drop=True)
    return jobs
