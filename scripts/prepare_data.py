"""Generate the deterministic development dataset and prepared CSV."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

# Allow the documented direct script command to import the application package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.preprocessing import prepare_jobs
import pandas as pd

OUTPUT_PATH = ROOT / "data" / "sample_jobs.csv"
PREPARED_PATH = ROOT / "data" / "processed" / "prepared_jobs.csv"

ROLE_TEMPLATES = [
    ("AI Engineer", "Python, Machine Learning, Deep Learning, NLP", "PyTorch, Docker", "Build applied AI services and evaluate models for production use."),
    ("Machine Learning Engineer", "Python, Machine Learning, SQL, Scikit-learn", "PyTorch, MLflow, Docker", "Develop training pipelines and deploy reliable machine learning systems."),
    ("Data Scientist", "Python, SQL, Statistics, Pandas", "Scikit-learn, Tableau, Machine Learning", "Analyze product data and build predictive models for business decisions."),
    ("Data Analyst", "SQL, Excel, Power BI, Statistics", "Python, Tableau, Pandas", "Turn operational data into dashboards, insights, and measurable recommendations."),
    ("NLP Engineer", "Python, NLP, Deep Learning, PyTorch", "Transformers, RAG, Docker", "Create language understanding systems for search, classification, and generation."),
    ("Computer Vision Engineer", "Python, Computer Vision, Deep Learning, PyTorch", "OpenCV, TensorFlow, Docker", "Build and optimize vision models for image and video applications."),
    ("Python Developer", "Python, Git, SQL, REST APIs", "FastAPI, Docker, AWS", "Implement maintainable backend services and integrations in Python."),
    ("Software Engineer", "Python, Java, Git, SQL", "Docker, AWS, Kubernetes", "Design, test, and ship scalable software features with a collaborative team."),
    ("Backend Developer", "Python, SQL, REST APIs, Git", "FastAPI, PostgreSQL, Docker", "Build secure APIs and data services for customer-facing products."),
    ("MLOps Engineer", "Python, Docker, Kubernetes, MLflow", "AWS, Terraform, CI/CD", "Automate model delivery, monitoring, and reproducible machine learning workflows."),
    ("Data Engineer", "Python, SQL, Apache Spark, Git", "AWS, Airflow, Docker", "Build dependable batch and streaming pipelines for analytics and ML teams."),
]
LOCATIONS = ["Hyderabad", "Bengaluru", "Pune", "Mumbai", "Remote"]
EXPERIENCE_LEVELS = ["Entry", "Mid", "Senior"]
EMPLOYMENT_TYPES = ["Full-time", "Contract", "Hybrid"]


def build_records(record_count: int = 100) -> list[dict[str, object]]:
    """Create varied but deterministic job records without external job content."""

    records = []
    for index in range(record_count):
        role, required, preferred, description = ROLE_TEMPLATES[index % len(ROLE_TEMPLATES)]
        level = EXPERIENCE_LEVELS[(index // len(ROLE_TEMPLATES)) % len(EXPERIENCE_LEVELS)]
        location = LOCATIONS[index % len(LOCATIONS)]
        employment = EMPLOYMENT_TYPES[index % len(EMPLOYMENT_TYPES)]
        salary_min = 50000 + (index % 10) * 5000 + EXPERIENCE_LEVELS.index(level) * 15000
        records.append(
            {
                "job_id": f"JOB-{index + 1:04d}",
                "job_title": role,
                "company": f"Northstar Labs {index % 10 + 1}",
                "location": location,
                "employment_type": employment,
                "experience_level": level,
                "salary_min": salary_min,
                "salary_max": salary_min + 25000,
                "required_skills": required,
                "preferred_skills": preferred,
                "job_description": f"{description} This {level.lower()} role partners with engineering and product stakeholders.",
                "education": "Bachelor's degree in Computer Science or a related field",
                "source": "synthetic-development-dataset",
            }
        )
    return records


def main() -> None:
    """Write raw and prepared job data to the repository."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPARED_PATH.parent.mkdir(parents=True, exist_ok=True)
    jobs = pd.DataFrame(build_records())
    jobs.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    prepare_jobs(jobs).to_csv(PREPARED_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(jobs)} jobs to {OUTPUT_PATH}")
    print(f"Wrote prepared jobs to {PREPARED_PATH}")


if __name__ == "__main__":
    main()
