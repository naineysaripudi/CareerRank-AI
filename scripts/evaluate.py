"""Run the reproducible synthetic ranking evaluation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.loader import load_jobs
from app.data.preprocessing import prepare_jobs
from app.evaluation.evaluation_runner import run_evaluation


if __name__ == "__main__":
    jobs = prepare_jobs(load_jobs(ROOT / "data" / "sample_jobs.csv"))
    print(run_evaluation(jobs))
