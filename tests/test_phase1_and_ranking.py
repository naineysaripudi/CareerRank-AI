from app.data.loader import load_jobs
from app.data.preprocessing import prepare_jobs
from app.nlp.profile_builder import CandidateProfile
from app.nlp.profile_builder import build_candidate_profile
from app.ranking.ranker import rank_jobs
from app.evaluation.ranking_metrics import ndcg_at_k


def test_sample_dataset_and_preprocessing() -> None:
    jobs = prepare_jobs(load_jobs("data/sample_jobs.csv"))
    assert len(jobs) == 100
    assert jobs["search_text"].str.len().gt(0).all()


def test_personalized_ranker_prefers_aligned_job() -> None:
    profile = CandidateProfile(skills=["Python"], preferred_roles=["AI Engineer"], preferred_locations=["Remote"])
    jobs = [
        {"job_id": "a", "job_title": "AI Engineer", "location": "Remote", "required_skills": "Python, NLP", "preferred_skills": "Docker", "experience_level": "Entry", "employment_type": "Full-time", "semantic_similarity": 1.0},
        {"job_id": "b", "job_title": "Data Analyst", "location": "Pune", "required_skills": "SQL", "preferred_skills": "Tableau", "experience_level": "Senior", "employment_type": "Contract", "semantic_similarity": 0.9},
    ]
    assert rank_jobs(profile, jobs)[0]["job_id"] == "a"


def test_profile_extracts_resume_sections_and_contacts() -> None:
    profile = build_candidate_profile("Alex Kumar\nalex@example.com\n\nEducation\nB.Tech\n\nExperience\nML Engineer")
    assert profile.name == "Alex Kumar"
    assert profile.email == "alex@example.com"
    assert profile.education == ["B.Tech"]
    assert profile.experience == ["ML Engineer"]


def test_ndcg_handles_relevant_results() -> None:
    assert 0 < ndcg_at_k([1, 0, 1], 3) <= 1
