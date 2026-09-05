"""Candidate-job matching features."""

from app.nlp.profile_builder import CandidateProfile
from app.nlp.skill_extractor import extract_skills


def calculate_features(profile: CandidateProfile, job: dict[str, object], semantic_similarity: float) -> dict[str, float | list[str]]:
    candidate_skills = {skill.casefold() for skill in profile.skills}
    required = extract_skills(str(job.get("required_skills", "")))
    preferred = extract_skills(str(job.get("preferred_skills", "")))
    required_set = {skill.casefold() for skill in required}
    preferred_set = {skill.casefold() for skill in preferred}
    matching = sorted(candidate_skills & (required_set | preferred_set))
    missing = sorted(required_set - candidate_skills)
    role_text = str(job.get("job_title", "")).casefold()
    role_match = float(any(role.casefold() in role_text or role_text in role.casefold() for role in profile.preferred_roles))
    location_match = float(not profile.preferred_locations or str(job.get("location", "")).casefold() in {item.casefold() for item in profile.preferred_locations})
    experience_match = float(not profile.experience_level or str(job.get("experience_level", "")).casefold() == profile.experience_level.casefold())
    employment_match = float(not profile.employment_type or str(job.get("employment_type", "")).casefold() == profile.employment_type.casefold())
    return {
        "semantic_similarity": max(0.0, min(1.0, (semantic_similarity + 1) / 2)),
        "skill_match": len(matching) / max(1, len(required_set | preferred_set)),
        "required_skill_match_ratio": len(candidate_skills & required_set) / max(1, len(required_set)),
        "preferred_skill_match_ratio": len(candidate_skills & preferred_set) / max(1, len(preferred_set)),
        "experience_match": experience_match,
        "role_match": role_match,
        "location_match": location_match,
        "employment_match": employment_match,
        "matching_skills": matching,
        "missing_skills": missing,
    }
