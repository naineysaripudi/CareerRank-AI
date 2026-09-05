"""Deterministic explanations and an extension point for LLM providers."""


def build_fallback_explanation(job: dict[str, object]) -> dict[str, object]:
    matching = list(job.get("matching_skills", []))
    missing = list(job.get("missing_skills", []))
    summary = f"Recommended with a {job.get('final_score', 0):.0f}% match based on skills and preferences."
    why = f"Strong alignment in {', '.join(matching)}." if matching else "The semantic profile match suggests this role is worth reviewing."
    gap = f"The main skill gaps are {', '.join(missing)}." if missing else "No required skill gaps were detected."
    return {"summary": summary, "why_recommended": [why], "matching_skills": matching, "missing_skills": missing, "skill_gap_priority": missing, "confidence_note": gap}
