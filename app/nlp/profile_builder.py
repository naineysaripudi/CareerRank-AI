"""Candidate profile construction from resume text and preferences."""

from dataclasses import dataclass, field
import re

from app.nlp.skill_extractor import extract_skills


@dataclass
class CandidateProfile:
    """Normalized candidate information used by recommendation services."""

    name: str = ""
    email: str = ""
    phone: str = ""
    education: list[str] = field(default_factory=list)
    experience: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    preferred_roles: list[str] = field(default_factory=list)
    preferred_locations: list[str] = field(default_factory=list)
    experience_level: str = ""
    employment_type: str = ""
    raw_text: str = ""

    def to_embedding_text(self) -> str:
        """Return a stable, human-readable representation for embedding."""

        return "\n".join(
            [
                f"Candidate Profile: {self.name}" if self.name else "Candidate Profile",
                f"Skills: {', '.join(self.skills)}",
                f"Experience: {' '.join(self.experience)}",
                f"Education: {' '.join(self.education)}",
                f"Target Role: {', '.join(self.preferred_roles)}",
                f"Location Preference: {', '.join(self.preferred_locations)}",
                f"Experience Level: {self.experience_level}",
            ]
        )


def build_candidate_profile(resume_text: str, **preferences: object) -> CandidateProfile:
    """Build a profile using extracted skills plus explicit user preferences."""

    sections = _extract_sections(resume_text)
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    email_match = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", resume_text)
    phone_match = re.search(r"(?:\+?\d[\d\s().-]{7,}\d)", resume_text)
    name = lines[0] if lines and "@" not in lines[0] else ""

    profile = CandidateProfile(
        name=name,
        email=email_match.group(0) if email_match else "",
        phone=phone_match.group(0).strip() if phone_match else "",
        education=sections.get("education", []),
        experience=sections.get("experience", []),
        skills=extract_skills(resume_text),
        projects=sections.get("projects", []),
        certifications=sections.get("certifications", []),
        raw_text=resume_text,
        preferred_roles=_as_list(preferences.get("preferred_roles", preferences.get("target_role", []))),
        preferred_locations=_as_list(preferences.get("preferred_locations", preferences.get("location", []))),
        experience_level=str(preferences.get("experience_level", "") or ""),
        employment_type=str(preferences.get("employment_type", "") or ""),
    )
    return profile


def _extract_sections(text: str) -> dict[str, list[str]]:
    """Collect short lines under common resume headings without over-parsing."""

    heading_names = {"education", "experience", "work experience", "projects", "certifications"}
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        normalized = re.sub(r"[^a-z ]", "", line.casefold()).strip()
        if normalized in heading_names:
            current = "experience" if normalized == "work experience" else normalized
            sections.setdefault(current, [])
        elif current and line:
            sections[current].append(line)
    return sections


def _as_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []
