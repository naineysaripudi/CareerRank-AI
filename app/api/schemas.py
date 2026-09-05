"""Pydantic request and response models for the public API."""

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    resume_text: str = Field(min_length=1)
    target_role: str = ""
    location: str = ""
    experience_level: str = ""
    employment_type: str = ""
    additional_skills: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
