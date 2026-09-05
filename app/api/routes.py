"""FastAPI application for CareerRank AI."""

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas import HealthResponse, RecommendationRequest
from app.core.config import get_settings
from app.nlp.resume_parser import ResumeProcessingError, extract_text_from_pdf_bytes
from app.services.pipeline import run_demo_recommendation

app = FastAPI(title="CareerRank AI", version="0.1.0")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/api/resume/upload")
async def upload_resume(file: UploadFile) -> JSONResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF resumes are supported.")
    content = await file.read()
    if len(content) > get_settings().max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Resume file is too large.")
    try:
        return JSONResponse({"text": extract_text_from_pdf_bytes(content)})
    except ResumeProcessingError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    from app.data.loader import load_jobs
    jobs = load_jobs(get_settings().data_path)
    matches = jobs[jobs["job_id"] == job_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Job not found.")
    return matches.iloc[0].to_dict()


@app.post("/api/recommend")
def recommend_jobs(request: RecommendationRequest) -> dict[str, object]:
    try:
        results = run_demo_recommendation(request.model_dump(), get_settings().data_path)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"recommendations": results}


@app.post("/api/skill-gap")
def skill_gap(request: RecommendationRequest) -> dict[str, object]:
    results = recommend_jobs(request)["recommendations"]
    return {"skill_gaps": [item["skill_gap"] for item in results]}
