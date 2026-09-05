import fitz
from fastapi.testclient import TestClient

from app.api.routes import app


def test_health_and_job_lookup() -> None:
    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/jobs/JOB-0001").status_code == 200
    assert client.get("/api/jobs/unknown").status_code == 404


def test_pdf_upload_extracts_text() -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Alex Kumar\nPython and SQL")
    payload = document.tobytes()
    document.close()
    response = TestClient(app).post(
        "/api/resume/upload",
        files={"file": ("resume.pdf", payload, "application/pdf")},
    )
    assert response.status_code == 200
    assert "Alex Kumar" in response.json()["text"]


def test_pdf_upload_rejects_unreadable_pdf() -> None:
    response = TestClient(app).post(
        "/api/resume/upload",
        files={"file": ("resume.pdf", b"not a PDF", "application/pdf")},
    )
    assert response.status_code == 400