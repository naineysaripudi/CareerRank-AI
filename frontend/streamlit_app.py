"""Streamlit demo interface for CareerRank AI."""

import sys
from pathlib import Path

# Streamlit Cloud runs this file from frontend/, so make the repository root
# importable before loading the application package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

import pandas as pd
import plotly.express as px

from app.core.config import get_settings
from app.data.loader import load_jobs
from app.nlp.resume_parser import extract_text_from_pdf_bytes
from app.services.pipeline import run_demo_recommendation

st.set_page_config(page_title="CareerRank AI", page_icon="CR", layout="wide")
st.title("CareerRank AI")
st.caption("Explainable, personalized job discovery")

uploaded_file = st.file_uploader("Upload resume PDF", type=["pdf"])
resume_text = ""
if uploaded_file:
    try:
        resume_text = extract_text_from_pdf_bytes(uploaded_file.getvalue())
    except Exception as error:
        st.error(str(error))
resume_text = st.text_area("Resume text", value=resume_text, height=220, placeholder="Upload a PDF or paste resume text...")
job_catalog = load_jobs(get_settings().data_path)
role_options = ["Any Role", *sorted(job_catalog["job_title"].dropna().unique().tolist())]
location_options = ["Any Location", *sorted(job_catalog["location"].dropna().unique().tolist())]
left, right = st.columns(2)
with left:
    target_role = st.selectbox("Target role", role_options, index=role_options.index("AI Engineer"))
    location = st.selectbox("Preferred location", location_options, index=location_options.index("Remote"))
with right:
    experience_level = st.selectbox("Experience level", ["", "Entry", "Mid", "Senior"])
    employment_type = st.selectbox("Employment type", ["", "Full-time", "Contract", "Hybrid"])
additional_skills = st.text_input("Additional skills", placeholder="Docker, AWS")

if st.button("Find my best jobs", type="primary", disabled=not resume_text.strip()):
    with st.spinner("Extracting, retrieving, and ranking..."):
        st.session_state.recommendation_results = run_demo_recommendation({
            "resume_text": resume_text,
            "target_role": "" if target_role == "Any Role" else target_role,
            "location": "" if location == "Any Location" else location,
            "experience_level": experience_level,
            "employment_type": employment_type,
            "additional_skills": [skill.strip() for skill in additional_skills.split(",") if skill.strip()],
        })

if "recommendation_results" in st.session_state:
    results = st.session_state.recommendation_results
    minimum_score = st.slider("Minimum match score", 0, 100, 0)
    filtered_results = [job for job in results if job["final_score"] >= minimum_score]
    st.subheader("Recommendation summary")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Jobs analyzed", len(job_catalog))
    summary_columns[1].metric("Jobs retrieved", get_settings().top_k_retrieval)
    summary_columns[2].metric("Top recommendations", len(filtered_results))
    summary_columns[3].metric("Average match", f"{sum(job['final_score'] for job in filtered_results) / max(1, len(filtered_results)):.0f}%")
    st.subheader(f"Top {len(filtered_results)} recommendations")
    for job in filtered_results:
        with st.expander(f"#{job['rank']} {job['job_title']} at {job['company']} - {job['final_score']:.0f}% match"):
            st.write(f"{job['location']} | {job['experience_level']} | {job['employment_type']}")
            st.write("**Matching skills:** " + (", ".join(job["matching_skills"]) or "None detected"))
            st.write("**Missing skills:** " + (", ".join(job["missing_skills"]) or "None detected"))
            st.write(job["explanation"]["summary"])
            st.write(job["explanation"]["confidence_note"])
            result_frame = pd.DataFrame(filtered_results)
    if not result_frame.empty:
        chart_columns = st.columns(2)
        with chart_columns[0]:
            st.plotly_chart(px.bar(result_frame, x="job_title", y="final_score", title="Match score by role"), width="stretch")
        with chart_columns[1]:
            missing = result_frame.explode("missing_skills").dropna(subset=["missing_skills"])
            counts = missing["missing_skills"].value_counts().rename_axis("skill").reset_index(name="count")
            st.plotly_chart(px.bar(counts, x="skill", y="count", title="Missing skill frequency"), width="stretch")
