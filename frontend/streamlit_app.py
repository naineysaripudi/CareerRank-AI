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
left, right = st.columns(2)
with left:
    target_role = st.text_input("Target role", "AI Engineer")
    location = st.text_input("Preferred location", "Remote")
with right:
    experience_level = st.selectbox("Experience level", ["", "Entry", "Mid", "Senior"])
    employment_type = st.selectbox("Employment type", ["", "Full-time", "Contract", "Hybrid"])
additional_skills = st.text_input("Additional skills", placeholder="Docker, AWS")

if st.button("Find my best jobs", type="primary", disabled=not resume_text.strip()):
    with st.spinner("Extracting, retrieving, and ranking..."):
        results = run_demo_recommendation({"resume_text": resume_text, "target_role": target_role, "location": location, "experience_level": experience_level, "employment_type": employment_type, "additional_skills": [skill.strip() for skill in additional_skills.split(",") if skill.strip()]})
    st.subheader("Recommendation summary")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Jobs analyzed", 100)
    summary_columns[1].metric("Jobs retrieved", 20)
    summary_columns[2].metric("Top recommendations", len(results))
    summary_columns[3].metric("Average match", f"{sum(job['final_score'] for job in results) / max(1, len(results)):.0f}%")
    result_frame = pd.DataFrame(results)
    minimum_score = st.slider("Minimum match score", 0, 100, 0)
    filtered_results = [job for job in results if job["final_score"] >= minimum_score]
    st.subheader(f"Top {len(filtered_results)} recommendations")
    for job in filtered_results:
        with st.expander(f"#{job['rank']} {job['job_title']} at {job['company']} - {job['final_score']:.0f}% match"):
            st.write(f"{job['location']} | {job['experience_level']} | {job['employment_type']}")
            st.write("**Matching skills:** " + (", ".join(job["matching_skills"]) or "None detected"))
            st.write("**Missing skills:** " + (", ".join(job["missing_skills"]) or "None detected"))
            st.write(job["explanation"]["summary"])
            st.write(job["explanation"]["confidence_note"])
    if not result_frame.empty:
        chart_columns = st.columns(2)
        with chart_columns[0]:
            st.plotly_chart(px.bar(result_frame, x="job_title", y="final_score", title="Match score by role"), width="stretch")
        with chart_columns[1]:
            missing = result_frame.explode("missing_skills").dropna(subset=["missing_skills"])
            counts = missing["missing_skills"].value_counts().rename_axis("skill").reset_index(name="count")
            st.plotly_chart(px.bar(counts, x="skill", y="count", title="Missing skill frequency"), width="stretch")
