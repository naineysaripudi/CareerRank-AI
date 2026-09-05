"""Streamlit demo interface for CareerRank AI."""

import sys
import html
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
from app.ranking.scoring import RankingConfig
from app.services.pipeline import run_demo_recommendation

st.set_page_config(page_title="CareerRank AI", page_icon="CR", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --cr-bg: #0b1018;
        --cr-panel: #121a24;
        --cr-panel-soft: #172230;
        --cr-border: #263547;
        --cr-text: #f3f7fb;
        --cr-muted: #93a4b8;
        --cr-coral: #f47d63;
        --cr-teal: #38c8b2;
        --cr-yellow: #f3c969;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--cr-bg); color: var(--cr-text); }
    .block-container { max-width: 1180px; padding: 3.5rem 2.5rem 2rem; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    h2 { margin-top: 2rem; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stFileUploaderDropzone"] { background: #101923; border: 1px dashed #38506a; border-radius: 12px; }
    [data-testid="stFileUploaderDropzoneInstructions"] { color: var(--cr-muted); }
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input,
    [data-baseweb="select"] > div { background: #151f2b; border-color: var(--cr-border); border-radius: 9px; }
    [data-testid="stTextArea"] textarea:focus, [data-testid="stTextInput"] input:focus { border-color: var(--cr-teal); }
    [data-testid="stSlider"] [role="slider"] { background: var(--cr-coral); }
    .stButton > button[kind="primary"] { background: var(--cr-coral); border: 0; border-radius: 9px; color: #1b1110; font-weight: 700; min-height: 2.8rem; }
    .stButton > button[kind="primary"]:hover { background: #ff927a; color: #1b1110; }
    [data-testid="stExpander"] { background: #121a24; border: 1px solid var(--cr-border); border-radius: 10px; }
    [data-testid="stMetric"] { background: var(--cr-panel); border: 1px solid var(--cr-border); border-radius: 10px; padding: 1rem 1.1rem; }
    [data-testid="stMetricLabel"] { color: var(--cr-muted); }
    [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
    .cr-hero { align-items: flex-end; border-bottom: 1px solid var(--cr-border); display: flex; justify-content: space-between; margin-bottom: 2.2rem; padding-bottom: 1.8rem; }
    .cr-eyebrow { color: var(--cr-teal); font-size: .72rem; font-weight: 700; letter-spacing: .16em; margin-bottom: .55rem; }
    .cr-title { color: var(--cr-text); font-family: 'Space Grotesk', sans-serif; font-size: clamp(2.4rem, 5vw, 4.2rem); font-weight: 700; line-height: .98; }
    .cr-subtitle { color: var(--cr-muted); font-size: 1rem; margin-top: .8rem; }
    .cr-signal { background: #102a2b; border: 1px solid #1e5e5a; border-radius: 999px; color: #8ce4d4; font-size: .8rem; padding: .55rem .8rem; }
    .cr-section { color: var(--cr-text); font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 600; margin: 2rem 0 1rem; }
    .cr-section-note { color: var(--cr-muted); font-size: .86rem; margin-top: -.65rem; margin-bottom: 1rem; }
    .cr-panel { background: var(--cr-panel); border: 1px solid var(--cr-border); border-radius: 12px; padding: 1.25rem; }
    .cr-panel-title { color: var(--cr-text); font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600; margin-bottom: .25rem; }
    .cr-panel-note { color: var(--cr-muted); font-size: .82rem; }
    .cr-job-head { align-items: center; display: flex; gap: 1rem; justify-content: space-between; padding: .25rem 0 .6rem; }
    .cr-job-rank { color: var(--cr-coral); font-size: .75rem; font-weight: 700; letter-spacing: .1em; }
    .cr-job-title { color: var(--cr-text); font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 600; }
    .cr-company { color: var(--cr-muted); font-size: .88rem; margin-top: .15rem; }
    .cr-score { color: var(--cr-teal); font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; white-space: nowrap; }
    .cr-meta { color: var(--cr-muted); display: flex; flex-wrap: wrap; font-size: .82rem; gap: .5rem 1rem; padding: .4rem 0 .65rem; }
    .cr-meta span::before { color: var(--cr-teal); content: '•'; margin-right: .35rem; }
    .cr-chip { background: #1c2b38; border: 1px solid #2c4153; border-radius: 999px; color: #c9d6e3; display: inline-block; font-size: .75rem; margin: .18rem .25rem .18rem 0; padding: .28rem .55rem; }
    .cr-chip.good { background: #12302d; border-color: #23665e; color: #8ce4d4; }
    .cr-chip.warn { background: #342c1b; border-color: #6c592e; color: #f5d88b; }
    .cr-skill-label { color: var(--cr-muted); font-size: .75rem; font-weight: 700; letter-spacing: .08em; margin: .5rem 0 .15rem; text-transform: uppercase; }
    .cr-breakdown { display: grid; gap: .55rem; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .cr-breakdown-item { background: #101923; border: 1px solid #213244; border-radius: 8px; padding: .65rem .75rem; }
    .cr-breakdown-label { color: var(--cr-muted); font-size: .78rem; }
    .cr-breakdown-value { color: var(--cr-text); font-weight: 600; margin-top: .2rem; }
    .cr-filter { align-items: center; background: #101923; border: 1px solid var(--cr-border); border-radius: 10px; display: flex; justify-content: space-between; padding: .7rem 1rem .2rem; }
    .cr-filter-value { color: var(--cr-coral); font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; }
    .cr-footer { border-top: 1px solid var(--cr-border); color: #73859a; font-size: .78rem; margin-top: 3rem; padding: 1.5rem 0 .5rem; text-align: center; }
    @media (max-width: 700px) { .block-container { padding: 2rem 1rem 1rem; } .cr-hero { align-items: flex-start; flex-direction: column; gap: 1rem; } .cr-breakdown { grid-template-columns: 1fr; } }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()
job_catalog = load_jobs(settings.data_path)
role_options = ["Any Role", *sorted(job_catalog["job_title"].dropna().unique().tolist())]
location_options = ["Any Location", *sorted(job_catalog["location"].dropna().unique().tolist())]
default_role = "AI Engineer" if "AI Engineer" in role_options else role_options[0]
default_location = "Remote" if "Remote" in location_options else location_options[0]
max_resume_size = settings.max_upload_size_mb * 1024 * 1024

st.markdown(
    """
    <div class="cr-hero">
        <div>
            <div class="cr-eyebrow">CAREER INTELLIGENCE / 01</div>
            <div class="cr-title">CareerRank AI</div>
            <div class="cr-subtitle">Explainable, personalized job discovery</div>
        </div>
        <div class="cr-signal">● Matching engine ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="cr-section">Build your candidate profile</div>', unsafe_allow_html=True)
st.markdown('<div class="cr-section-note">Upload a resume or paste your experience, then tune the preferences that matter to you.</div>', unsafe_allow_html=True)
st.markdown('<div class="cr-panel"><div class="cr-panel-title">Resume signal</div><div class="cr-panel-note">PDF resumes are supported. Application upload limit: ' + str(settings.max_upload_size_mb) + ' MB.</div></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload resume PDF", type=["pdf"], label_visibility="collapsed")
resume_text = ""
if uploaded_file:
    if uploaded_file.size > max_resume_size:
        st.error(f"This resume is larger than the {settings.max_upload_size_mb} MB application limit.")
    else:
        try:
            resume_text = extract_text_from_pdf_bytes(uploaded_file.getvalue())
            st.caption(f"Loaded {uploaded_file.name}")
        except Exception as error:
            st.error(str(error))
resume_text = st.text_area("Resume text", value=resume_text, height=150, placeholder="Upload a PDF or paste resume text...", label_visibility="collapsed")

st.markdown('<div class="cr-section">Career preferences</div>', unsafe_allow_html=True)
st.markdown('<div class="cr-section-note">These preferences are used directly by the personalized ranking model.</div>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    target_role_selection = st.selectbox("Target role", role_options, index=role_options.index(default_role))
    location_selection = st.selectbox("Preferred location", location_options, index=location_options.index(default_location))
with right:
    experience_selection = st.selectbox("Experience level", ["Any experience", "Entry", "Mid", "Senior"])
    employment_selection = st.selectbox("Employment type", ["Any employment type", "Full-time", "Contract", "Hybrid"])
additional_skills = st.text_input("Additional skills", placeholder="Add skills separated by commas, e.g. Docker, AWS")

target_role = "" if target_role_selection == "Any Role" else target_role_selection
location = "" if location_selection == "Any Location" else location_selection
experience_level = "" if experience_selection == "Any experience" else experience_selection
employment_type = "" if employment_selection == "Any employment type" else employment_selection

find_jobs = st.button("Find my best jobs  →", type="primary", disabled=not resume_text.strip(), width="stretch")
if find_jobs:
    with st.spinner("Extracting profile, retrieving jobs, and calculating matches..."):
        st.session_state.recommendation_results = run_demo_recommendation({
            "resume_text": resume_text,
            "target_role": target_role,
            "location": location,
            "experience_level": experience_level,
            "employment_type": employment_type,
            "additional_skills": [skill.strip() for skill in additional_skills.split(",") if skill.strip()],
        })


def skill_chips(skills: list[str], style: str) -> str:
    if not skills:
        return '<span class="cr-panel-note">None detected</span>'
    return "".join(f'<span class="cr-chip {style}">{html.escape(str(skill))}</span>' for skill in skills)


if "recommendation_results" in st.session_state:
    results = st.session_state.recommendation_results
    minimum_score = st.slider("Minimum match score", 0, 100, 0, key="minimum_score")
    filtered_results = [job for job in results if job["final_score"] >= minimum_score]
    average_score = sum(job["final_score"] for job in filtered_results) / max(1, len(filtered_results))

    st.markdown('<div class="cr-section">Recommendation summary</div>', unsafe_allow_html=True)
    summary_columns = st.columns(4)
    summary_columns[0].metric("Jobs analyzed", len(job_catalog))
    summary_columns[1].metric("Jobs retrieved", settings.top_k_retrieval)
    summary_columns[2].metric("Top recommendations", len(filtered_results))
    summary_columns[3].metric("Average match", f"{average_score:.0f}%")

    st.markdown(
        f'<div class="cr-filter"><span>Showing ranked jobs at or above your threshold</span><span class="cr-filter-value">{minimum_score}%</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="cr-section">Top {len(filtered_results)} recommendations</div>', unsafe_allow_html=True)
    result_frame = pd.DataFrame(filtered_results)
    ranking_config = RankingConfig()

    for job in filtered_results:
        final_score = float(job["final_score"])
        st.markdown(
            f"""
            <div class="cr-panel">
                <div class="cr-job-head">
                    <div><div class="cr-job-rank">RANK #{job['rank']:02d}</div><div class="cr-job-title">{html.escape(str(job['job_title']))}</div><div class="cr-company">{html.escape(str(job['company']))}</div></div>
                    <div class="cr-score">{final_score:.0f}%</div>
                </div>
                <div class="cr-meta"><span>{html.escape(str(job['location']))}</span><span>{html.escape(str(job['experience_level']))}</span><span>{html.escape(str(job['employment_type']))}</span></div>
                <div class="cr-skill-label">Your strong skills</div>
                <div>{skill_chips(job.get('matching_skills', []), 'good')}</div>
                <div class="cr-skill-label">Skills to improve</div>
                <div>{skill_chips(job.get('missing_skills', []), 'warn')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(final_score / 100, 0.0), 1.0), text=f"Overall match score: {final_score:.2f}%")
        with st.expander("Why this job?"):
            st.write(job["explanation"]["summary"])
            st.caption(job["explanation"]["confidence_note"])
        with st.expander("View match-score breakdown"):
            breakdown = [
                ("Semantic similarity", "semantic_similarity", ranking_config.semantic_weight),
                ("Skill match", "skill_match", ranking_config.skill_weight),
                ("Experience", "experience_match", ranking_config.experience_weight),
                ("Role", "role_match", ranking_config.role_weight),
                ("Location", "location_match", ranking_config.location_weight),
                ("Employment", "employment_match", ranking_config.employment_weight),
            ]
            breakdown_html = "".join(
                f'<div class="cr-breakdown-item"><div class="cr-breakdown-label">{label} · {weight:.0%} weight</div><div class="cr-breakdown-value">{float(job.get(key, 0.0)) * 100:.0f}% match</div></div>'
                for label, key, weight in breakdown
            )
            st.markdown(f'<div class="cr-breakdown">{breakdown_html}</div>', unsafe_allow_html=True)

    if not filtered_results:
        st.info("No recommendations meet this minimum match score. Lower the threshold to see more ranked jobs.")

    if not result_frame.empty:
        st.markdown('<div class="cr-section">Match insights</div>', unsafe_allow_html=True)
        chart_columns = st.columns(2)
        with chart_columns[0]:
            score_chart = px.bar(result_frame, x="job_title", y="final_score", title="Match score by role", template="plotly_dark")
            score_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Score", xaxis_title=None)
            st.plotly_chart(score_chart, width="stretch")
        with chart_columns[1]:
            missing = result_frame.explode("missing_skills").dropna(subset=["missing_skills"])
            counts = missing["missing_skills"].value_counts().rename_axis("skill").reset_index(name="count")
            if counts.empty:
                st.info("No missing skills detected in the filtered recommendations.")
            else:
                gap_chart = px.bar(counts, x="skill", y="count", title="Skills to improve", template="plotly_dark")
                gap_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Jobs", xaxis_title=None)
                st.plotly_chart(gap_chart, width="stretch")

st.markdown('<div class="cr-footer">CareerRank AI • Explainable &amp; Personalized Job Recommendations</div>', unsafe_allow_html=True)
