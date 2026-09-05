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
        --cr-blue: #6f8cff;
        --cr-purple: #8b5cf6;
        --cr-teal: #38c8b2;
        --cr-yellow: #f3c969;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--cr-bg); color: var(--cr-text); }
    .block-container { max-width: 1220px; padding: 1.25rem 2rem 1.75rem; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    h2 { margin-top: 2rem; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stFileUploaderDropzone"] { background: #101923; border: 1px dashed #38506a; border-radius: 12px; }
    [data-testid="stFileUploaderDropzoneInstructions"] { color: var(--cr-muted); }
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input,
    [data-baseweb="select"] > div { background: #151f2b; border-color: var(--cr-border); border-radius: 9px; }
    [data-testid="stTextArea"] textarea:focus, [data-testid="stTextInput"] input:focus { border-color: var(--cr-teal); }
    [data-testid="stSlider"] [role="slider"] { background: var(--cr-blue); }
    .stButton > button[kind="primary"] { background: linear-gradient(100deg, var(--cr-blue), var(--cr-purple)); border: 0; border-radius: 9px; color: #fff; font-weight: 700; min-height: 2.8rem; }
    .stButton > button[kind="primary"]:hover { background: linear-gradient(100deg, #86a0ff, #a276ff); color: #fff; }
    [data-testid="stExpander"] { background: #121a24; border: 1px solid var(--cr-border); border-radius: 10px; }
    [data-testid="stMetric"] { background: var(--cr-panel); border: 1px solid var(--cr-border); border-radius: 10px; padding: 1rem 1.1rem; }
    [data-testid="stMetricLabel"] { color: var(--cr-muted); }
    [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
    .cr-nav { align-items: center; border-bottom: 1px solid var(--cr-border); display: flex; justify-content: space-between; margin-bottom: 1.4rem; padding: .25rem 0 .85rem; }
    .cr-brand { color: var(--cr-text); font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700; }
    .cr-nav-links { color: var(--cr-muted); font-size: .78rem; letter-spacing: .02em; word-spacing: 1.1rem; }
    .cr-hero { align-items: center; border-bottom: 1px solid var(--cr-border); display: flex; justify-content: space-between; margin-bottom: 1.4rem; padding-bottom: 1.35rem; }
    .cr-eyebrow { color: var(--cr-teal); font-size: .72rem; font-weight: 700; letter-spacing: .16em; margin-bottom: .55rem; }
    .cr-title { color: var(--cr-text); font-family: 'Space Grotesk', sans-serif; font-size: clamp(2rem, 4vw, 3rem); font-weight: 700; line-height: .98; }
    .cr-subtitle { color: var(--cr-muted); font-size: .94rem; margin-top: .55rem; }
    .cr-description { color: #c0ccda; font-size: .84rem; line-height: 1.5; margin-top: .65rem; max-width: 610px; }
    .cr-features { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .8rem; }
    .cr-feature { background: #131e2d; border: 1px solid #2b3b57; border-radius: 999px; color: #cbd7f0; font-size: .76rem; padding: .4rem .65rem; }
    .cr-signal { background: #102a2b; border: 1px solid #1e5e5a; border-radius: 999px; color: #8ce4d4; font-size: .75rem; padding: .45rem .7rem; }
    .cr-hero-visual { background: #101923; border: 1px solid #2b3b57; border-radius: 12px; padding: .8rem; }
    .cr-flow-title { color: var(--cr-muted); font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; }
    .cr-flow { align-items: center; display: flex; gap: .35rem; margin-top: .55rem; }
    .cr-flow-card { background: #172538; border: 1px solid #334b6b; border-radius: 8px; color: #dbe6f8; font-size: .72rem; padding: .6rem .65rem; text-align: center; }
    .cr-flow-arrow { color: var(--cr-teal); font-size: .85rem; }
    .cr-section { color: var(--cr-text); font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 600; margin: 1.25rem 0 .65rem; }
    .cr-section-note { color: var(--cr-muted); font-size: .78rem; margin-top: -.35rem; margin-bottom: .7rem; }
    .cr-panel { background: var(--cr-panel); border: 1px solid var(--cr-border); border-radius: 10px; padding: .95rem; }
    .cr-panel-title { color: var(--cr-text); font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600; margin-bottom: .25rem; }
    .cr-panel-note { color: var(--cr-muted); font-size: .82rem; }
    .cr-job-head { align-items: center; display: flex; gap: .75rem; justify-content: space-between; padding: .05rem 0 .45rem; }
    .cr-job-rank { color: var(--cr-blue); font-size: .75rem; font-weight: 700; letter-spacing: .1em; }
    .cr-job-title { color: var(--cr-text); font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 600; }
    .cr-company { color: var(--cr-muted); font-size: .88rem; margin-top: .15rem; }
    .cr-score { color: var(--cr-teal); font-family: 'Space Grotesk', sans-serif; font-size: 1.55rem; font-weight: 700; white-space: nowrap; }
    .cr-meta { color: var(--cr-muted); display: flex; flex-wrap: wrap; font-size: .76rem; gap: .4rem .75rem; padding: .3rem 0 .45rem; }
    .cr-meta span::before { color: var(--cr-teal); content: '•'; margin-right: .35rem; }
    .cr-chip { background: #1c2b38; border: 1px solid #2c4153; border-radius: 999px; color: #c9d6e3; display: inline-block; font-size: .75rem; margin: .18rem .25rem .18rem 0; padding: .28rem .55rem; }
    .cr-chip.good { background: #12302d; border-color: #23665e; color: #8ce4d4; }
    .cr-chip.warn { background: #342c1b; border-color: #6c592e; color: #f5d88b; }
    .cr-skill-label { color: var(--cr-muted); font-size: .75rem; font-weight: 700; letter-spacing: .08em; margin: .5rem 0 .15rem; text-transform: uppercase; }
    .cr-breakdown { display: grid; gap: .55rem; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .cr-breakdown-item { background: #101923; border: 1px solid #213244; border-radius: 8px; padding: .65rem .75rem; }
    .cr-breakdown-label { color: var(--cr-muted); font-size: .78rem; }
    .cr-breakdown-value { color: var(--cr-text); font-weight: 600; margin-top: .2rem; }
    .cr-filter { align-items: center; background: #101923; border: 1px solid var(--cr-border); border-radius: 10px; display: flex; justify-content: space-between; padding: .55rem .85rem .1rem; }
    .cr-filter-value { color: var(--cr-blue); font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; }
    .cr-skill-card { min-height: 9rem; }
    .cr-skill-card-title { color: var(--cr-teal); font-size: .76rem; font-weight: 700; letter-spacing: .1em; margin-bottom: .55rem; text-transform: uppercase; }
    .cr-skill-card-title.warn { color: var(--cr-yellow); }
    .cr-footer { border-top: 1px solid var(--cr-border); color: #73859a; font-size: .78rem; margin-top: 3rem; padding: 1.5rem 0 .5rem; text-align: center; }
    .cr-footer strong { color: #c0ccda; display: block; font-family: 'Space Grotesk', sans-serif; font-size: .9rem; margin-bottom: .3rem; }
    @media (max-width: 700px) { .block-container { padding: 1rem .8rem 1rem; } .cr-nav-links { display: none; } .cr-hero { align-items: flex-start; flex-direction: column; gap: .9rem; } .cr-breakdown { grid-template-columns: 1fr; } }
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
    '<div class="cr-nav"><div class="cr-brand">🚀 CareerRank AI</div><div class="cr-nav-links">Home Find Jobs Insights About</div><div class="cr-signal">● AI Matching Ready</div></div>',
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([1.35, .85], gap="large")
with hero_left:
    st.markdown(
        """
        <div class="cr-hero">
            <div>
                <div class="cr-eyebrow">CAREER INTELLIGENCE / 01</div>
                <div class="cr-title">CareerRank AI</div>
                <div class="cr-subtitle">Explainable, personalized job discovery</div>
                <div class="cr-description">Upload your resume, set your career preferences, and discover job opportunities ranked according to your skills and preferences.</div>
                <div class="cr-features"><span class="cr-feature">✦ AI-Powered Matching</span><span class="cr-feature">◈ Personalized Recommendations</span><span class="cr-feature">◎ Explainable Results</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    st.markdown(
        """
        <div class="cr-hero-visual">
            <div class="cr-flow-title">Live matching pipeline</div>
            <div class="cr-flow"><div class="cr-flow-card">Resume<br><small>uploaded</small></div><div class="cr-flow-arrow">→</div><div class="cr-flow-card">Skills<br><small>extracted</small></div><div class="cr-flow-arrow">→</div><div class="cr-flow-card">Jobs<br><small>ranked</small></div></div>
            <div class="cr-panel-note" style="margin-top:.65rem">Personalized ranking is ready for your preferences.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="cr-section">Candidate setup</div>', unsafe_allow_html=True)
upload_column, preference_column = st.columns([.9, 1.1], gap="medium")
with upload_column:
    with st.container(border=True):
        st.markdown('<div class="cr-panel-title">📄 Upload Your Resume</div><div class="cr-panel-note">PDF only • Maximum ' + str(settings.max_upload_size_mb) + ' MB</div>', unsafe_allow_html=True)
        if "resume_text" not in st.session_state:
            st.session_state.resume_text = ""
        uploaded_file = st.file_uploader("Upload resume PDF", type=["pdf"], label_visibility="collapsed", key="resume_upload")
        if uploaded_file:
            upload_signature = (uploaded_file.name, uploaded_file.size)
            if upload_signature != st.session_state.get("resume_upload_signature"):
                st.session_state.resume_upload_signature = upload_signature
                if uploaded_file.size > max_resume_size:
                    st.session_state.resume_text = ""
                    st.error(f"This resume is larger than the {settings.max_upload_size_mb} MB application limit.")
                else:
                    try:
                        st.session_state.resume_text = extract_text_from_pdf_bytes(uploaded_file.getvalue())
                    except Exception as error:
                        st.session_state.resume_text = ""
                        st.error(str(error))
        if uploaded_file and st.session_state.resume_text:
            st.caption(f"Loaded {uploaded_file.name}")
        resume_text = st.text_area("Resume text", height=112, placeholder="Or paste resume text here...", label_visibility="collapsed", key="resume_text")
with preference_column:
    with st.container(border=True):
        st.markdown('<div class="cr-panel-title">🎯 Career Preferences</div><div class="cr-panel-note">Tune the factors used by personalized ranking.</div>', unsafe_allow_html=True)
        preference_left, preference_right = st.columns(2)
        with preference_left:
            target_role_selection = st.selectbox("Target role", role_options, index=role_options.index(default_role))
            location_selection = st.selectbox("Preferred location", location_options, index=location_options.index(default_location))
        with preference_right:
            experience_selection = st.selectbox("Experience level", ["Any experience", "Entry", "Mid", "Senior"])
            employment_selection = st.selectbox("Employment type", ["Any employment type", "Full-time", "Contract", "Hybrid"])
        additional_skills = st.text_input("Additional skills", placeholder="Docker, AWS, Kubernetes")

target_role = "" if target_role_selection == "Any Role" else target_role_selection
location = "" if location_selection == "Any Location" else location_selection
experience_level = "" if experience_selection == "Any experience" else experience_selection
employment_type = "" if employment_selection == "Any employment type" else employment_selection

find_jobs = st.button("🔍 Find My Best Jobs", type="primary", disabled=not resume_text.strip(), width="stretch")
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
    threshold_label, threshold_control = st.columns([.55, 1.45], gap="medium")
    with threshold_label:
        st.markdown('<div class="cr-panel-title">Minimum Match Score</div><div class="cr-panel-note">Filter ranked jobs by their actual final score.</div>', unsafe_allow_html=True)
    with threshold_control:
        minimum_score = st.slider("Minimum Match Score", 0, 100, 0, key="minimum_score", label_visibility="collapsed")
    filtered_results = [job for job in results if job["final_score"] >= minimum_score]
    average_score = sum(job["final_score"] for job in filtered_results) / max(1, len(filtered_results))

    st.markdown('<div class="cr-section">📊 Recommendation Summary</div>', unsafe_allow_html=True)
    summary_columns = st.columns(4)
    summary_columns[0].metric("Jobs analyzed", len(job_catalog))
    summary_columns[1].metric("Jobs retrieved", settings.top_k_retrieval)
    summary_columns[2].metric("Top recommendations", len(filtered_results))
    summary_columns[3].metric("Average match", f"{average_score:.0f}%")

    st.markdown(
        f'<div class="cr-filter"><span>Showing ranked jobs at or above your threshold</span><span class="cr-filter-value">{minimum_score}%</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="cr-section">🔥 Top Job Matches <span class="cr-panel-note">({len(filtered_results)} shown)</span></div>', unsafe_allow_html=True)
    result_frame = pd.DataFrame(filtered_results)
    ranking_config = RankingConfig()
    recommendation_columns = st.columns(2, gap="medium")

    for index, job in enumerate(filtered_results):
        with recommendation_columns[index % 2]:
            final_score = float(job["final_score"])
            st.markdown(
                f"""
                <div class="cr-panel">
                    <div class="cr-job-head">
                        <div><div class="cr-job-rank">RANK #{job['rank']:02d}</div><div class="cr-job-title">{html.escape(str(job['job_title']))}</div><div class="cr-company">{html.escape(str(job['company']))}</div></div>
                        <div class="cr-score">{final_score:.0f}% <small>MATCH</small></div>
                    </div>
                    <div class="cr-meta"><span>{html.escape(str(job['location']))}</span><span>{html.escape(str(job['employment_type']))}</span><span>{html.escape(str(job['experience_level']))}</span></div>
                    <div class="cr-skill-label">Matching skills</div>
                    <div>{skill_chips(job.get('matching_skills', []), 'good')}</div>
                    <div class="cr-skill-label">Skills to improve</div>
                    <div>{skill_chips(job.get('missing_skills', []), 'warn')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(final_score / 100, 0.0), 1.0), text=f"Match score: {final_score:.2f}%")
            with st.expander("View Match Details"):
                st.markdown("**💡 Why this job?**")
                st.write(job["explanation"]["summary"])
                st.caption(job["explanation"]["confidence_note"])
                gap = job.get("skill_gap", {})
                st.markdown("**Matching skills**")
                st.markdown(skill_chips(gap.get("strong_skills", job.get("matching_skills", [])), "good"), unsafe_allow_html=True)
                st.markdown("**Missing required skills**")
                st.markdown(skill_chips(gap.get("missing_required_skills", job.get("missing_skills", [])), "warn"), unsafe_allow_html=True)
                st.markdown("**Missing preferred skills**")
                st.markdown(skill_chips(gap.get("missing_preferred_skills", []), ""), unsafe_allow_html=True)
                st.markdown("**📊 Match Score Breakdown**")
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
        strong_skills = sorted({skill for job in filtered_results for skill in job.get("skill_gap", {}).get("strong_skills", [])})
        skills_to_improve = sorted({skill for job in filtered_results for skill in job.get("skill_gap", {}).get("missing_required_skills", []) + job.get("skill_gap", {}).get("missing_preferred_skills", [])})
        st.markdown('<div class="cr-section">🎯 Skill Gap Analysis</div>', unsafe_allow_html=True)
        gap_columns = st.columns(2)
        with gap_columns[0]:
            st.markdown(f'<div class="cr-panel cr-skill-card"><div class="cr-skill-card-title">Your strong skills</div>{skill_chips(strong_skills, "good")}</div>', unsafe_allow_html=True)
        with gap_columns[1]:
            st.markdown(f'<div class="cr-panel cr-skill-card"><div class="cr-skill-card-title warn">Skills to improve</div>{skill_chips(skills_to_improve, "warn")}</div>', unsafe_allow_html=True)
        st.markdown('<div class="cr-section">📈 Career Insights</div>', unsafe_allow_html=True)
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

st.markdown('<div class="cr-footer"><strong>CareerRank AI</strong>Explainable &amp; Personalized Job Recommendations<br>Built with Python • Sentence Transformers • FAISS • FastAPI • Streamlit</div>', unsafe_allow_html=True)
