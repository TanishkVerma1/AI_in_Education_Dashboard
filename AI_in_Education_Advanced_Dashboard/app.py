
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="AI in Education – Advanced Dashboard", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "ai_in_education_synthetic_dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["survey_date"])
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_data()

def pick_col(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

COL_ROLE = pick_col(["role"])
COL_REGION = pick_col(["region"])
COL_DATE = pick_col(["survey_date"])
COL_USAGE = pick_col(["ai_usage_hours_per_week"])
COL_TRUST = pick_col(["trust_in_ai_outputs_1to5"])
COL_DEP = pick_col(["perceived_dependency_risk_1to5"])
COL_PERF = pick_col(["academic_performance_change_1to5"])
COL_LEVEL = pick_col(["education_level"])
COL_INST = pick_col(["institution_type"])
COL_ADOPT = pick_col(["ai_adoption_category"])
COL_MISUSE = pick_col(["staff_reported_ai_misuse_cases_last_term"])

st.sidebar.header("Filters")
role_sel = st.sidebar.multiselect("Role", df[COL_ROLE].unique(), df[COL_ROLE].unique())
region_sel = st.sidebar.multiselect("Region", df[COL_REGION].unique(), df[COL_REGION].unique())

dff = df[df[COL_ROLE].isin(role_sel) & df[COL_REGION].isin(region_sel)]

page = st.sidebar.radio(
    "Pages",
    ["📊 Overview", "🎓 Usage Patterns", "🧠 Trust & Dependency", "👩‍🏫 Faculty & Misuse", "📈 Outcomes & Adoption"]
)

if page == "📊 Overview":
    st.title("AI in Education – Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", len(dff))
    c2.metric("Avg AI Usage (hrs/week)", round(dff[COL_USAGE].mean(), 2))
    c3.metric("Avg Trust Score", round(dff[COL_TRUST].mean(), 2))
    c4.metric("Avg Academic Impact", round(dff[COL_PERF].mean(), 2))

    fig = px.histogram(dff, x=COL_USAGE, color=COL_ROLE, title="Weekly AI Usage Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Key Insights**")
    st.markdown("- Students generally use AI more than faculty.")
    st.markdown("- Usage varies across regions and demographics.")

elif page == "🎓 Usage Patterns":
    st.title("AI Usage Patterns")
    fig1 = px.box(dff, x=COL_LEVEL, y=COL_USAGE, color=COL_ROLE, title="AI Usage by Education Level")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.violin(dff, x=COL_REGION, y=COL_USAGE, color=COL_ROLE, box=True, title="AI Usage Across Regions")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Key Insights**")
    st.markdown("- Higher education correlates with higher AI usage.")
    st.markdown("- Regional disparities indicate access and policy differences.")

elif page == "🧠 Trust & Dependency":
    st.title("Trust in AI & Dependency Risk")
    fig1 = px.scatter(dff, x=COL_USAGE, y=COL_TRUST, color=COL_ROLE, title="AI Usage vs Trust")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(dff, x=COL_USAGE, y=COL_DEP, color=COL_ROLE, title="AI Usage vs Dependency Risk")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Key Insights**")
    st.markdown("- Trust increases with usage.")
    st.markdown("- High usage shows stronger dependency risk.")

elif page == "👩‍🏫 Faculty & Misuse":
    st.title("Faculty Perspective & AI Misuse")
    faculty = dff[dff[COL_ROLE] == "Faculty"]

    fig1 = px.bar(
        faculty.groupby(COL_INST)[COL_MISUSE].mean().reset_index(),
        x=COL_INST, y=COL_MISUSE,
        title="Reported AI Misuse by Institution Type"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(faculty, x=COL_INST, y=COL_TRUST, title="Faculty Trust by Institution")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Key Insights**")
    st.markdown("- Traditional institutions report more misuse cases.")
    st.markdown("- Faculty trust differs by institution type.")

elif page == "📈 Outcomes & Adoption":
    st.title("Outcomes & AI Adoption")
    fig1 = px.bar(
        dff.groupby(COL_ADOPT)[COL_PERF].mean().reset_index(),
        x=COL_ADOPT, y=COL_PERF,
        title="Academic Impact by AI Adoption Level"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(dff, x=COL_USAGE, y=COL_PERF, color=COL_ADOPT, title="AI Usage vs Academic Performance")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Key Insights**")
    st.markdown("- Moderate adoption yields best outcomes.")
    st.markdown("- Overuse may reduce learning effectiveness.")
