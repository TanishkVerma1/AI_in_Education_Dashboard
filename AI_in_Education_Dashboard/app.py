import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="AI in Education Dashboard", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "ai_in_education_synthetic_dataset.csv"

@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(
            f"Dataset file not found at:\n{DATA_PATH}\n\n"
            "✅ Fix:\n"
            "1) Create a folder named `data` in your repo (same level as app.py)\n"
            "2) Put the CSV inside it\n"
            "3) Ensure the filename is exactly: `ai_in_education_synthetic_dataset.csv`"
        )
        st.stop()

    df = pd.read_csv(DATA_PATH)

    # Parse date safely (won't crash if column missing)
    if "survey_date" in df.columns:
        df["survey_date"] = pd.to_datetime(df["survey_date"], errors="coerce")

    return df

df = load_data()

# ---------------- Sidebar ----------------
st.sidebar.header("Global Filters")
role = st.sidebar.multiselect("Role", df["role"].unique(), df["role"].unique())
region = st.sidebar.multiselect("Region", df["region"].unique(), df["region"].unique())

dff = df[df["role"].isin(role) & df["region"].isin(region)]

page = st.sidebar.radio(
    "Pages",
    ["Overview", "Student Usage", "Faculty View", "Risks & Ethics", "Outcomes"]
)

# ---------------- Overview ----------------
if page == "Overview":
    st.title("AI in Education – Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Respondents", len(dff))
    c2.metric("Avg AI Usage (hrs/week)", round(dff["ai_usage_hours_per_week"].mean(), 2))
    c3.metric("Avg Academic Impact", round(dff["academic_performance_change_1to5"].mean(), 2))

    fig = px.histogram(dff, x="ai_usage_hours_per_week", color="role",
                       title="AI Usage Hours per Week")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("- Students use AI more frequently than faculty overall.")

# ---------------- Student Usage ----------------
elif page == "Student Usage":
    st.title("Student AI Usage Patterns")
    students = dff[dff["role"] == "Student"]

    fig = px.box(students, x="education_level", y="ai_usage_hours_per_week",
                 title="AI Usage by Education Level")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("- Higher education levels show increased AI usage.")

# ---------------- Faculty View ----------------
elif page == "Faculty View":
    st.title("Faculty Perspective on AI")

    faculty = dff[dff["role"] == "Faculty"]
    fig = px.bar(
        faculty.groupby("institution_type")["staff_reported_ai_misuse_cases_last_term"].mean().reset_index(),
        x="institution_type",
        y="staff_reported_ai_misuse_cases_last_term",
        title="Reported AI Misuse Cases"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("- Traditional institutions report higher misuse cases.")

# ---------------- Risks & Ethics ----------------
elif page == "Risks & Ethics":
    st.title("Risks and Ethical Concerns")

    fig = px.scatter(
        dff,
        x="ai_usage_hours_per_week",
        y="perceived_dependency_risk_1to5",
        color="role",
        title="AI Usage vs Dependency Risk"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("- Dependency risk rises with increased AI usage.")

# ---------------- Outcomes ----------------
elif page == "Outcomes":
    st.title("Educational Outcomes & Adoption")

    fig = px.bar(
        dff.groupby("ai_adoption_category")["academic_performance_change_1to5"].mean().reset_index(),
        x="ai_adoption_category",
        y="academic_performance_change_1to5",
        title="Academic Impact by AI Adoption Level"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("- Moderate AI adoption yields the best outcomes.")
