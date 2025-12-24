import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI in Education – Advanced Analytics Dashboard",
    layout="wide"
)

# -------------------- DATA LOADING --------------------
DATA_PATH = Path(__file__).parent / "data" / "ai_in_education_synthetic_dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["survey_date"])
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_data()

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.title("Global Filters")

role_filter = st.sidebar.multiselect(
    "Role",
    df["role"].unique(),
    df["role"].unique()
)

region_filter = st.sidebar.multiselect(
    "Region",
    df["region"].unique(),
    df["region"].unique()
)

discipline_filter = st.sidebar.multiselect(
    "Discipline",
    df["discipline"].unique(),
    df["discipline"].unique()
)

dff = df[
    df["role"].isin(role_filter) &
    df["region"].isin(region_filter) &
    df["discipline"].isin(discipline_filter)
]

page = st.sidebar.radio(
    "Dashboard Pages",
    [
        "📊 Overview",
        "🎓 AI Usage Patterns",
        "🧠 Learning, Trust & Dependency",
        "👩‍🏫 Faculty, Policy & Misuse",
        "📈 Outcomes & Adoption"
    ]
)

# ======================================================
# PAGE 1: OVERVIEW
# ======================================================
if page == "📊 Overview":

    st.title("AI in Education – Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", len(dff))
    c2.metric("Avg AI Usage Days/Week", round(dff["ai_use_days_per_week"].mean(), 2))
    c3.metric("Avg Time Saved (hrs/week)", round(dff["time_saved_hours_per_week"].mean(), 2))
    c4.metric("Avg Trust in AI", round(dff["trust_in_ai_outputs_1to5"].mean(), 2))

    st.divider()

    fig1 = px.histogram(dff, x="ai_use_days_per_week", color="role",
                        title="AI Usage Days per Week")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("**Insight:** Students tend to use AI more frequently than faculty.")

    fig2 = px.box(dff, x="role", y="time_saved_hours_per_week",
                  title="Time Saved Using AI by Role")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Insight:** Students report significantly higher time savings.")

    fig3 = px.bar(dff.groupby("region")["ai_use_days_per_week"].mean().reset_index(),
                  x="region", y="ai_use_days_per_week",
                  title="Average AI Usage by Region")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.pie(dff, names="primary_ai_tool",
                  title="Most Used AI Tools")
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.histogram(dff, x="ai_literacy_1to5",
                        title="AI Literacy Distribution")
    st.plotly_chart(fig5, use_container_width=True)

# ======================================================
# PAGE 2: AI USAGE PATTERNS
# ======================================================
elif page == "🎓 AI Usage Patterns":

    st.title("AI Usage Patterns")

    fig1 = px.box(dff, x="program_level", y="ai_use_days_per_week",
                  title="AI Usage by Program Level")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(dff, x="ai_literacy_1to5", y="ai_use_days_per_week",
                      color="role",
                      title="AI Literacy vs AI Usage")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(
        dff.groupby("top_ai_use_cases")["ai_use_days_per_week"].mean().reset_index(),
        x="top_ai_use_cases", y="ai_use_days_per_week",
        title="AI Usage by Use Case"
    )
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.violin(dff, x="delivery_mode", y="ai_use_days_per_week",
                     title="AI Usage by Delivery Mode", box=True)
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.histogram(dff, x="year_of_study",
                        title="AI Usage by Year of Study")
    st.plotly_chart(fig5, use_container_width=True)

# ======================================================
# PAGE 3: LEARNING, TRUST & DEPENDENCY
# ======================================================
elif page == "🧠 Learning, Trust & Dependency":

    st.title("Learning Impact, Trust & Dependency")

    fig1 = px.scatter(dff, x="ai_use_days_per_week",
                      y="perceived_learning_benefit_1to5",
                      color="role",
                      title="AI Usage vs Learning Benefit")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(dff, x="ai_use_days_per_week",
                      y="perceived_dependency_risk_1to5",
                      color="role",
                      title="AI Usage vs Dependency Risk")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(dff, x="trust_in_ai_outputs_1to5",
                      y="verifies_ai_outputs_rate_0to1",
                      title="Trust vs Verification Behavior")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.box(dff, x="role", y="student_critical_thinking_change_index",
                  title="Critical Thinking Change by Role")
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.scatter(dff, x="perceived_cheating_risk_1to5",
                      y="fairness_equity_concern_1to5",
                      title="Cheating Risk vs Fairness Concern")
    st.plotly_chart(fig5, use_container_width=True)

# ======================================================
# PAGE 4: FACULTY, POLICY & MISUSE
# ======================================================
elif page == "👩‍🏫 Faculty, Policy & Misuse":

    st.title("Faculty View, Policy Awareness & Misuse")

    fig1 = px.bar(
        dff.groupby("institution_type")["staff_reported_ai_misuse_cases_last_term"].mean().reset_index(),
        x="institution_type", y="staff_reported_ai_misuse_cases_last_term",
        title="Reported AI Misuse by Institution Type"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(dff, x="institution_policy_awareness",
                        title="Institution Policy Awareness")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(dff.groupby("institution_ai_stance").size().reset_index(name="count"),
                  x="institution_ai_stance", y="count",
                  title="Institution AI Stance")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.scatter(dff, x="ai_training_hours_last_6_months",
                      y="trust_in_ai_outputs_1to5",
                      title="AI Training vs Trust")
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.histogram(dff, x="plagiarism_or_misuse_flag",
                        title="Plagiarism / Misuse Flags")
    st.plotly_chart(fig5, use_container_width=True)

# ======================================================
# PAGE 5: OUTCOMES & ADOPTION
# ======================================================
elif page == "📈 Outcomes & Adoption":

    st.title("Outcomes & AI Adoption")

    fig1 = px.bar(
        dff.groupby("ai_adoption_category")["student_grade_change_pct"].mean().reset_index(),
        x="ai_adoption_category", y="student_grade_change_pct",
        title="Grade Change by AI Adoption Level"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        dff.groupby("ai_adoption_category")["student_stress_change_index"].mean().reset_index(),
        x="ai_adoption_category", y="student_stress_change_index",
        title="Stress Change by AI Adoption Level"
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(dff, x="ai_use_days_per_week",
                      y="student_grade_change_pct",
                      color="ai_adoption_category",
                      title="AI Usage vs Grade Change")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.scatter(dff, x="ai_use_days_per_week",
                      y="student_stress_change_index",
                      color="ai_adoption_category",
                      title="AI Usage vs Stress Change")
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.scatter(dff, x="time_saved_hours_per_week",
                      y="student_critical_thinking_change_index",
                      title="Time Saved vs Critical Thinking")
    st.plotly_chart(fig5, use_container_width=True)
