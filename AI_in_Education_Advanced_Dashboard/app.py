import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="AI in Education – Insight Dashboard",
    layout="wide"
)

# ----------------------------------------------------
# DATA LOADING
# ----------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "ai_in_education_synthetic_dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["survey_date"])
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_data()

# ----------------------------------------------------
# SIDEBAR FILTERS (GLOBAL)
# ----------------------------------------------------
st.sidebar.title("Global Filters")

role_sel = st.sidebar.multiselect(
    "Role",
    sorted(df["role"].unique()),
    default=sorted(df["role"].unique())
)

region_sel = st.sidebar.multiselect(
    "Region",
    sorted(df["region"].unique()),
    default=sorted(df["region"].unique())
)

program_sel = st.sidebar.multiselect(
    "Program Level",
    sorted(df["program_level"].unique()),
    default=sorted(df["program_level"].unique())
)

dff = df[
    (df["role"].isin(role_sel)) &
    (df["region"].isin(region_sel)) &
    (df["program_level"].isin(program_sel))
]

page = st.sidebar.radio(
    "Dashboard Pages",
    [
        "📘 Overview",
        "🎓 AI Usage Patterns",
        "🧠 Learning, Trust & Dependency",
        "👩‍🏫 Faculty, Policy & Misuse",
        "📈 Student Outcomes & Adoption"
    ]
)

# ----------------------------------------------------
# PAGE 1 — OVERVIEW
# ----------------------------------------------------
if page == "📘 Overview":

    st.title("AI in Education – Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Respondents", len(dff))
    c2.metric("Avg AI Usage Days / Week", round(dff["ai_use_days_per_week"].mean(), 2))
    c3.metric("Avg Trust in AI (1–5)", round(dff["trust_in_ai_outputs_1to5"].mean(), 2))
    c4.metric("Avg Learning Benefit (1–5)", round(dff["perceived_learning_benefit_1to5"].mean(), 2))

    st.divider()

    # 1️⃣ Role Distribution
    fig1 = px.pie(
        dff,
        names="role",
        title="Respondent Distribution by Role",
        hole=0.4
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- The dataset captures perspectives from both **students and faculty**, enabling balanced insights.\n"
        "- A strong student presence reflects real-world AI usage trends among learners."
    )

    # 2️⃣ AI Adoption Category
    fig2 = px.bar(
        dff["ai_adoption_category"].value_counts().reset_index(),
        x="index", y="ai_adoption_category",
        labels={"index": "AI Adoption Level", "ai_adoption_category": "Count"},
        title="AI Adoption Levels"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- Most respondents fall into **Moderate or High adoption** categories.\n"
        "- This suggests AI tools are no longer experimental but mainstream in education."
    )

    # 3️⃣ AI Literacy
    fig3 = px.histogram(
        dff,
        x="ai_literacy_1to5",
        nbins=5,
        title="AI Literacy Levels (1–5)"
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- Majority of users report **medium to high AI literacy**.\n"
        "- Training programs are likely contributing positively."
    )

    # 4️⃣ Digital Comfort
    fig4 = px.box(
        dff,
        y="digital_comfort_1to5",
        title="Digital Comfort Distribution"
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- High digital comfort indicates readiness for AI-integrated education systems."
    )

    # 5️⃣ Internet Quality
    fig5 = px.bar(
        dff.groupby("region")["internet_quality_1to5"].mean().reset_index(),
        x="region", y="internet_quality_1to5",
        title="Average Internet Quality by Region"
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- Regions with lower internet quality may face barriers in effective AI adoption."
    )

    st.success(
        "### Page Conclusion & Actions\n"
        "- AI adoption is widespread and supported by good digital readiness.\n"
        "- Institutions should **focus on improving infrastructure** in low-connectivity regions.\n"
        "- Continued **AI literacy training** will strengthen responsible usage."
    )

# ----------------------------------------------------
# PAGE 2 — AI USAGE PATTERNS
# ----------------------------------------------------
elif page == "🎓 AI Usage Patterns":

    st.title("AI Usage Patterns")

    # 1️⃣ Days per Week
    fig1 = px.histogram(
        dff,
        x="ai_use_days_per_week",
        title="AI Usage Frequency (Days per Week)"
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- Most users rely on AI **3–6 days per week**, indicating habitual use."
    )

    # 2️⃣ Time Saved
    fig2 = px.box(
        dff,
        y="time_saved_hours_per_week",
        title="Weekly Time Saved Using AI"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- AI provides **significant time savings**, especially for repetitive academic tasks."
    )

    # 3️⃣ Primary AI Tool
    fig3 = px.bar(
        dff["primary_ai_tool"].value_counts().reset_index(),
        x="index", y="primary_ai_tool",
        title="Most Used AI Tools"
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- Generative AI tools dominate due to versatility across learning activities."
    )

    # 4️⃣ Top Use Cases
    fig4 = px.bar(
        dff["top_ai_use_cases"].value_counts().head(10).reset_index(),
        x="index", y="top_ai_use_cases",
        title="Top AI Use Cases"
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- AI is primarily used for **assignments, exam prep, and concept clarification**."
    )

    # 5️⃣ Assessment Impact
    fig5 = px.bar(
        dff["assessment_context_most_impacted"].value_counts().reset_index(),
        x="index", y="assessment_context_most_impacted",
        title="Assessment Areas Most Impacted by AI"
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- Written assessments and take-home assignments are most influenced by AI."
    )

    st.success(
        "### Page Conclusion & Actions\n"
        "- AI is deeply embedded in daily academic workflows.\n"
        "- Institutions should **redesign assessments** toward application-based evaluation.\n"
        "- Provide **clear guidance** on acceptable AI usage."
    )

# ----------------------------------------------------
# PAGE 3 — LEARNING, TRUST & DEPENDENCY
# ----------------------------------------------------
elif page == "🧠 Learning, Trust & Dependency":

    st.title("Learning Impact, Trust & Dependency")

    fig1 = px.bar(
        dff.groupby("ai_use_days_per_week")["perceived_learning_benefit_1to5"].mean().reset_index(),
        x="ai_use_days_per_week", y="perceived_learning_benefit_1to5",
        title="Learning Benefit vs AI Usage Frequency"
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- Moderate AI usage delivers the **highest perceived learning benefit**."
    )

    fig2 = px.bar(
        dff.groupby("ai_use_days_per_week")["perceived_dependency_risk_1to5"].mean().reset_index(),
        x="ai_use_days_per_week", y="perceived_dependency_risk_1to5",
        title="Dependency Risk vs AI Usage"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- Heavy usage correlates with **higher dependency risk**."
    )

    fig3 = px.histogram(
        dff,
        x="trust_in_ai_outputs_1to5",
        title="Trust in AI Outputs"
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- Users generally trust AI but not blindly — trust is cautious."
    )

    fig4 = px.box(
        dff,
        y="fairness_equity_concern_1to5",
        title="Fairness & Bias Concerns"
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- Ethical concerns remain significant among users."
    )

    fig5 = px.bar(
        dff.groupby("ai_literacy_1to5")["verifies_ai_outputs_rate_0to1"].mean().reset_index(),
        x="ai_literacy_1to5", y="verifies_ai_outputs_rate_0to1",
        title="Verification Behavior vs AI Literacy"
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- Higher AI literacy leads to **more verification of AI outputs**."
    )

    st.success(
        "### Page Conclusion & Actions\n"
        "- AI improves learning when used responsibly.\n"
        "- Institutions should **teach verification and critical thinking**.\n"
        "- Promote AI as an assistant, not a replacement."
    )

# ----------------------------------------------------
# PAGE 4 — FACULTY, POLICY & MISUSE
# ----------------------------------------------------
elif page == "👩‍🏫 Faculty, Policy & Misuse":

    st.title("Faculty Perspective, Policy & Misuse")

    faculty = dff[dff["role"] == "Faculty"]

    fig1 = px.bar(
        faculty.groupby("institution_type")["staff_reported_ai_misuse_cases_last_term"].mean().reset_index(),
        x="institution_type", y="staff_reported_ai_misuse_cases_last_term",
        title="Reported AI Misuse by Institution Type"
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- Traditional institutions report more misuse incidents."
    )

    fig2 = px.pie(
        dff,
        names="institution_ai_stance",
        title="Institutional AI Policy Stance",
        hole=0.4
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- Many institutions still lack clear AI policies."
    )

    fig3 = px.bar(
        dff.groupby("institution_policy_awareness")["plagiarism_or_misuse_flag"].mean().reset_index(),
        x="institution_policy_awareness", y="plagiarism_or_misuse_flag",
        title="Misuse vs Policy Awareness"
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(
        "**Key Insight:**\n"
        "- Better policy awareness reduces misuse."
    )

    fig4 = px.histogram(
        faculty,
        x="ai_training_hours_last_6_months",
        title="Faculty AI Training Hours"
    )
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.box(
        faculty,
        y="trust_in_ai_outputs_1to5",
        title="Faculty Trust in AI"
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.success(
        "### Page Conclusion & Actions\n"
        "- Clear policies + faculty training reduce misuse.\n"
        "- Institutions must **formalize AI governance frameworks**.\n"
        "- Invest in **faculty AI upskilling programs**."
    )

# ----------------------------------------------------
# PAGE 5 — STUDENT OUTCOMES & ADOPTION
# ----------------------------------------------------
elif page == "📈 Student Outcomes & Adoption":

    students = dff[dff["role"] == "Student"]

    st.title("Student Outcomes & AI Adoption")

    fig1 = px.bar(
        students.groupby("ai_adoption_category")["student_grade_change_pct"].mean().reset_index(),
        x="ai_adoption_category", y="student_grade_change_pct",
        title="Grade Change by AI Adoption Level"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        students.groupby("ai_adoption_category")["student_critical_thinking_change_index"].mean().reset_index(),
        x="ai_adoption_category", y="student_critical_thinking_change_index",
        title="Critical Thinking Impact"
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(
        students.groupby("ai_adoption_category")["student_stress_change_index"].mean().reset_index(),
        x="ai_adoption_category", y="student_stress_change_index",
        title="Stress Change Due to AI"
    )
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.box(
        students,
        y="student_grade_change_pct",
        title="Overall Grade Change Distribution"
    )
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.scatter(
        students,
        x="ai_use_days_per_week",
        y="student_grade_change_pct",
        title="AI Usage vs Grade Change"
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.success(
        "### Page Conclusion & Actions\n"
        "- Moderate AI adoption yields **better grades and thinking skills**.\n"
        "- Excessive usage may increase stress.\n"
        "- Encourage **balanced AI usage strategies** for students."
    )
