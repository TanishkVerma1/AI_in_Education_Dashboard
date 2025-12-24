import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go

def add_linear_trendline(fig, df_in, x_col, y_col, name="Trend line"):
    """
    Adds a simple linear regression line using numpy (no statsmodels needed).
    """
    tmp = df_in[[x_col, y_col]].dropna().copy()
    if len(tmp) < 2:
        return fig

    x = tmp[x_col].astype(float).to_numpy()
    y = tmp[y_col].astype(float).to_numpy()

    # Fit y = m*x + b
    m, b = np.polyfit(x, y, 1)

    # Make a smooth line across the x-range
    xs = np.linspace(x.min(), x.max(), 100)
    ys = m * xs + b

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=name,
            line=dict(width=3, dash="dash"),
        )
    )
    return fig

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI in Education – Insight Dashboard", layout="wide")
px.defaults.template = "plotly_dark"

DATA_PATH = Path(__file__).parent / "data" / "ai_in_education_synthetic_dataset.csv"

# -------------------------
# HELPERS
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    # parse date safely
    if "survey_date" in df.columns:
        df["survey_date"] = pd.to_datetime(df["survey_date"], errors="coerce")
    return df

def safe_sorted_unique(series):
    return sorted([x for x in series.dropna().unique().tolist()])

def vc_df(dff: pd.DataFrame, col: str, top_n: int | None = None):
    """Stable value_counts dataframe: columns = [col, count, pct] always."""
    if col not in dff.columns or len(dff) == 0:
        return pd.DataFrame(columns=[col, "count", "pct"])
    s = dff[col].astype(str).fillna("Missing")
    out = s.value_counts(dropna=False).rename_axis(col).reset_index(name="count")
    out["pct"] = (out["count"] / out["count"].sum() * 100).round(1)
    if top_n is not None:
        out = out.head(top_n)
    return out

def mean_by(dff: pd.DataFrame, group_col: str, value_col: str):
    if len(dff) == 0 or group_col not in dff.columns or value_col not in dff.columns:
        return pd.DataFrame(columns=[group_col, value_col])
    tmp = dff.groupby(group_col, dropna=False)[value_col].mean().reset_index()
    tmp[group_col] = tmp[group_col].astype(str)
    return tmp

def insight_box(lines):
    st.markdown("**Key insights (easy explanation):**")
    for ln in lines:
        st.markdown(f"- {ln}")

def page_conclusion(title, bullets, actions):
    st.success(f"### {title}\n" +
               "\n".join([f"- {b}" for b in bullets]) +
               "\n\n**What can be done (actions):**\n" +
               "\n".join([f"- {a}" for a in actions]))

def guard_nonempty(dff, msg="No data left after filters. Please widen filters from the sidebar."):
    if len(dff) == 0:
        st.warning(msg)
        st.stop()

# -------------------------
# LOAD
# -------------------------
df = load_data()

# -------------------------
# SIDEBAR: GLOBAL FILTERS
# -------------------------
st.sidebar.title("Global Filters")

# Date filter (if available)
if "survey_date" in df.columns and df["survey_date"].notna().any():
    min_d = df["survey_date"].min()
    max_d = df["survey_date"].max()
    date_range = st.sidebar.date_input("Survey date range", value=(min_d.date(), max_d.date()))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d = pd.to_datetime(date_range[0])
        end_d = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        start_d, end_d = min_d, max_d
else:
    start_d, end_d = None, None

role_sel = st.sidebar.multiselect("Role", safe_sorted_unique(df["role"]), default=safe_sorted_unique(df["role"]))
region_sel = st.sidebar.multiselect("Region", safe_sorted_unique(df["region"]), default=safe_sorted_unique(df["region"]))
program_sel = st.sidebar.multiselect("Program Level", safe_sorted_unique(df["program_level"]), default=safe_sorted_unique(df["program_level"]))

# Apply global filters
dff = df.copy()

if start_d is not None and "survey_date" in dff.columns:
    dff = dff[(dff["survey_date"].isna()) | ((dff["survey_date"] >= start_d) & (dff["survey_date"] <= end_d))]

dff = dff[
    (dff["role"].isin(role_sel)) &
    (dff["region"].isin(region_sel)) &
    (dff["program_level"].isin(program_sel))
].copy()

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Dashboard Pages",
    ["📘 Overview", "🎓 AI Usage Patterns", "🧠 Learning • Trust • Risk", "👩‍🏫 Policy • Faculty • Misuse", "📈 Student Outcomes"]
)

# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================
if page == "📘 Overview":
    st.title("AI in Education – Overview (Easy-to-understand)")

    guard_nonempty(dff)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", f"{len(dff):,}")
    c2.metric("Avg AI use (days/week)", f"{dff['ai_use_days_per_week'].mean():.2f}")
    c3.metric("Avg Trust (1–5)", f"{dff['trust_in_ai_outputs_1to5'].mean():.2f}")
    c4.metric("Avg Learning Benefit (1–5)", f"{dff['perceived_learning_benefit_1to5'].mean():.2f}")

    st.divider()

    # 1) Role distribution (donut)
    role_counts = vc_df(dff, "role")
    fig1 = px.pie(role_counts, names="role", values="count", hole=0.45, title="1) Who answered the survey? (Role split)")
    st.plotly_chart(fig1, use_container_width=True)
    top_role = role_counts.iloc[0]["role"] if len(role_counts) else "N/A"
    insight_box([
        f"Most responses are from **{top_role}**, so their experience influences overall results.",
        "Having both Student and Faculty responses helps compare real classroom impact."
    ])

    # 2) Adoption category bar (SAFE)
    adopt_counts = vc_df(dff, "ai_adoption_category")
    fig2 = px.bar(
        adopt_counts, x="ai_adoption_category", y="count", text="pct",
        title="2) AI Adoption Levels (how strongly people use AI)"
    )
    fig2.update_traces(texttemplate="%{text}%", textposition="outside")
    fig2.update_layout(xaxis_title="Adoption category", yaxis_title="Respondents")
    st.plotly_chart(fig2, use_container_width=True)
    if len(adopt_counts):
        insight_box([
            f"The biggest group is **{adopt_counts.iloc[0]['ai_adoption_category']}** users.",
            "If adoption is high, institutions need clearer rules (what is allowed vs not allowed)."
        ])

    # 3) AI literacy histogram (simpler bins)
    fig3 = px.histogram(dff, x="ai_literacy_1to5", nbins=5, title="3) AI Literacy (1=low, 5=high)")
    fig3.update_layout(xaxis_title="AI literacy rating", yaxis_title="People")
    st.plotly_chart(fig3, use_container_width=True)
    insight_box([
        "Higher AI literacy usually means people can use AI more responsibly.",
        "Low literacy groups need training: prompts, verification, and ethical use."
    ])

    # 4) Internet quality by region (mean)
    iq = mean_by(dff, "region", "internet_quality_1to5")
    fig4 = px.bar(iq, x="region", y="internet_quality_1to5", title="4) Average Internet Quality by Region (1–5)")
    fig4.update_layout(yaxis_title="Avg internet quality", xaxis_title="Region")
    st.plotly_chart(fig4, use_container_width=True)
    if len(iq):
        worst = iq.sort_values("internet_quality_1to5").iloc[0]["region"]
        best = iq.sort_values("internet_quality_1to5", ascending=False).iloc[0]["region"]
        insight_box([
            f"**{best}** shows stronger internet quality → easier AI adoption.",
            f"**{worst}** may struggle with AI-based learning due to connectivity."
        ])

    # 5) Discipline mix (top 10)
    disc_counts = vc_df(dff, "discipline", top_n=10)
    fig5 = px.bar(disc_counts, x="discipline", y="count", text="pct", title="5) Top Disciplines in the Survey (Top 10)")
    fig5.update_traces(texttemplate="%{text}%", textposition="outside")
    fig5.update_layout(xaxis_title="Discipline", yaxis_title="Respondents")
    st.plotly_chart(fig5, use_container_width=True)
    insight_box([
        "Different disciplines use AI differently (coding vs writing vs research).",
        "Policy should be flexible by discipline, not one rule for all."
    ])

    page_conclusion(
        "Overview Conclusion",
        bullets=[
            "AI usage is mainstream in education (not a niche activity).",
            "Infrastructure (internet quality) affects equal access to AI benefits.",
            "AI literacy varies — some groups will need help to use AI responsibly."
        ],
        actions=[
            "Run short AI literacy workshops (how to prompt + how to verify answers).",
            "Support low-connectivity regions with better access (labs, Wi-Fi, campus resources).",
            "Create policy templates by discipline (writing-heavy vs coding-heavy programs)."
        ]
    )

# =========================================================
# PAGE 2 — AI USAGE PATTERNS
# =========================================================
elif page == "🎓 AI Usage Patterns":
    st.title("AI Usage Patterns (How people use AI day-to-day)")
    guard_nonempty(dff)

    # local filter (NOT global)
    st.caption("Local filter (only affects this page): choose a single tool to explore patterns.")
    tool_list = ["All"] + safe_sorted_unique(dff["primary_ai_tool"])
    tool_pick = st.selectbox("Primary AI tool", tool_list, index=0)
    dpage = dff.copy()
    if tool_pick != "All":
        dpage = dpage[dpage["primary_ai_tool"] == tool_pick]

    guard_nonempty(dpage, "No records for this tool. Select another tool or choose 'All'.")

    # 1) Usage frequency
    fig1 = px.histogram(dpage, x="ai_use_days_per_week", nbins=7, title="1) AI Use Frequency (days per week)")
    fig1.update_layout(xaxis_title="Days per week", yaxis_title="People")
    st.plotly_chart(fig1, use_container_width=True)
    insight_box([
        "If most users are at 5–7 days/week → AI is part of daily academic life.",
        "If most users are 1–3 days/week → AI is used for specific tasks only."
    ])

    # 2) Time saved
    fig2 = px.box(dpage, y="time_saved_hours_per_week", title="2) Time Saved per Week using AI (hours)")
    fig2.update_layout(yaxis_title="Hours saved per week")
    st.plotly_chart(fig2, use_container_width=True)
    insight_box([
        "Higher time saved usually means AI is used for writing, summarization, or quick prep.",
        "Very high outliers can indicate heavy dependence or misuse risk."
    ])

    # 3) Most used tools (counts)
    tool_counts = vc_df(dff, "primary_ai_tool", top_n=12)  # use full dff for overview
    fig3 = px.bar(tool_counts, x="primary_ai_tool", y="count", text="pct", title="3) Most Used Primary AI Tools (Top 12)")
    fig3.update_traces(texttemplate="%{text}%", textposition="outside")
    fig3.update_layout(xaxis_title="Tool", yaxis_title="Users")
    st.plotly_chart(fig3, use_container_width=True)
    insight_box([
        "Popular tools are usually general-purpose (writing + reasoning + coding).",
        "Tool popularity helps institutions decide what to train students on."
    ])

    # 4) Top use cases (top 10)
    use_counts = vc_df(dff, "top_ai_use_cases", top_n=10)
    fig4 = px.bar(use_counts, x="top_ai_use_cases", y="count", text="pct", title="4) Top AI Use Cases (Top 10)")
    fig4.update_traces(texttemplate="%{text}%", textposition="outside")
    fig4.update_layout(xaxis_title="Use case", yaxis_title="Users")
    st.plotly_chart(fig4, use_container_width=True)
    insight_box([
        "Assignments + exam prep + concept learning are usually the biggest categories.",
        "If 'interview prep' is high, AI is supporting employability skills too."
    ])

    # 5) Assessment impacted (counts)
    assess_counts = vc_df(dff, "assessment_context_most_impacted")
    fig5 = px.bar(assess_counts, x="assessment_context_most_impacted", y="count", text="pct",
                  title="5) Assessment Areas Most Impacted by AI")
    fig5.update_traces(texttemplate="%{text}%", textposition="outside")
    fig5.update_layout(xaxis_title="Assessment context", yaxis_title="Users")
    st.plotly_chart(fig5, use_container_width=True)
    insight_box([
        "If take-home/writing assessments dominate → higher chance of AI-assisted submissions.",
        "Institutions may need more viva/oral checks or applied problem tasks."
    ])

    # 6) Usage vs time saved (easy scatter)
    fig6 = px.scatter(
    dpage,
    x="ai_use_days_per_week",
    y="time_saved_hours_per_week",
    title="6) More AI days → More time saved? (with trend line)"
    )
    fig6 = add_linear_trendline(fig6, dpage, "ai_use_days_per_week", "time_saved_hours_per_week", "Trend line")
    st.plotly_chart(fig6, use_container_width=True)

    fig6.update_layout(xaxis_title="Days per week using AI", yaxis_title="Hours saved per week")
    st.plotly_chart(fig6, use_container_width=True)
    insight_box([
        "If the trend line goes up → more usage is linked to more time saved.",
        "If the trend is flat → people may use AI frequently but for small tasks."
    ])

    page_conclusion(
        "Usage Patterns Conclusion",
        bullets=[
            "AI is frequently used across the week and saves measurable time.",
            "The most impacted assessments are usually writing/take-home tasks.",
            "Popular tools and use-cases show where training should focus."
        ],
        actions=[
            "Update assessment design: add applied tasks + in-class components + short vivas.",
            "Teach students how to use AI for learning (explain, quiz, feedback), not copying.",
            "Create tool-specific guidance: how to cite AI help, how to verify outputs."
        ]
    )

# =========================================================
# PAGE 3 — LEARNING • TRUST • RISK
# =========================================================
elif page == "🧠 Learning • Trust • Risk":
    st.title("Learning Impact, Trust & Risk (Simple explanations)")

    guard_nonempty(dff)

    # local filter
    role_local = st.selectbox("Local view: focus role", ["All"] + safe_sorted_unique(dff["role"]), index=0)
    dp = dff.copy()
    if role_local != "All":
        dp = dp[dp["role"] == role_local]
    guard_nonempty(dp, "No data for this role. Try 'All'.")

    # 1) Learning benefit vs usage (mean)
    lb = mean_by(dp, "ai_use_days_per_week", "perceived_learning_benefit_1to5")
    fig1 = px.line(lb, x="ai_use_days_per_week", y="perceived_learning_benefit_1to5", markers=True,
                   title="1) Learning Benefit vs AI Usage Frequency")
    fig1.update_layout(xaxis_title="Days/week using AI", yaxis_title="Avg learning benefit (1–5)")
    st.plotly_chart(fig1, use_container_width=True)
    insight_box([
        "Often, benefit increases up to a point (balanced use).",
        "Too much use can reduce real learning if students stop practicing themselves."
    ])

    # 2) Dependency risk vs usage (mean)
    dep = mean_by(dp, "ai_use_days_per_week", "perceived_dependency_risk_1to5")
    fig2 = px.line(dep, x="ai_use_days_per_week", y="perceived_dependency_risk_1to5", markers=True,
                   title="2) Dependency Risk vs AI Usage Frequency")
    fig2.update_layout(xaxis_title="Days/week using AI", yaxis_title="Avg dependency risk (1–5)")
    st.plotly_chart(fig2, use_container_width=True)
    insight_box([
        "Higher usage tends to increase dependency risk.",
        "Goal: use AI to support thinking, not replace thinking."
    ])

    # 3) Trust distribution
    fig3 = px.histogram(dp, x="trust_in_ai_outputs_1to5", nbins=5, title="3) Trust in AI Outputs (1–5)")
    fig3.update_layout(xaxis_title="Trust score", yaxis_title="People")
    st.plotly_chart(fig3, use_container_width=True)
    insight_box([
        "Moderate trust is healthy: trust AI but verify.",
        "Very high trust without verification increases misuse risk."
    ])

    # 4) Verification vs literacy (mean)
    ver = mean_by(dp, "ai_literacy_1to5", "verifies_ai_outputs_rate_0to1")
    fig4 = px.bar(ver, x="ai_literacy_1to5", y="verifies_ai_outputs_rate_0to1",
                  title="4) Higher AI Literacy → More Verification?")
    fig4.update_layout(xaxis_title="AI literacy (1–5)", yaxis_title="Avg verification rate (0–1)")
    st.plotly_chart(fig4, use_container_width=True)
    insight_box([
        "Training works: literacy is linked to verification behavior.",
        "Teach students to check sources, numbers, and logic."
    ])

    # 5) Fairness/bias concerns
    fig5 = px.histogram(dp, x="fairness_equity_concern_1to5", nbins=5, title="5) Fairness & Bias Concern (1–5)")
    fig5.update_layout(xaxis_title="Concern level", yaxis_title="People")
    st.plotly_chart(fig5, use_container_width=True)
    insight_box([
        "If concern is high, users are worried about bias (gender, region, language).",
        "Institutions should discuss fairness and inclusive AI usage."
    ])

    # 6) Cheating risk vs policy awareness (mean)
    cr = mean_by(dp, "institution_policy_awareness", "perceived_cheating_risk_1to5")
    fig6 = px.bar(cr, x="institution_policy_awareness", y="perceived_cheating_risk_1to5",
                  title="6) Policy Awareness vs Perceived Cheating Risk")
    fig6.update_layout(xaxis_title="Policy awareness", yaxis_title="Avg cheating risk (1–5)")
    st.plotly_chart(fig6, use_container_width=True)
    insight_box([
        "When people know the policy, they understand boundaries better.",
        "Clear policy reduces confusion and misuse."
    ])

    page_conclusion(
        "Learning • Trust • Risk Conclusion",
        bullets=[
            "Balanced AI use gives strong learning benefits with manageable risk.",
            "Higher literacy increases verification habits.",
            "Clear policy awareness supports safer usage and reduces cheating pressure."
        ],
        actions=[
            "Add a short module: verification checklist + examples of AI mistakes.",
            "Encourage “AI-assisted learning”: explain concepts, generate quizzes, get feedback.",
            "Standardize policy communication: orientation + course syllabus + LMS announcements."
        ]
    )

# =========================================================
# PAGE 4 — POLICY • FACULTY • MISUSE
# =========================================================
elif page == "👩‍🏫 Policy • Faculty • Misuse":
    st.title("Policy, Faculty Training & Misuse (Institution view)")
    guard_nonempty(dff)

    faculty = dff[dff["role"] == "Faculty"].copy()
    if len(faculty) == 0:
        st.warning("No Faculty records under current global filters. Add 'Faculty' in Role filter.")
        st.stop()

    # local filter
    stance_local = st.selectbox(
        "Local filter: Institution AI stance",
        ["All"] + safe_sorted_unique(faculty["institution_ai_stance"]),
        index=0
    )
    fp = faculty.copy()
    if stance_local != "All":
        fp = fp[fp["institution_ai_stance"] == stance_local]
    guard_nonempty(fp, "No data for this institution stance. Choose another option.")

    # 1) Policy stance distribution
    stance_counts = vc_df(faculty, "institution_ai_stance")
    fig1 = px.pie(stance_counts, names="institution_ai_stance", values="count", hole=0.45,
                  title="1) Institutional AI Stance (from Faculty perspective)")
    st.plotly_chart(fig1, use_container_width=True)
    insight_box([
        "If many are 'Unclear/Developing' → policy is not settled yet.",
        "Clear stance helps reduce confusion and misuse."
    ])

    # 2) Policy awareness distribution (all roles)
    aware_counts = vc_df(dff, "institution_policy_awareness")
    fig2 = px.bar(aware_counts, x="institution_policy_awareness", y="count", text="pct",
                  title="2) Policy Awareness (Do people know the rules?)")
    fig2.update_traces(texttemplate="%{text}%", textposition="outside")
    fig2.update_layout(xaxis_title="Awareness level", yaxis_title="People")
    st.plotly_chart(fig2, use_container_width=True)
    insight_box([
        "Low awareness → people may misuse AI unintentionally.",
        "High awareness usually reduces confusion in assignments."
    ])

    # 3) Misuse rate vs policy awareness
    misuse = mean_by(dff, "institution_policy_awareness", "plagiarism_or_misuse_flag")
    fig3 = px.bar(misuse, x="institution_policy_awareness", y="plagiarism_or_misuse_flag",
                  title="3) Misuse Flag Rate vs Policy Awareness")
    fig3.update_layout(xaxis_title="Policy awareness", yaxis_title="Avg misuse flag (0–1)")
    st.plotly_chart(fig3, use_container_width=True)
    insight_box([
        "If misuse drops with awareness → communication is working.",
        "If misuse stays high → policy exists but enforcement/assessment design needs improvement."
    ])

    # 4) Faculty training hours distribution
    fig4 = px.histogram(fp, x="ai_training_hours_last_6_months", nbins=20,
                        title="4) Faculty AI Training Hours (Last 6 months)")
    fig4.update_layout(xaxis_title="Training hours", yaxis_title="Faculty count")
    st.plotly_chart(fig4, use_container_width=True)
    insight_box([
        "Higher training means faculty can design better AI-resistant assessments.",
        "Low training suggests faculty need structured workshops."
    ])

    # 5) Reported misuse cases by institution type
    inst_misuse = mean_by(fp, "institution_type", "staff_reported_ai_misuse_cases_last_term")
    fig5 = px.bar(inst_misuse, x="institution_type", y="staff_reported_ai_misuse_cases_last_term",
                  title="5) Reported AI Misuse Cases (Last term) by Institution Type")
    fig5.update_layout(xaxis_title="Institution type", yaxis_title="Avg reported misuse cases")
    st.plotly_chart(fig5, use_container_width=True)
    insight_box([
        "Higher reported cases can mean either more misuse OR better detection/reporting.",
        "Look at training + policy awareness together for the full story."
    ])

    # 6) Trust vs training (scatter)
    fig6 = px.scatter(
        fp, x="ai_training_hours_last_6_months", y="trust_in_ai_outputs_1to5",
        trendline="ols", title="6) Faculty Training vs Trust in AI (trend line)"
    )
    fig6.update_layout(xaxis_title="Training hours (last 6 months)", yaxis_title="Trust in AI (1–5)")
    st.plotly_chart(fig6, use_container_width=True)
    insight_box([
        "More training can lead to more realistic trust (not blind trust).",
        "Training helps faculty understand strengths + limitations."
    ])

    page_conclusion(
        "Policy • Faculty • Misuse Conclusion",
        bullets=[
            "Clear policy awareness is linked to lower misuse.",
            "Faculty training supports better assessment design and governance.",
            "Reported misuse cases must be interpreted carefully (misuse vs detection)."
        ],
        actions=[
            "Create a simple AI policy: allowed use + citation rules + consequences.",
            "Train faculty to design applied assessments and short vivas/orals.",
            "Set up reporting + academic integrity workflows (consistent across departments)."
        ]
    )

# =========================================================
# PAGE 5 — STUDENT OUTCOMES
# =========================================================
elif page == "📈 Student Outcomes":
    st.title("Student Outcomes & Adoption (Grades • Stress • Thinking)")
    guard_nonempty(dff)

    students = dff[dff["role"] == "Student"].copy()
    if len(students) == 0:
        st.warning("No Student records under current global filters. Add 'Student' in Role filter.")
        st.stop()

    # local filter
    adoption_local = st.multiselect(
        "Local filter: choose adoption categories (this page only)",
        safe_sorted_unique(students["ai_adoption_category"]),
        default=safe_sorted_unique(students["ai_adoption_category"])
    )
    sp = students[students["ai_adoption_category"].isin(adoption_local)].copy()
    guard_nonempty(sp, "No student data for selected adoption categories.")

    # 1) Grade change by adoption
    g1 = mean_by(sp, "ai_adoption_category", "student_grade_change_pct")
    fig1 = px.bar(g1, x="ai_adoption_category", y="student_grade_change_pct",
                  title="1) Average Grade Change (%) by AI Adoption Category")
    fig1.update_layout(xaxis_title="Adoption category", yaxis_title="Avg grade change (%)")
    st.plotly_chart(fig1, use_container_width=True)
    insight_box([
        "If moderate adoption has best grade change → balanced AI use improves learning.",
        "If high adoption does not improve grades → possible over-dependence."
    ])

    # 2) Critical thinking change by adoption
    g2 = mean_by(sp, "ai_adoption_category", "student_critical_thinking_change_index")
    fig2 = px.bar(g2, x="ai_adoption_category", y="student_critical_thinking_change_index",
                  title="2) Critical Thinking Change by AI Adoption Category")
    fig2.update_layout(xaxis_title="Adoption category", yaxis_title="Avg critical thinking change index")
    st.plotly_chart(fig2, use_container_width=True)
    insight_box([
        "Critical thinking can improve if AI is used for practice and feedback.",
        "It can drop if AI is used to avoid solving problems independently."
    ])

    # 3) Stress change by adoption
    g3 = mean_by(sp, "ai_adoption_category", "student_stress_change_index")
    fig3 = px.bar(g3, x="ai_adoption_category", y="student_stress_change_index",
                  title="3) Stress Change Index by Adoption Category")
    fig3.update_layout(xaxis_title="Adoption category", yaxis_title="Avg stress change index")
    st.plotly_chart(fig3, use_container_width=True)
    insight_box([
        "AI can reduce stress by saving time and giving guidance.",
        "But constant AI use can increase stress due to confusion, dependency, or policy fear."
    ])

    # 4) Usage vs grade change scatter
    fig4 = px.scatter(
        sp, x="ai_use_days_per_week", y="student_grade_change_pct",
        trendline="ols", title="4) AI Usage Frequency vs Grade Change (trend line)"
    )
    fig4.update_layout(xaxis_title="Days/week using AI", yaxis_title="Grade change (%)")
    st.plotly_chart(fig4, use_container_width=True)
    insight_box([
        "If the trend is positive → more AI use is linked with improved grades.",
        "If trend is flat/negative → students may use AI but not learn better."
    ])

    # 5) Dependency risk vs grade change
    fig5 = px.scatter(
        sp, x="perceived_dependency_risk_1to5", y="student_grade_change_pct",
        trendline="ols", title="5) Dependency Risk vs Grade Change (trend line)"
    )
    fig5.update_layout(xaxis_title="Dependency risk (1–5)", yaxis_title="Grade change (%)")
    st.plotly_chart(fig5, use_container_width=True)
    insight_box([
        "Higher dependency risk can harm long-term learning, even if grades rise short-term.",
        "Teach students to use AI as a tutor, not an answer machine."
    ])

    # 6) Cheating risk vs misuse flag
    g6 = mean_by(sp, "perceived_cheating_risk_1to5", "plagiarism_or_misuse_flag")
    fig6 = px.line(g6, x="perceived_cheating_risk_1to5", y="plagiarism_or_misuse_flag", markers=True,
                   title="6) Perceived Cheating Risk vs Misuse Flag Rate")
    fig6.update_layout(xaxis_title="Cheating risk (1–5)", yaxis_title="Avg misuse flag (0–1)")
    st.plotly_chart(fig6, use_container_width=True)
    insight_box([
        "If misuse increases with cheating risk → students feel pressure or unclear rules.",
        "Strong guidance and assessment redesign can reduce misuse."
    ])

    page_conclusion(
        "Student Outcomes Conclusion",
        bullets=[
            "Balanced AI adoption often performs best across grades and skills.",
            "Very high adoption may increase dependency risk and can reduce true skill growth.",
            "Outcomes improve when students verify outputs and practice thinking themselves."
        ],
        actions=[
            "Promote balanced usage: AI for feedback + quizzes + explanations.",
            "Add assignments that require personal reflection, drafts, and oral checks.",
            "Teach citation + verification + academic integrity practices."
        ]
    )
