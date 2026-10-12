"""SkillScope dashboard – reads the Gold-layer CSVs exported by notebooks/SkillScope_BDA.ipynb."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# CSVs can sit in a data/ subfolder or next to app.py (GitHub web upload may flatten folders)
HERE = Path(__file__).parent
DATA = HERE / "data" if (HERE / "data" / "kpis.json").exists() else HERE
st.set_page_config(page_title="SkillScope – Job Skill Demand", page_icon="📊", layout="wide")


@st.cache_data
def load(name):
    return pd.read_csv(DATA / f"{name}.csv")


kpis = json.loads((DATA / "kpis.json").read_text())
skills = load("skills_by_group")
postings = load("postings_by_group")

st.title("SkillScope: Job-Market Skill Demand Analytics")
st.caption(f"Apache Spark {kpis['spark_version']} pipeline on {kpis['silver_postings']:,} LinkedIn job postings "
           f"({kpis['date_min']} to {kpis['date_max']})")

c = st.columns(5)
c[0].metric("Job postings", f"{kpis['silver_postings']:,}")
c[1].metric("Job-skill records", f"{kpis['job_skill_rows']:,}")
c[2].metric("Distinct skills", f"{kpis['distinct_skills']:,}")
c[3].metric("Companies", f"{kpis['companies']:,}")
c[4].metric("Countries", kpis["countries"])

# ---- filters ----
st.sidebar.header("Filters")
roles = sorted(skills["role_family"].dropna().unique())
countries = sorted(skills["country"].dropna().unique())
levels = sorted(skills["job_level"].dropna().unique())
default_role = ["Data Analyst"] if "Data Analyst" in roles else roles[:1]
sel_roles = st.sidebar.multiselect("Role family", roles, default=default_role)
sel_countries = st.sidebar.multiselect("Country", countries, default=countries)
sel_levels = st.sidebar.multiselect("Job level", levels, default=levels)
top_n = st.sidebar.slider("Top N skills", 5, 30, 15)

f_sk = skills[skills.role_family.isin(sel_roles) & skills.country.isin(sel_countries) & skills.job_level.isin(sel_levels)]
f_po = postings[postings.role_family.isin(sel_roles) & postings.country.isin(sel_countries) & postings.job_level.isin(sel_levels)]
n_post = int(f_po["postings"].sum())

tab1, tab2, tab3, tab4 = st.tabs(["Skill demand", "Market overview", "Skill combinations", "Big Data tests"])

with tab1:
    st.subheader(f"Top {top_n} skills for the selected filters ({n_post:,} postings)")
    if f_sk.empty:
        st.info("No data for this selection.")
    else:
        agg = f_sk.groupby("skill", as_index=False)["postings"].sum().nlargest(top_n, "postings")
        agg["% of postings"] = (100 * agg["postings"] / max(n_post, 1)).round(1)
        fig = px.bar(agg.sort_values("postings"), x="% of postings", y="skill", orientation="h",
                     hover_data=["postings"], color="% of postings", color_continuous_scale="Blues")
        fig.update_layout(height=30 * top_n + 120, coloraxis_showscale=False, yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Skill-gap check: Data Analyst profile")
    gap = load("skill_gap")
    gap["have"] = gap["have"].map({"Yes": "✅ Have", "Gap": "❌ Gap"})
    # translucent tints keep the cell text readable in both light and dark themes
    tint = lambda v: ("background-color: rgba(229, 62, 62, 0.28); font-weight: 600" if "Gap" in v
                      else "background-color: rgba(56, 161, 105, 0.22)")
    st.dataframe(gap.style.map(tint, subset=["have"]), width="stretch", hide_index=True,
                 column_config={"skill": "Skill", "postings": st.column_config.NumberColumn("Postings", format="%d"),
                                "pct_of_da_postings": st.column_config.NumberColumn("% of Data Analyst postings", format="%.1f%%"),
                                "have": "In my profile?"})

with tab2:
    a, b = st.columns(2)
    wt = f_po.groupby("job_type", as_index=False)["postings"].sum()
    a.plotly_chart(px.pie(wt, names="job_type", values="postings", hole=0.45, title="Work type (selected filters)"),
                   width="stretch")
    rf = load("role_family_counts")
    b.plotly_chart(px.bar(rf.sort_values("count"), x="count", y="role_family", orientation="h",
                          title="Postings by role family (all data)"), width="stretch")
    a, b = st.columns(2)
    comp = load("top_companies").head(15)
    a.plotly_chart(px.bar(comp.sort_values("postings"), x="postings", y="company", orientation="h",
                          title="Top hiring companies"), width="stretch")
    cities = load("top_cities").head(15)
    cities["place"] = cities["city"] + ", " + cities["state"].fillna("")
    b.plotly_chart(px.bar(cities.sort_values("postings"), x="postings", y="place", orientation="h",
                          title="Top hiring cities"), width="stretch")
    daily = load("daily_trend")
    st.plotly_chart(px.line(daily, x="first_seen", y="postings", markers=True, title="Daily new postings"),
                    width="stretch")

with tab3:
    st.subheader("Most frequent skill pairs in data roles")
    pairs = load("skill_pairs").head(20)
    pairs["pair"] = pairs["skill_a"] + " + " + pairs["skill_b"]
    st.plotly_chart(px.bar(pairs.sort_values("postings"), x="postings", y="pair", orientation="h", height=600),
                    width="stretch")
    st.subheader("FP-Growth association rules (Spark MLlib)")
    st.caption("Read as: postings that ask for the *antecedent* skills also ask for the *consequent* with this confidence. "
               "Lift > 1 means the skills appear together more often than by chance.")
    st.dataframe(load("assoc_rules"), width="stretch", hide_index=True)

with tab4:
    st.subheader("Volume: execution time vs data size")
    vol = load("volume_test")
    a, b = st.columns([3, 2])
    a.plotly_chart(px.line(vol, x="job_skill_rows", y="seconds", text="fraction", markers=True), width="stretch")
    b.dataframe(vol, hide_index=True, width="stretch")
    st.subheader("Velocity: Structured Streaming micro-batches")
    sl = load("stream_log")
    st.plotly_chart(px.bar(sl, x="batch", y="input_rows", hover_data=["top_3"], title="Postings per micro-batch"),
                    width="stretch")
    st.dataframe(sl, hide_index=True, width="stretch")
    if (DATA / "variety.csv").exists():
        st.subheader("Variety: data types handled")
        st.dataframe(load("variety"), hide_index=True, width="stretch")
