"""
app.py — TacticAI dashboard
---------------------------
A Streamlit front end that ties the whole pipeline together:

    StatsBomb data  ->  structured summary  ->  IBM Granite analysis
              (+ optional Docling-parsed scouting PDF)

Run locally:    streamlit run src/app.py
Deploy free:    push to GitHub, then deploy on Streamlit Community Cloud
                (share.streamlit.io) — zero cost.
"""

import sys
import os
import tempfile
import streamlit as st
import pandas as pd
import altair as alt

# Make sibling modules importable whether run from repo root or /src.
sys.path.append(os.path.dirname(__file__))

import data_loader as dl
import granite_engine as ge
import docling_parser as dp


st.set_page_config(page_title="TacticAI", page_icon="⚽", layout="wide")


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("⚽ TacticAI")
st.caption(
    "Plain-language football tactical analysis, powered by IBM Granite, "
    "Docling and open World Cup data. Built for grassroots coaches who "
    "don't have a pro analytics department."
)


# --------------------------------------------------------------------------
# Sidebar: pick a competition and match
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Choose a match")

    @st.cache_data(show_spinner=False)
    def load_seasons():
        seasons = dl.find_world_cup_seasons()
        # Order so the men's FIFA World Cup comes first, newest year at the
        # top — this makes the most recognizable tournament (2022: Argentina,
        # France, etc.) the default selection. Women's and U20 cups follow.
        def sort_key(s):
            name = s["competition_name"]
            # Lower number = higher in the list.
            if name == "FIFA World Cup":
                tier = 0
            elif name == "Women's World Cup":
                tier = 1
            else:
                tier = 2
            # Newest season first within each tier.
            year = s["season_name"]
            return (tier, "0000" if not year[:4].isdigit() else
                    str(9999 - int(year[:4])))
        return sorted(seasons, key=sort_key)

    seasons = load_seasons()
    season_labels = [
        f"{s['competition_name']} — {s['season_name']}" for s in seasons
    ]
    # Default to the first entry, which is now the most recent men's World Cup.
    chosen = st.selectbox("Competition / season", season_labels, index=0)
    season = seasons[season_labels.index(chosen)]

    @st.cache_data(show_spinner=False)
    def load_matches(cid, sid):
        return dl.get_matches(cid, sid)

    matches = load_matches(season["competition_id"], season["season_id"])
    match_labels = [
        f"{m['home_team']['home_team_name']} vs "
        f"{m['away_team']['away_team_name']}"
        for m in matches
    ]
    chosen_match = st.selectbox("Match", match_labels)
    match = matches[match_labels.index(chosen_match)]

    st.header("2. Optional: add a scouting PDF")
    st.caption("Docling will parse it and fold it into the analysis.")
    uploaded = st.file_uploader("Scouting report", type=["pdf", "docx", "pptx"])

    run = st.button("Analyze match", type="primary", use_container_width=True)


# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------
if run:
    home = match["home_team"]["home_team_name"]
    away = match["away_team"]["away_team_name"]

    with st.spinner("Pulling event data from StatsBomb open data..."):
        events = dl.get_events(match["match_id"])
        summary = dl.summarize_match(events)

    # --- Headline metrics -------------------------------------------------
    st.subheader(f"{home} vs {away}")
    teams = list(summary.keys())
    cols = st.columns(len(teams))
    for col, team in zip(cols, teams):
        s = summary[team]
        with col:
            st.markdown(f"**{team}**")
            st.metric("Pass completion", f"{s['pass_completion_pct']}%")
            st.metric("Shots (on target)",
                      f"{s['shots']} ({s['shots_on_target']})")
            st.metric("Pressures", s["pressures"])

    # --- Pressing-over-time chart ----------------------------------------
    st.markdown("#### Pressing intensity over the match")
    st.caption(
        "Each bar is the number of pressing actions in a 15-minute window. "
        "A falling line means a team's press is fading — often where matches "
        "are won or lost."
    )
    rows = []
    for team in teams:
        for bucket, count in summary[team]["pressures_by_15min"].items():
            rows.append({"Team": team,
                         "Minute": f"{bucket}-{int(bucket)+15}",
                         "MinuteSort": int(bucket),
                         "Pressures": count})
    if rows:
        df = pd.DataFrame(rows).sort_values("MinuteSort")
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("Minute:N", sort=None, title="Match minute"),
                y=alt.Y("Pressures:Q", title="Pressing actions"),
                color=alt.Color("Team:N"),
                tooltip=["Team", "Minute", "Pressures"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

    # --- Optional scouting doc via Docling -------------------------------
    scouting = ""
    if uploaded is not None:
        with st.spinner("Parsing scouting document with Docling..."):
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            scouting = dp.extract_scouting_notes(tmp_path)
            os.unlink(tmp_path)
        if scouting:
            with st.expander("Parsed scouting notes (from Docling)"):
                st.text(scouting)

    # --- Granite analysis -------------------------------------------------
    st.markdown("#### Tactical analysis")
    with st.spinner("IBM Granite is analyzing the match..."):
        if scouting:
            summary_for_prompt = dict(summary)
            summary_for_prompt["_scouting_notes"] = scouting
            analysis = ge.analyze(home, away, summary_for_prompt)
        else:
            analysis = ge.analyze(home, away, summary)
    st.write(analysis)

    st.divider()
    st.caption(
        "Data: StatsBomb Open Data (free, non-commercial). "
        "Analysis: IBM Granite. Document parsing: IBM Docling."
    )

else:
    st.info(
        "Pick a competition and match in the sidebar, then click "
        "**Analyze match**. Optionally upload a scouting PDF and Docling "
        "will fold it into the analysis."
    )