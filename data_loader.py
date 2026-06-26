"""
data_loader.py
--------------
Fetches free, open football match-event data from the StatsBomb Open Data
repository on GitHub. No API key, no payment, no rate limits beyond GitHub's
generous public limits.

StatsBomb Open Data license: free for non-commercial use (perfect for a
hackathon prototype). Repo: https://github.com/statsbomb/open-data

Key concepts:
- A "competition" (e.g. FIFA World Cup) has many "matches".
- Each match has an "events" file: every pass, shot, tackle, etc., with
  location coordinates, timestamps, players and outcomes.

We pull events for a single match and reduce them into a compact, structured
summary that we can feed to IBM Granite for natural-language tactical analysis.
"""

import json
import requests
from collections import defaultdict
from typing import Dict, List, Any

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def get_competitions() -> List[Dict[str, Any]]:
    """Return every competition/season available in the open dataset."""
    url = f"{BASE}/competitions.json"
    return requests.get(url, timeout=30).json()


def find_world_cup_seasons() -> List[Dict[str, Any]]:
    """Filter the competition list down to FIFA World Cup entries."""
    comps = get_competitions()
    return [
        c for c in comps
        if "world cup" in c["competition_name"].lower()
    ]


def get_matches(competition_id: int, season_id: int) -> List[Dict[str, Any]]:
    """List all matches for a given competition + season."""
    url = f"{BASE}/matches/{competition_id}/{season_id}.json"
    return requests.get(url, timeout=30).json()


def get_events(match_id: int) -> List[Dict[str, Any]]:
    """Pull the full event stream for a single match."""
    url = f"{BASE}/events/{match_id}.json"
    return requests.get(url, timeout=60).json()


def summarize_match(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reduce a raw event stream (often 3,000+ events) into a compact summary
    that fits comfortably inside an LLM prompt.

    We extract, per team:
      - total passes and pass completion %
      - shots and shots on target
      - pressures (a proxy for pressing intensity)
      - a minute-by-minute tally of pressures so we can see when a team's
        press faded — the kind of insight a tactical analyst cares about.
    """
    teams = defaultdict(lambda: {
        "passes": 0,
        "passes_completed": 0,
        "shots": 0,
        "shots_on_target": 0,
        "pressures": 0,
        "pressures_by_15min": defaultdict(int),
    })

    for ev in events:
        t = ev.get("team", {}).get("name")
        if not t:
            continue
        etype = ev.get("type", {}).get("name", "")
        minute = ev.get("minute", 0)
        bucket = (minute // 15) * 15  # 0,15,30,45,60,75...

        if etype == "Pass":
            teams[t]["passes"] += 1
            # In StatsBomb data, an incomplete pass carries an outcome;
            # a completed pass has no outcome field.
            outcome = ev.get("pass", {}).get("outcome")
            if outcome is None:
                teams[t]["passes_completed"] += 1

        elif etype == "Shot":
            teams[t]["shots"] += 1
            shot_outcome = ev.get("shot", {}).get("outcome", {}).get("name", "")
            if shot_outcome in ("Goal", "Saved"):
                teams[t]["shots_on_target"] += 1

        elif etype == "Pressure":
            teams[t]["pressures"] += 1
            teams[t]["pressures_by_15min"][bucket] += 1

    # Convert defaultdicts to plain dicts and add completion %.
    clean = {}
    for team, s in teams.items():
        pct = (s["passes_completed"] / s["passes"] * 100) if s["passes"] else 0
        clean[team] = {
            "passes": s["passes"],
            "pass_completion_pct": round(pct, 1),
            "shots": s["shots"],
            "shots_on_target": s["shots_on_target"],
            "pressures": s["pressures"],
            "pressures_by_15min": dict(sorted(s["pressures_by_15min"].items())),
        }
    return clean


if __name__ == "__main__":
    # Quick self-test: list World Cup seasons and summarize one match.
    print("Finding World Cup seasons in StatsBomb open data...")
    seasons = find_world_cup_seasons()
    for s in seasons:
        print(f"  {s['competition_name']} — {s['season_name']} "
              f"(comp {s['competition_id']}, season {s['season_id']})")

    if seasons:
        s = seasons[0]
        matches = get_matches(s["competition_id"], s["season_id"])
        print(f"\n{len(matches)} matches found. Using the first one.")
        m = matches[0]
        print(f"  {m['home_team']['home_team_name']} vs "
              f"{m['away_team']['away_team_name']}")
        events = get_events(m["match_id"])
        summary = summarize_match(events)
        print("\nMatch summary:")
        print(json.dumps(summary, indent=2))
