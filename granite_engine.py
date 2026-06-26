"""
granite_engine.py
-----------------
Turns the structured match summary into natural-language tactical insight
using IBM Granite.

This module is deliberately written so Granite can be accessed in whichever
FREE way is available to you. It tries, in order:

  1. IBM watsonx.ai  (free trial tier)         -> if WATSONX_* env vars set
  2. Hugging Face Inference API (free tier)     -> if HF_TOKEN env var set
  3. A local Granite model via transformers     -> if you have it downloaded
  4. A clearly-labelled offline fallback         -> rule-based, so the app
     still runs end-to-end with no credentials at all (great for demos /
     when wifi is flaky on stage).

Granite model used: ibm-granite/granite-3.3-2b-instruct
  - Small enough to run on a free Colab GPU or a decent laptop.
  - Instruction-tuned, so it follows our analysis prompt well.
"""

import os
import json
from typing import Dict, Any

GRANITE_MODEL = "ibm-granite/granite-3.3-2b-instruct"

SYSTEM_PROMPT = (
    "You are a professional football (soccer) tactical analyst. "
    "You explain matches in clear, concrete language a grassroots coach "
    "can act on. You avoid jargon unless you immediately explain it. "
    "You focus on pressing, possession, attacking threat and momentum "
    "shifts, and you always tie observations to a practical recommendation."
)


def build_prompt(home: str, away: str, summary: Dict[str, Any]) -> str:
    """Construct the user prompt from the structured match summary."""
    return (
        f"Analyze this match between {home} and {away}.\n\n"
        f"Match data (per team):\n{json.dumps(summary, indent=2)}\n\n"
        "Write a tactical analysis with exactly these four short sections:\n"
        "1. Possession & Control — who dictated play and how we can tell.\n"
        "2. Pressing — read the pressures_by_15min numbers. Did either "
        "team's press fade, and in which part of the match?\n"
        "3. Attacking Threat — compare shots and shots on target.\n"
        "4. Coaching Takeaway — one concrete adjustment for the team that "
        "struggled.\n\n"
        "Keep each section to 2-3 sentences."
    )


def _strip_echoed_instructions(text: str) -> str:
    """
    Remove any stray instruction lines the model sometimes echoes before the
    real answer (e.g. a leading "Use a direct, instructional tone." line).
    We drop leading lines until we hit the first real section ("1.").
    """
    if not text:
        return text
    lines = text.strip().splitlines()
    # Find the first line that looks like the start of the analysis.
    for i, line in enumerate(lines):
        if line.strip().startswith("1."):
            return "\n".join(lines[i:]).strip()
    return text.strip()


# --------------------------------------------------------------------------
# Backend 1: IBM watsonx.ai (free trial)
# --------------------------------------------------------------------------
def _try_watsonx(prompt: str) -> str | None:
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    if not (api_key and project_id):
        return None
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        model = ModelInference(
            # NOTE: model availability varies by watsonx region/plan. This
            # model is confirmed available in the eu-de (Frankfurt) trial
            # environment. If you see a "model not supported" error in your
            # Space logs, swap this for any Granite model the log lists as
            # supported for your environment.
            model_id="ibm/granite-4-h-small",
            credentials=Credentials(api_key=api_key, url=url),
            project_id=project_id,
            params={"max_new_tokens": 600, "temperature": 0.4},
        )
        # Use the chat interface so the system prompt (tone/role instructions)
        # and the user prompt stay in separate roles. A plain text completion
        # can cause the model to echo instruction text like "Use a direct
        # tone" into its answer; chat formatting prevents that.
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            resp = model.chat(messages=messages)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            # Older library versions may not expose .chat(); fall back to a
            # plain completion, and strip any echoed instruction lines.
            full = f"{SYSTEM_PROMPT}\n\n{prompt}"
            text = model.generate_text(prompt=full)
            return _strip_echoed_instructions(text)
    except Exception as e:
        print(f"[watsonx backend unavailable: {e}]")
        return None


# --------------------------------------------------------------------------
# Backend 2: Hugging Face Inference API (free tier)
# --------------------------------------------------------------------------
def _try_hf_api(prompt: str) -> str | None:
    token = os.getenv("HF_TOKEN")
    if not token:
        return None
    try:
        import requests
        url = f"https://api-inference.huggingface.co/models/{GRANITE_MODEL}"
        headers = {"Authorization": f"Bearer {token}"}
        chat = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{prompt}\n<|assistant|>\n"
        )
        payload = {
            "inputs": chat,
            "parameters": {"max_new_tokens": 600, "temperature": 0.4,
                           "return_full_text": False},
        }
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").strip()
        return None
    except Exception as e:
        print(f"[HF API backend unavailable: {e}]")
        return None


# --------------------------------------------------------------------------
# Backend 3: Local transformers (free, runs on your machine / Colab)
# --------------------------------------------------------------------------
_local_pipe = None  # cached so we load the model only once


def _try_local(prompt: str) -> str | None:
    global _local_pipe
    try:
        if _local_pipe is None:
            from transformers import pipeline
            import torch
            _local_pipe = pipeline(
                "text-generation",
                model=GRANITE_MODEL,
                torch_dtype=torch.float16 if torch.cuda.is_available()
                else torch.float32,
                device_map="auto",
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        out = _local_pipe(messages, max_new_tokens=600, temperature=0.4)
        return out[0]["generated_text"][-1]["content"]
    except Exception as e:
        print(f"[local backend unavailable: {e}]")
        return None


# --------------------------------------------------------------------------
# Backend 4: Offline rule-based fallback (no credentials needed)
# --------------------------------------------------------------------------
def _offline_fallback(home: str, away: str, summary: Dict[str, Any]) -> str:
    teams = list(summary.keys())
    if len(teams) < 2:
        return "Not enough data to analyze."
    a, b = teams[0], teams[1]
    sa, sb = summary[a], summary[b]

    dominant = a if sa["pass_completion_pct"] > sb["pass_completion_pct"] else b
    other = b if dominant == a else a

    def press_fade(s):
        buckets = s["pressures_by_15min"]
        if not buckets:
            return "no pressure data"
        keys = sorted(buckets.keys(), key=lambda x: int(x))
        first = buckets[keys[0]]
        last = buckets[keys[-1]]
        if last < first * 0.6:
            return f"faded sharply (from {first} to {last} pressures)"
        return f"held steady ({first} to {last} pressures)"

    return (
        f"[Offline analysis — connect Granite for richer output]\n\n"
        f"1. Possession & Control: {dominant} controlled the ball with "
        f"{summary[dominant]['pass_completion_pct']}% pass completion versus "
        f"{summary[other]['pass_completion_pct']}% — they dictated tempo.\n\n"
        f"2. Pressing: {a}'s press {press_fade(sa)}. {b}'s press "
        f"{press_fade(sb)}. The team whose press faded gave the opponent "
        f"more time on the ball late on.\n\n"
        f"3. Attacking Threat: {a} had {sa['shots']} shots "
        f"({sa['shots_on_target']} on target); {b} had {sb['shots']} "
        f"({sb['shots_on_target']} on target).\n\n"
        f"4. Coaching Takeaway: {other} should tighten midfield spacing to "
        f"contest possession and sustain pressing intensity past the hour."
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def analyze(home: str, away: str, summary: Dict[str, Any]) -> str:
    """
    Run tactical analysis, automatically using the best available free
    backend. Always returns a string so the app never crashes on stage.
    """
    prompt = build_prompt(home, away, summary)

    for backend in (_try_watsonx, _try_hf_api, _try_local):
        result = backend(prompt)
        if result:
            return result

    return _offline_fallback(home, away, summary)


if __name__ == "__main__":
    # Demo with a tiny synthetic summary so it runs instantly with no creds.
    demo = {
        "Team A": {"passes": 525, "pass_completion_pct": 85.9, "shots": 19,
                   "shots_on_target": 8, "pressures": 116,
                   "pressures_by_15min": {"0": 30, "15": 22, "30": 18,
                                          "45": 25, "60": 12, "75": 9}},
        "Team B": {"passes": 303, "pass_completion_pct": 71.3, "shots": 11,
                   "shots_on_target": 4, "pressures": 146,
                   "pressures_by_15min": {"0": 39, "15": 21, "30": 12,
                                          "45": 41, "60": 24, "75": 9}},
    }
    print(analyze("Team A", "Team B", demo))