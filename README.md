# ⚽ TacticAI — Plain-Language Football Tactical Analysis

> **🔴 Live demo:** **[https://huggingface.co/spaces/Dakshu007/tacticai](https://huggingface.co/spaces/Dakshu007/tacticai)**
> Pick any FIFA World Cup match, click **Analyze match**, and read an IBM Granite–generated tactical breakdown in seconds.

**Built for the IBM SkillsBuild AI Builders Challenge — FIFA World Cup challenge.**

TacticAI turns raw match data into clear, actionable tactical insight that a grassroots coach can read in under a minute — no analytics department required.

---

## 🎯 The problem we are solving

Professional football clubs spend millions on tactical-analysis platforms and employ teams of analysts. Youth coaches, amateur clubs, and fans have none of that. The data exists — public match-event datasets are rich and free — but it is locked behind technical complexity and expert interpretation.

A volunteer U-15 coach can see *that* their team conceded late, but not *why*: that their pressing collapsed after the 60th minute and handed the opponent time on the ball. That gap between data and understanding is what keeps tactical insight a luxury good.

**TacticAI closes that gap.** It reads the same event data the pros use, detects momentum and pressing shifts, optionally folds in a scouting document, and explains the match in plain language — with one concrete coaching takeaway.

## 🧠 Our AI / technical approach

TacticAI is a pipeline of IBM technologies, each doing what it is best at:

1. **Data ingestion** — Real match-event streams (every pass, shot, tackle, and pressure) are pulled from StatsBomb Open Data, including FIFA World Cup matches. A reducer compresses 3,000+ raw events into a compact structured summary, including a minute-by-minute pressing tally.

2. **IBM Granite** — The structured match summary is sent to **IBM Granite** (`ibm/granite-4-h-small`, running live on IBM watsonx.ai) with a tactical-analyst system prompt. Granite produces a four-part read: possession & control, pressing, attacking threat, and a concrete coaching takeaway. This is genuine generative analysis — written fresh for every match, not a template.

3. **IBM Docling** — Coaches don't only have stats; they have scouting PDFs and federation reports. Docling parses those unstructured documents into clean structured text so they can enter the same analysis pipeline as the live data.

4. **Langflow** — The orchestration (document → summary → prompt → Granite) is also expressed as a Langflow flow (`/langflow`), making the pipeline visual, inspectable, and easy to extend.

A **Streamlit** dashboard ties it together with headline metrics and a pressing-intensity chart, so the numbers and the narrative sit side by side.

```
StatsBomb open data ─┐
                     ├─► structured summary ─► IBM Granite ─► tactical analysis
Scouting PDF ─Docling┘                          ▲
                                    (orchestrated in Langflow)
```

## 🏆 Why it matters in the context of the FIFA World Cup challenge

The World Cup is the moment football's tactical battles reach their largest audience — and the moment the gap between expert analysis and everyone else is most visible. TacticAI uses real World Cup match data (up to the 2022 tournament) to make that elite analysis accessible to the millions of grassroots coaches and fans worldwide who shape the game from the bottom up. It is the World Cup's tactical sophistication, handed to the people who can't normally afford it.

The pipeline is **tournament-agnostic**: the moment StatsBomb releases 2026 World Cup data to their open dataset, TacticAI supports it with zero code changes.

## 🛠️ Tech stack

| Layer            | Technology                          | Cost |
|------------------|-------------------------------------|------|
| Analysis model   | **IBM Granite** (`granite-4-h-small`) on watsonx.ai | Free tier |
| Document parsing | **IBM Docling**                     | Free |
| Orchestration    | **Langflow**                        | Free |
| Data             | StatsBomb Open Data                 | Free |
| Dashboard        | Streamlit                           | Free |
| Hosting          | Hugging Face Spaces                 | Free |

Everything in this project runs on free and open-source tooling.

## 🚀 Try it / run it

**Easiest:** just open the **[live demo](https://huggingface.co/spaces/Dakshu007/tacticai)**, pick a 2022 World Cup match, and click **Analyze match**.

**Run locally:**

```bash
git clone https://github.com/<your-username>/tacticai.git
cd tacticai
pip install -r requirements.txt

# Connect IBM Granite via watsonx.ai (free trial):
export WATSONX_API_KEY="your-key"
export WATSONX_PROJECT_ID="your-project-id"
export WATSONX_URL="https://eu-de.ml.cloud.ibm.com"   # match your region

streamlit run streamlit_app.py
```

Full setup, including the three free ways to connect Granite, is in [`docs/SETUP.md`](docs/SETUP.md). A quick credential check is available via `python test_connection.py`.

## 📂 Project structure

```
tacticai/
├── streamlit_app.py        # Streamlit dashboard (app entry point)
├── data_loader.py          # StatsBomb open-data fetch + match summarizer
├── granite_engine.py       # IBM Granite analysis (multi-backend, all free)
├── docling_parser.py       # IBM Docling scouting-document parser
├── test_connection.py      # Quick watsonx credential check
├── langflow/
│   └── tacticai_flow.json  # Importable Langflow orchestration
├── docs/
│   ├── SETUP.md            # Free setup + deployment guide
│   └── PITCH.md            # 3-minute video script
├── requirements.txt
└── README.md
```

## 🗺️ Roadmap

- Live in-match analysis via a streaming data source
- Follow-up Q&A ("how should the losing team set up next match?") through the Langflow flow
- Player-level heatmaps from event coordinates
- Automatic support for the 2026 World Cup once its data is released

## 📊 Data & license

Match data: StatsBomb Open Data, free for non-commercial use. This project is released under the [MIT License](LICENSE).

---

*Built with IBM Granite, IBM Docling, and Langflow for the IBM SkillsBuild AI Builders Challenge.*
