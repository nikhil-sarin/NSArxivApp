# NSArxivApp - ArXiv Paper Wiki

A full-stack application to discover, summarize, and explore connections between ArXiv research papers. Build a personal knowledge base of academic papers with semantic search, visual connections, and an AI chat interface.

## Features

- **Search ArXiv**: Find papers by keywords, categories, and date range (today / last 7 days / last 30 days)
- **Research inbox**: Triage new papers and rank them against your profile, active ideas, and relevance feedback
- **Auto-summarization**: Extract and summarize paper content using a configurable LLM
- **Detailed guided reports**: Generate ArXivSelaa-style reading reports for any saved paper with your configured LLM
- **Chat with papers**: Ask questions about any paper in your library — the LLM reads the PDF and answers in context
- **Unified evidence search**: Hybrid keyword/vector retrieval across pages, sections, figures, summaries, reports, notes, projects, and meetings
- **Evidence-backed synthesis**: Literature reviews with source-linked evidence matrices and BibTeX export
- **Project intelligence**: Link papers, inspect Git/Notion-export changes, record meetings, and generate private weekly briefings
- **Research operations**: Emerging-theme detection, durable background indexing jobs, index repair, and persisted model evaluations
- **Handwritten notes**: Local vision-model transcription directly into citation-aware paper notes
- **Citation opportunities**: Manually compare a full paper with a versioned personal contribution catalogue, inspect verbatim evidence/reference checks, and explicitly propose a local email-draft action
- **Knowledge graph**: Visualize connections between papers by category and authors
- **Persistent library**: Papers, summaries, and metadata are saved locally and reload automatically on restart
- **Scheduled fetch**: Run automated daily searches or ArXivSelaa-style new-submission fetches via cron or macOS launchd
- **Multi-provider LLM**: Switch between Ollama (local), Gemini, Anthropic, and OpenAI via a single env var
- **Privacy-aware routing**: Keep notes and project context local while allowing paper-only workflows to use cloud models
- **Remote access**: Run on a workstation, access from anywhere via Tailscale

## Installation

### Quick Start

```bash
git clone git@github.com:nikhil-sarin/NSArxivApp.git
cd NSArxivApp
pip install -r requirements.txt
```

Copy and configure the environment file:
```bash
cp .env.example .env   # or edit .env directly
```

Run the app:
```bash
streamlit run main.py
```

Open `http://localhost:8501` in your browser.

### Workstation setup with persistent tmux

If you want this running continuously on a workstation:

1. Create a local `.env`:
   ```env
   SUMMARIZER_PROVIDER=ollama
   OLLAMA_MODEL=gemma4:latest
   OLLAMA_HOST=http://localhost:11434
   ```
2. Keep your existing library by preserving the `data/` directory. If you copied `data/` from another machine, the saved papers, summaries, ideas, profile, and vector database remain available.
3. Use the tmux helper:
   ```bash
   ./manage_app_tmux.sh start
   ./manage_app_tmux.sh status
   ./manage_app_tmux.sh logs
   ./manage_app_tmux.sh attach
   ./manage_app_tmux.sh restart
   ./manage_app_tmux.sh stop
   ```

The helper starts Streamlit on `127.0.0.1:8501` inside a tmux session named `nsarxiv-app`.

### Using conda/mamba

```bash
conda create -n arxiv-app python=3.10
conda activate arxiv-app
pip install -r requirements.txt
```

---

## LLM Configuration

The app supports multiple LLM providers. Set `SUMMARIZER_PROVIDER` in `.env` — no code changes needed. The provider is read live, so you can switch without restarting.

### Data-routing policy

Choose what may be sent to cloud providers:

```env
# local_only: everything stays local
# paper_cloud: paper content may use cloud; notes/profile/projects stay local (default)
# allow_cloud: all workflows may use the selected cloud provider
DATA_ROUTING_POLICY=paper_cloud
PRIVATE_LLM_PROVIDER=ollama
```

The sidebar shows the effective route for paper-only and private-context workflows.
The same policy can be changed in **System health** unless `DATA_ROUTING_POLICY` is explicitly set in the environment; an environment value remains authoritative for deployment.

Handwritten-note OCR always uses the local Ollama endpoint:

```env
OCR_MODEL=gemma3:latest
OLLAMA_HOST=http://localhost:11434
```

### Option 1: Ollama (local, default)

Install Ollama and pull a model:
```bash
brew install ollama          # macOS
ollama pull gemma4:latest    # recommended
ollama serve                 # start the server
```

`.env`:
```
SUMMARIZER_PROVIDER=ollama
OLLAMA_MODEL=gemma4:latest
OLLAMA_HOST=http://localhost:11434
```

**Recommended models:**
| Model | RAM needed | Speed | Quality |
|---|---|---|---|
| `llama3.1:latest` | ~8 GB | medium | good |
| `gemma4:12b` | ~12 GB | medium | very good |
| `gemma4:31b` | ~32 GB | slow | excellent |

### Option 2: Google Gemini (recommended for cloud)

Get an API key at [aistudio.google.com](https://aistudio.google.com).

`.env`:
```
SUMMARIZER_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
LLM_MODEL=gemini-2.0-flash          # optional, this is the default
```

Gemini 2.0 Flash has a 1M token context window — entire papers fit without truncation.

### Option 3: Anthropic Claude

`.env`:
```
SUMMARIZER_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key-here
LLM_MODEL=claude-3-5-haiku-20241022  # optional
```

### Option 4: OpenAI

`.env`:
```
SUMMARIZER_PROVIDER=openai
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini                # optional
```

`OPENAI_BASE_URL` may point to an OpenAI-compatible service; the app appends
`/chat/completions`. Set `PUBLIC_LLM_PROVIDER=openai` to keep public-paper chat
on this route even if Gemini credentials are also present.

### Remote Ollama (workstation offload)

Run Ollama on a more powerful machine and point the app at it:
```
OLLAMA_HOST=http://<workstation-ip>:11434
```
Works over a local network or Tailscale VPN.

---

## Remote Access via Tailscale

Run the app on your workstation and access it from anywhere (laptop, tablet, etc.) without port forwarding.

1. Install Tailscale on both machines: [tailscale.com/download](https://tailscale.com/download) or `brew install tailscale`
2. Sign in on both with the same account: `sudo tailscale up`
3. On your workstation, start the app:
   ```bash
   ./manage_app_tmux.sh start
   ```
4. On any other device, browse to:
    ```
    http://<workstation-tailscale-ip>:8501
    ```
   Find your Tailscale IP in the Tailscale menu bar app or with `tailscale ip`.

To also offload LLM inference to the workstation, set in `.env`:
```
OLLAMA_HOST=http://<workstation-tailscale-ip>:11434
```

If you prefer HTTPS over Tailscale Serve, expose Streamlit on a separate Tailscale HTTPS port:

```bash
tailscale serve --bg --https=8443 http://127.0.0.1:8501
```

Then browse to:

```text
https://<workstation-name>.<tailnet>.ts.net:8443
```

---

## Scheduled Daily Fetch

Automatically fetch and summarize new papers on a schedule, even when the UI is closed.

### Manual CLI

```bash
python -m app.fetch_job \
  --mode new-submissions \
  --categories astro-ph.HE gr-qc \
  --max-results 20 \
  --days-back 1
```

In `new-submissions` mode, the job starts from the requested UTC announcement day and automatically backs up to the latest non-empty announcement date, so weekend cron runs still pick up the newest ArXivSelaa-style batch.

Keyword-search mode remains available:

```bash
python -m app.fetch_job \
  --mode query-search \
  --query "neutron star kilonova" \
  --categories astro-ph.HE gr-qc \
  --max-results 20 \
  --days-back 1
```

### cron

Add to your crontab (`crontab -e`):
```
0 7 * * * cd /path/to/NSArxivApp && /path/to/python -m app.fetch_job --mode new-submissions --categories astro-ph.HE --max-results 20 --days-back 1 >> /path/to/NSArxivApp/data/fetch.log 2>&1
```

The **Schedule** tab can also install a managed Linux cron entry for you directly.

### macOS launchd

Use the **Schedule** tab in the app UI to generate and install a launchd plist automatically. It runs either fetch mode daily at a time you choose and logs to `data/fetch.log`.

---

## Usage

1. **Search**: Enter keywords and/or select categories in the sidebar, optionally filter by date, then click **Search**
2. **Library**: All saved papers appear in the Library tab — filter by category, regenerate summaries, generate detailed reports, or chat with individual papers
3. **Chat with a paper**: Click **Chat with paper** inside any library entry to ask questions — the LLM reads the PDF and answers in context
4. **Unified search**: Search page-level paper text, reports, figures, notes, projects, and meeting records from one evidence view
5. **Detailed report**: Click **Generate detailed report** on any saved paper to create a cached HTML guided-reading report
6. **Knowledge graph**: Visualize category and author connections in the Knowledge Graph tab
7. **Schedule**: Set up automated daily fetching in the Schedule tab
8. **Projects**: Attach repositories, a Notion Markdown export, meetings, and paper evidence; imported TheLocalWhisperer actions remain proposals until explicitly approved
9. **Library Health**: Inspect search indexing and model checks, and review supported trends with representative papers

---

## Data Storage

All data is stored locally under `data/` in the app directory:

```
data/
├── research.db      # papers, ideas, profile, projects, triage, actions, evaluations, jobs, and FTS index
├── papers.json      # legacy source retained unchanged after first-run migration
├── ideas.json       # legacy source retained unchanged after first-run migration
├── profile.json     # legacy source retained unchanged after first-run migration
├── papers/          # downloaded PDFs
├── reports/         # cached detailed HTML reports
├── sources/         # cached ArXiv source downloads for reports
├── vector_db/       # ChromaDB embeddings
└── fetch.log        # scheduled job logs
```

SQLite runs in WAL mode and migrations are non-destructive: legacy JSON files are read once when the corresponding tables are empty and are never rewritten. Back up `data/research.db`, `data/vector_db/`, and the legacy JSON files before moving an installation.

### TheLocalWhisperer handoff contract

Upload a JSON list in a project's **Actions** view. Only these action types are accepted: `task.create`, `calendar.create`, `notion.update`, and `agent.invoke`.

```json
[
  {
    "action_type": "task.create",
    "title": "Read the new opacity paper",
    "payload": {"paper_id": "2609.12345"}
  }
]
```

Imports always enter `proposed` state. The app does not execute an action, and an action cannot be marked `executed` until it has first been explicitly approved.

### Citation-opportunity workflow

`config/contributions.example.json` contains the validated catalogue schema and verified Redback citation metadata. Copy it to the ignored `config/contributions.json` to customize private scope notes, or point `NSARXIV_CONTRIBUTIONS_PATH` at another file. Disabled entries are never analysed.

Newly ingested papers are checked automatically using deterministic signal and reference filters before any model call. Only filtered candidates receive a strict evidence-grounded model judgement. The **Citation Opportunities** view is the manual review and export queue; its advanced controls can rerun a specific paper when the contribution catalogue changes.

Backfill a bounded recent slice after expanding the catalogue:

```bash
python -m app.citation_backfill --days 90 --max-papers 100
```

Only `strong_citation_opportunity` and `potentially_useful` findings can be confirmed. Confirmation posts a stable, bounded ImportBundle to `LOCAL_ORCHESTRATOR_URL` using `LOCAL_ORCHESTRATOR_API_TOKEN`; it includes quotes and catalogue metadata, never full paper text or secrets. LocalOrchestrator then requires separate candidate and exact-plan approval before a local model can generate editable subject/body text. Neither app sends email or creates a Gmail draft.

---

## Architecture

```
app/
├── arxiv_announcements.py # RSS/catchup announcement-day fetch
├── arxiv_client.py    # ArXiv API client
├── pdf_extractor.py   # PDF text extraction
├── report_generator.py # Detailed report generation and HTML rendering
├── source_extractor.py # ArXiv source download and figure extraction
├── summarizer.py      # Multi-provider LLM summarization and chat
├── vector_db.py       # ChromaDB vector storage
├── knowledge_graph.py # NetworkX graph for connections
├── paper_store.py     # transactional paper/triage persistence
├── research_db.py     # SQLite schema, migrations, and full-text index
├── fetch_job.py       # CLI script for scheduled fetching
└── ui.py              # Streamlit frontend

main.py                # Entry point
.env                   # LLM provider config (not committed)
.streamlit/config.toml # Streamlit settings
```

## System Requirements

- **Python**: 3.10+
- **RAM**: 8 GB minimum; 16 GB+ for large local models
- **Storage**: ~5 GB for a local model + paper PDFs
- **Ollama** (optional): only needed for local inference

## Tests

Run the standard-library test suite with the Python environment used by the app:

```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

**Ollama not connecting?**
```bash
ollama serve          # start the server
ollama list           # check available models
```

**Wrong model name?** The model must be listed in `ollama list`. Update `OLLAMA_MODEL` in `.env` to match exactly.

**Summaries empty after fetch?** Use the **Fill missing** button in the Library tab to generate summaries for any papers that were stored without one.

**App slow to start?** The sentence-transformer embedding model (~90 MB) loads on first run and is cached — subsequent reruns are fast.

**Module warnings on startup?** These are suppressed by `.streamlit/config.toml` (`fileWatcherType = poll`). If you see them, make sure the config file exists.

## License

MIT
