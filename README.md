# NSArxivApp - ArXiv Paper Wiki

A full-stack application to discover, summarize, and explore connections between ArXiv research papers. Build a personal knowledge base of academic papers with semantic search, visual connections, and an AI chat interface.

## Features

- **Search ArXiv**: Find papers by keywords, categories, and date range (today / last 7 days / last 30 days)
- **Auto-summarization**: Extract and summarize paper content using a configurable LLM
- **Detailed guided reports**: Generate ArXivSelaa-style reading reports for any saved paper with your configured LLM
- **Chat with papers**: Ask questions about any paper in your library — the LLM reads the PDF and answers in context
- **Vector search**: Semantic search powered by sentence transformers
- **Knowledge graph**: Visualize connections between papers by category and authors
- **Persistent library**: Papers, summaries, and metadata are saved locally and reload automatically on restart
- **Scheduled fetch**: Run automated daily searches or ArXivSelaa-style new-submission fetches via cron or macOS launchd
- **Multi-provider LLM**: Switch between Ollama (local), Gemini, Anthropic, and OpenAI via a single env var
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
LLM_MODEL=gpt-4o-mini                # optional
```

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
4. **Semantic search**: Describe what you're looking for in plain language in the Semantic Search tab
5. **Detailed report**: Click **Generate detailed report** on any saved paper to create a cached HTML guided-reading report
6. **Knowledge graph**: Visualize category and author connections in the Knowledge Graph tab
7. **Schedule**: Set up automated daily fetching in the Schedule tab

---

## Data Storage

All data is stored locally under `data/` in the app directory:

```
data/
├── papers.json      # paper metadata and summaries (persistent across restarts)
├── papers/          # downloaded PDFs
├── reports/         # cached detailed HTML reports
├── sources/         # cached ArXiv source downloads for reports
├── vector_db/       # ChromaDB embeddings
└── fetch.log        # scheduled job logs
```

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
├── paper_store.py     # JSON persistence layer
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
