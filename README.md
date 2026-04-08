# NSArxivApp - ArXiv Paper Wiki

A full-stack application to discover, summarize, and explore connections between ArXiv research papers. Build a personal knowledge base of academic papers with semantic search and visual connections.

## Features

- **Search ArXiv**: Find papers by keywords and categories
- **Auto-summarization**: Extract and summarize paper content using local LLMs
- **Vector Search**: Semantic search powered by sentence transformers
- **Knowledge Graph**: Visualize connections between papers by category and authors
- **PDF Downloads**: Save papers locally for offline reading
- **Cross-platform**: Runs on Linux, macOS, and Windows via browser

## Installation

### Quick Start (macOS)

1. **Install Ollama** (downloads a local AI model runner):
```bash
# Using Homebrew (recommended):
brew install ollama

# Or download from https://ollama.com and double-click the installer
```

2. **Pull a model** (choose one):
```bash
# Qwen (good balance of quality and speed):
ollama pull qwen

# Smaller/faster option:
ollama pull llama3.2

# Larger/more capable (slower, needs more RAM):
ollama pull qwen:7b
```

3. **Clone and set up the app**:
```bash
git clone git@github.com:nikhil-sarin/NSArxivApp.git
cd NSArxivApp
pip install -r requirements.txt
```

4. **Start Ollama** (run in background):
```bash
ollama serve
```

5. **Run the app**:
```bash
streamlit run main.py
```

### Alternative: Using conda/mamba

```bash
# Create environment
conda create -n arxiv-app python=3.10
conda activate arxiv-app

# Install dependencies
pip install -r requirements.txt
```

## Usage

Once running, open your browser to `http://localhost:8501` and:

1. **Search**: Enter keywords (e.g., "transformer attention") or select categories
2. **Browse**: Papers are auto-summarized and stored for later
3. **Explore**: Use semantic search or view the knowledge graph of connections

## System Requirements

- **macOS**: 8GB+ RAM recommended (16GB for larger models)
- **Linux/Windows**: 8GB+ RAM recommended
- **Storage**: ~5GB for the model + paper PDFs
- **Python**: 3.8+

## Troubleshooting

**Ollama not starting?**
```bash
# Check if Ollama is running:
ps aux | grep ollama

# Restart:
brew services restart ollama
```

**Model too slow?**
Try a smaller model: `ollama pull qwen:0.5b` or `llama3.2`

**Out of memory?**
Close other applications or use a smaller model.

## How It Works

1. **Search**: Enter keywords or select categories to find papers
2. **Summarize**: Papers are automatically summarized using Ollama local LLM (e.g., llama3.2)
3. **Store**: Papers are stored in a vector database for semantic search
4. **Connect**: Knowledge graph tracks relationships between papers
5. **Explore**: Search semantically and visualize connections

## Architecture

```
app/
├── arxiv_client.py    # ArXiv API client
├── pdf_extractor.py   # PDF text extraction
├── summarizer.py      # LLM-based summarization
├── vector_db.py       # ChromaDB vector storage
├── knowledge_graph.py # NetworkX graph for connections
└── ui.py             # Streamlit frontend

main.py               # Entry point
requirements.txt      # Dependencies
```

## Dependencies

- `arxiv` - ArXiv API client
- `pypdf` - PDF text extraction
- `streamlit` - Web UI framework
- `chromadb` - Vector database
- `sentence-transformers` - Embedding model
- `networkx` - Knowledge graph
- `plotly` - Graph visualization

## Platform Support

✅ Linux  
✅ macOS  
✅ Windows

The app runs entirely in your browser and requires Python 3.8+.

## License

MIT
