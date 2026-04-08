# NSArxivApp - ArXiv Paper Wiki

A full-stack application to discover, summarize, and explore connections between ArXiv research papers. Build a personal knowledge base of academic papers with semantic search and visual connections.

## Features

- **Search ArXiv**: Find papers by keywords and categories
- **Auto-summarization**: Extract and summarize paper content using LLMs
- **Vector Search**: Semantic search powered by sentence transformers
- **Knowledge Graph**: Visualize connections between papers by category and authors
- **PDF Downloads**: Save papers locally for offline reading
- **Cross-platform**: Runs on Linux, macOS, and Windows via browser

## Installation

1. Clone the repository:
```bash
git clone git@github.com:nikhil-sarin/NSArxivApp.git
cd NSArxivApp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running with qwen:
```bash
# Check if qwen is available:
ollama list

# If not, pull it (only if needed):
ollama pull qwen
```

4. (Optional) Configure environment variables:
```bash
cp .env.example .env
# Edit .env to customize model and host
```

## Usage

Run the Streamlit app:
```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`.

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
