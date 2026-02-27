# DocQuery - Offline PDF Question Answering

A production-ready offline PDF question answering system using Ollama (llama3), sentence-transformers, and FAISS.

**Features:**
- 📄 PDF text extraction and processing
- 🧠 Local embeddings using sentence-transformers (CPU-optimized)
- 🗂️ Vector storage with FAISS for efficient retrieval
- 🤖 Local LLM via Ollama (llama3)
- 💾 Intelligent caching with SHA256 file hashing
- 📊 Comprehensive diagnostics panel
- 🔒 100% offline - no cloud APIs required
- 🪵 Full error logging and monitoring

## Project Structure

```
DocQuery/
├── app.py                 # Main Streamlit application
├── rag_pipeline.py        # RAG pipeline implementation
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── data/                  # Uploaded PDF files
├── indexes/               # FAISS vector store indexes
└── logs/                  # Error and debug logs
```

## Prerequisites

- Windows 10/11
- Python 3.11
- [Ollama](https://ollama.ai) installed locally

## Setup Instructions

### 1. Install Ollama

Download and install Ollama from: https://ollama.ai

### 2. Pull llama3 Model

Open PowerShell and run:
```powershell
ollama pull llama3
```

Start Ollama service:
```powershell
ollama serve
```

**Keep this terminal window open** - Ollama runs on `http://localhost:11434`

### 3. Create Virtual Environment

Open a new PowerShell terminal in the project directory:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. Install Dependencies

```powershell
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

This installs:
- streamlit (UI)
- pypdf (PDF extraction)
- langchain ecosystem (orchestration)
- sentence-transformers (embeddings)
- faiss-cpu (vector storage)

### 5. Run Application

```powershell
python -m streamlit run app.py
```

The application will open at `http://localhost:8501`

## Usage

1. **Upload PDF**: Click "Upload PDF" and select a PDF file
2. **Process Document**: Click "Process PDF" to:
   - Extract text from PDF
   - Split into chunks (800 chars with 120 char overlap)
   - Generate embeddings using sentence-transformers
   - Store in FAISS index
   - Cache for future use
3. **Ask Questions**: Type a question and click "Get Answer"
4. **View Diagnostics**: Click "Diagnostics" in sidebar to check system status

## Configuration

Edit `rag_pipeline.py` to modify:
- `chunk_size`: Document chunk size (default: 800)
- `chunk_overlap`: Chunk overlap (default: 120)
- `top_k`: Number of chunks to retrieve (default: 4)
- `model_name`: Embedding model (default: "all-MiniLM-L6-v2")
- `ollama_model`: LLM model name (default: "llama3")

## Architecture

### Data Flow
```
PDF Upload
    ↓
Text Extraction (pypdf)
    ↓
Text Chunking (langchain)
    ↓
Embedding Generation (sentence-transformers)
    ↓
FAISS Indexing (faiss-cpu)
    ↓
Cache with SHA256 Hash
    ↓
─────────────────────────
    ↓
User Question
    ↓
Retrieve Top-4 Chunks (FAISS)
    ↓
Create Context Window
    ↓
Send to Local Ollama (llama3)
    ↓
Generate Answer
    ↓
Display in UI
```

## Caching System

- Each PDF is identified by SHA256 file hash
- FAISS indexes stored in `indexes/index_{hash}/`
- Same PDF uploaded again → reuses cached index
- Eliminates redundant embedding generation

## Error Handling

- Graceful degradation (no stack traces in UI)
- Full error logging to `logs/error.log`
- Automatic directory creation
- Validation of dependencies in diagnostics
- Ollama connectivity checks

## Troubleshooting

### "Ollama not running"
- Start Ollama: `ollama serve`
- Check it's running at `http://localhost:11434`

### "llama3 model not found"
- Pull the model: `ollama pull llama3`

### Slow first run
- First embedding generation takes time
- Model caching occurs automatically
- Subsequent runs are faster

### Memory issues
- Using CPU-optimized models only
- FAISS indexes stored on disk
- Streams processing for large PDFs

### PDF extraction issues
- Some PDFs may have protection
- See `logs/error.log` for details

## Performance

- **Embedding Model**: all-MiniLM-L6-v2 (33M parameters)
- **LLM**: llama3 (local via Ollama)
- **Vector Store**: FAISS (CPU-based, fast retrieval)
- **Typical Q&A latency**: 5-30 seconds depending on context and model

## Logs

All errors logged to `logs/error.log`:
- PDF extraction issues
- Embedding errors
- LLM processing failures
- System diagnostics

## Requirements

See `requirements.txt` for all pinned versions compatible with Python 3.11.

Key packages:
- streamlit==1.28.1
- pypdf==3.16.0
- langchain==0.1.1
- sentence-transformers==2.2.2
- faiss-cpu==1.7.4

## System Requirements

- **OS**: Windows 10/11
- **Python**: 3.11+
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 2GB for models + PDF storage
- **CPU**: Any multi-core processor
- **GPU**: Not required (CPU-optimized)

## Environment Variables

No external API keys or authentication required. Everything runs locally.

## Development

Code follows production standards:
- Type hints throughout
- Comprehensive error handling
- Modular design
- Proper logging
- Clean code practices

## License

This project is provided as-is for local use.

## Support

For issues:
1. Check `logs/error.log`
2. Verify Ollama is running: `ollama serve`
3. Verify llama3 model: `ollama list`
4. Check diagnostics panel in Streamlit UI
5. Ensure all packages installed: `pip list`
