# 📄 PaperLens

**Premium Local AI Document Assistant**

A production-ready RAG (Retrieval-Augmented Generation) application that lets you upload documents and chat with an AI that understands them. 100% local processing, complete privacy, instant answers.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![React](https://img.shields.io/badge/react-18.2.0-61dafb)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Key Features

- **💻 100% Local Processing** - Your documents never leave your computer
- **🤖 AI-Powered Analysis** - Powered by Ollama + Llama3 LLM
- **📄 Multi-Format Support** - PDF, DOCX, TXT, Markdown, pasted text
- **⚡ Lightning Fast** - FAISS vector search, sentence-transformers embeddings
- **🔒 Complete Privacy** - No internet required, no data sent to cloud
- **🎯 Smart Quick Actions** - Summarize, key points, explain simply, find dates, quiz mode
- **⚙️ Performance Modes** - Fast (⚡) / Balanced (⚖️) / Quality (🎯)
- **💬 Multi-turn Chat** - Context-aware conversations about documents
- **📊 Source Tracking** - See exactly which document sections were used
- **🔐 Login System** - Frontend authentication demo

---

## 🏗️ Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Browser   │◄───────►│  React + UI  │         │   FastAPI   │
│   (Login)   │         │    (Vite)    │◄───────►│   Backend   │
└─────────────┘         └──────────────┘         └─────────────┘
                            Port 5173                Port 8000
                                                          │
                        ┌─────────────────────────────────┼─────────────────┐
                        │                                 │                 │
                   ┌────▼─────┐                   ┌──────▼──────┐   ┌──────▼──────┐
                   │  Ollama   │                   │  FAISS      │   │ Sentence    │
                   │  (Llama3) │                   │  Indexes    │   │ Transformers│
                   └───────────┘                   └─────────────┘   └─────────────┘
                   Port 11434                    Vector Search      Embeddings
```

**Components:**
- **Frontend:** React 18 + Vite + Tailwind CSS + Framer Motion
- **Backend:** FastAPI + Pydantic
- **RAG:** FAISS + SentenceTransformers + Ollama

---

## 📋 Prerequisites

- **Python 3.9+** with pip
- **Node.js 18+** with npm
- **Ollama** (running with llama3 model pulled)
- **Git** (for cloning)

**System Requirements:**
- Minimum 8GB RAM (16GB recommended)
- 20GB+ disk space (for models and documents)
- CPU with decent performance or GPU acceleration

---

## 🚀 Quick Start (5 minutes)

### 1️⃣ Prerequisites: Start Ollama

**macOS:**
```bash
brew install ollama
# Terminal 1: Start server
ollama serve
# Terminal 2: Pull model
ollama pull llama3
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
# Terminal 1: Start server
ollama serve
# Terminal 2: Pull model
ollama pull llama3
```

**Windows:**
- Download from [ollama.ai](https://ollama.ai)
- Install and run
- In PowerShell: `ollama pull llama3`

✅ **Ollama should be running at `http://localhost:11434` before starting the backend.**

---

### 2️⃣ Clone Repository
```bash
git clone <your-repo-url>
cd PaperLens
```

---

### 3️⃣ Backend Setup

```bash
cd api
pip install -r requirements.txt
python main.py
```

✅ Backend running at `http://localhost:8000`  
📖 API docs at `http://localhost:8000/docs`

---

### 4️⃣ Frontend Setup (New Terminal)

```bash
cd ui
npm install
npm run dev
```

✅ Frontend running at `http://localhost:5173`

---

### 5️⃣ Open in Browser
```
http://localhost:5173
```

**Demo Login:**
- Email: `demo@paperlens.ai` (any email works)
- Password: `demo` (any password works)
- Or click **"Try Demo Mode"**

---

## 🎯 Usage Guide

### 📁 Uploading Documents
1. Click **"Upload File"** in sidebar
2. Select PDF, DOCX, TXT, MD, or paste text
3. Document is instantly indexed
4. Start asking questions!

### ❓ Asking Questions
1. Select a document from the sidebar
2. Type your question in the chat box
3. Choose a **Performance Mode** (affects quality vs. speed)
4. Press Enter or click Send

### ⚡ Quick Actions
Available after selecting a document:
- **Summarize** - Concise overview in 3-5 sentences
- **Key Points** - Top 5-7 important points
- **Explain Simply** - Beginner-friendly explanation
- **Find Dates/Numbers** - All dates, stats, and numbers
- **Quiz Me** - Generate 5 interview-style questions

### ⚙️ Performance Modes
| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| ⚡ Fast | 2-3s | Basic | Quick answers, many questions |
| ⚖️ Balanced | 4-6s | Good | Default, best overall |
| 🎯 Quality | 8-15s | Excellent | Deep analysis, important answers |

### 📊 Viewing Sources
Toggle **"Show Sources"** to see which document chunks the AI used. Great for verifying facts and tracing answers back to source material.

---

## 🔌 API Endpoints

All endpoints are prefixed with `/api`

### Status
```
GET /api/health          → {"status": "ok", "service": "PaperLens API"}
GET /api/status          → {"embeddings_loaded": true, "ollama_model": "llama3", ...}
```

### Documents
```
GET /api/documents       → List all documents
GET /api/documents/{id}  → Get document details
POST /api/upload         → Upload document file (multipart)
POST /api/upload-text    → Upload plain text (form-data)
DELETE /api/documents/{id} → Delete document
```

### RAG & Chat
```
POST /api/ask
{
  "document_id": "abc123",
  "question": "What is the main topic?",
  "top_k": 3,
  "mode": "balanced"
}

POST /api/quick-action
{
  "document_id": "abc123",
  "action": "summarize"
}

POST /api/chat
{
  "message": "Hello, what can you do?"
}
```

**Response:**
```json
{
  "answer": "...",
  "sources": [
    {
      "content": "...",
      "score": 0.85,
      "index": 5
    }
  ],
  "timings": {
    "search": 0.23,
    "generation": 2.45,
    "total": 2.68
  }
}
```

---

## ⚙️ Configuration

### Backend Configuration
Edit `api/main.py` to customize:

```python
# Line 36-41
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast (22MB, default)
OLLAMA_MODEL = "llama3"                # LLM model
DEFAULT_CHUNK_SIZE = 800               # Text chunk size
DEFAULT_CHUNK_OVERLAP = 120            # Overlap between chunks
```

### Available Ollama Models
```bash
ollama pull mistral          # Faster, decent quality
ollama pull llama2           # Older, smaller
ollama pull neural-chat      # Optimized for chat
ollama pull dolphin-mixtral  # Very smart, heavier
```

### Environment Variables
```bash
# Optional (defaults shown)
export OLLAMA_BASE_URL="http://localhost:11434"
export VITE_API_URL="http://127.0.0.1:8000"  # Frontend API endpoint
```

### Alternative Embedding Models
Update `api/main.py` line 35:
```python
# Faster, lower quality
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Better quality, slower
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
```

---

## 🧠 How It Works

### Upload & Indexing
1. Document uploaded → saved to `temp_uploads/`
2. Extract text (PDF, DOCX, TXT, MD)
3. Split into overlapping chunks (800 tokens default)
4. Generate embeddings (SentenceTransformers)
5. Build FAISS index for fast search
6. Cache index to disk

### Question Answering
1. Embed user question
2. Search FAISS index → get top-k similar chunks
3. Build context from chunks
4. Send to Ollama LLM with system prompt
5. Return answer + sources + timings

### Performance
- **Embedding:** ~50-100ms (cached)
- **Search:** ~20-50ms (FAISS is very fast)
- **Generation:** ~1-10s (depends on Ollama)
- **Total:** ~2-15s per query

---

## 📁 Project Structure

```
PaperLens/
├── api/                          # FastAPI Backend
│   ├── main.py                  # Core API & RAG pipeline
│   ├── requirements.txt         # Python dependencies
│   └── __pycache__/
├── ui/                          # React Frontend
│   ├── src/
│   │   ├── api.js              # API client
│   │   ├── App.jsx             # Root component + routing
│   │   ├── main.jsx            # Entry point
│   │   ├── index.css           # Global styles
│   │   └── pages/
│   │       ├── LoginPage.jsx   # Login/auth page
│   │       ├── LandingPage.jsx # Marketing homepage
│   │       └── ChatPage.jsx    # Main chat interface
│   ├── index.html              # HTML template
│   ├── package.json            # Node dependencies
│   ├── vite.config.js         # Vite config
│   ├── tailwind.config.js     # Tailwind config
│   └── postcss.config.js      # PostCSS config
├── indexes/                     # FAISS vector stores (auto-created)
├── temp_uploads/              # Uploaded files (auto-created)
├── logs/                       # Application logs (auto-created)
├── data/                       # Misc data (auto-created)
└── README.md                   # This file
```

---

## 🐛 Troubleshooting

### Backend Error: "Connection refused"
```bash
# Check if port 8000 is in use
lsof -i :8000              # macOS/Linux
netstat -ano | findstr 8000  # Windows

# Kill the process or use different port
python main.py --port 8001
```

### "Cannot connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve

# Verify it's accessible
curl http://localhost:11434/api/tags

# If using different port, update api/main.py:
# base_url="http://localhost:11434"
```

### Frontend shows "Failed to load documents"
1. Check backend is running: `http://localhost:8000/api/health`
2. Open DevTools (F12) → Network tab → check response
3. Check browser console for CORS errors
4. Verify ports in `ui/src/api.js`

### Embedding model downloads slowly
- First run downloads ~100MB model (one-time)
- Models cached in `~/.cache/huggingface/`
- Subsequent runs use cached version

### Password/Login not working
- Demo mode accepts any email/password
- Frontend auth is demo-only (no backend validation)
- To add real auth, implement JWT in backend

### PDF extraction issues
- Works best with text-based PDFs
- Scanned PDFs (images) won't extract text
- Try OCR tools first (tesseract) or convert to images

### Out of Memory
1. Reduce `DEFAULT_CHUNK_SIZE` in `api/main.py`
2. Use smaller embedding model: `all-MiniLM-L6-v2`
3. Process on a machine with more RAM
4. Close other applications

---

## 🚀 Production Deployment

### Using Gunicorn (Linux/macOS)
```bash
cd api
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### Docker Deployment
**Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY api/requirements.txt .
RUN pip install -r requirements.txt
COPY api/ .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

**Build & Run:**
```bash
docker build -t paperlens-api .
docker run -p 8000:8000 \
  -e OLLAMA_BASE_URL="http://ollama:11434" \
  -v paperlens-indexes:/app/indexes \
  paperlens-api
```

### Environment Checklist
- [ ] Ollama running and accessible
- [ ] All Python dependencies installed
- [ ] All Node dependencies installed
- [ ] Backend starts without errors
- [ ] Frontend loads without CORS errors
- [ ] Can upload a test document
- [ ] Can ask a question and get an answer

---

## 🔐 Security Notes

**Current Design (Demo Mode):**
- No external API calls - 100% local
- Frontend-only authentication (localStorage)
- Files stored unencrypted in `temp_uploads/`

**For Production, Add:**
```python
# 1. JWT Authentication
from fastapi.security import HTTPBearer, HTTPAuthCredential

# 2. Request signing
from cryptography.hazmat.primitives import hashes

# 3. Document encryption
from cryptography.fernet import Fernet

# 4. Environment variables
import os
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

# 5. Rate limiting
from slowapi import Limiter
```

---

## 📊 Performance Optimization

1. **Use appropriate mode**
   - Fast mode for UX testing
   - Balanced for most use cases
   - Quality for research/analysis

2. **Tune chunk size**
   - Smaller = more precise, longer generation
   - Default 800 is good balance

3. **Optimize embedding model**
   - `all-MiniLM-L6-v2` - Default, fastest
   - `all-mpnet-base-v2` - Better quality
   - Model choice impacts accuracy

4. **Enable GPU support** (5-10x speedup)
   ```bash
   pip install torch torchvision torchaudio \
     --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Use document caching**
   - Indexes cached automatically
   - Re-uploading same doc is instant

---

## 📚 Resources

- **Ollama**: https://ollama.ai/library
- **SentenceTransformers**: https://www.sbert.net/
- **FAISS**: https://github.com/facebookresearch/faiss
- **FastAPI**: https://fastapi.tiangolo.com/
- **React**: https://react.dev
- **Tailwind CSS**: https://tailwindcss.com/

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🎯 Roadmap

- [ ] Real user authentication (JWT)
- [ ] Persistent database (PostgreSQL)
- [ ] Multi-user support
- [ ] Advanced RAG (reranking, hyde, expansion)
- [ ] More formats (images, web content)
- [ ] Real-time streaming responses
- [ ] Custom system prompts
- [ ] Document sharing
- [ ] Webhook integrations
- [ ] Docker Compose setup

---

**Made with ❤️ for document lovers everywhere**

PaperLens v2.0 © 2024 | [MIT License](LICENSE)
