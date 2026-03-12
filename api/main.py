"""
FastAPI Backend for DocQuery
Premium local AI document assistant API
"""
import os
import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import faiss
import pickle

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INDEXES_DIR = BASE_DIR / "indexes"
UPLOADS_DIR = BASE_DIR / "temp_uploads"
LOGS_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, INDEXES_DIR, UPLOADS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


# ============================================================
# DATA MODELS
# ============================================================
class ChatMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    top_k: int = 3
    mode: str = "balanced"

class QuickActionRequest(BaseModel):
    action: str
    document_id: str

class DocumentMetadata(BaseModel):
    id: str
    name: str
    type: str
    size: int
    created_at: str
    chunk_count: int = 0
    indexed: bool = False


# ============================================================
# VECTOR STORE
# ============================================================
class VectorStore:
    def __init__(self):
        self.index = None
        self.texts = []
        self.vectors = None
        self.emb_model = None
        self.metadata = {}
        self._query_cache = {}
    
    def initialize(self, emb_model: SentenceTransformer):
        self.emb_model = emb_model
    
    def build(self, texts: List[str], metadata: Dict = None):
        if not self.emb_model:
            raise ValueError("Embedding model not initialized")
        
        self.texts = texts
        self.metadata = metadata or {}
        
        # Encode in batches
        vectors = self.emb_model.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=32)
        
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        
        self.vectors = vectors
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        if not self.emb_model or not self.index:
            return []
        
        # Check cache
        if query in self._query_cache:
            qvec = self._query_cache[query]
        else:
            qvec = self.emb_model.encode([query], show_progress_bar=False, convert_to_numpy=True)
            qvec = qvec / (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12)
            if len(self._query_cache) < 100:
                self._query_cache[query] = qvec
        
        D, I = self.index.search(qvec, k)
        
        results = []
        for pos, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append({
                "content": self.texts[idx],
                "score": float(D[0][pos]),
                "index": idx
            })
        return results
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/meta.pkl", "wb") as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)
    
    def load(self, path: str, emb_model: SentenceTransformer):
        self.emb_model = emb_model
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/meta.pkl", "rb") as f:
            data = pickle.load(f)
            self.texts = data.get("texts", [])
            self.metadata = data.get("metadata", {})


# ============================================================
# DOCUMENT PROCESSORS
# ============================================================
class DocumentProcessor:
    """Process multiple document formats."""
    
    @staticmethod
    def process_pdf(file_path: str) -> str:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    
    @staticmethod
    def process_txt(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    @staticmethod
    def process_docx(file_path: str) -> str:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    
    @staticmethod
    def process_md(file_path: str) -> str:
        return DocumentProcessor.process_txt(file_path)
    
    @staticmethod
    def process_markdown(file_path: str) -> str:
        return DocumentProcessor.process_txt(file_path)
    
    @classmethod
    def process(cls, file_path: str, file_type: str) -> str:
        """Process document based on type."""
        processors = {
            "pdf": cls.process_pdf,
            "txt": cls.process_txt,
            "docx": cls.process_docx,
            "md": cls.process_md,
            "markdown": cls.process_markdown,
        }
        
        processor = processors.get(file_type.lower())
        if not processor:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return processor(file_path)


# ============================================================
# RAG PIPELINE
# ============================================================
class RAGPipeline:
    def __init__(self):
        self.emb_model = None
        self.llm = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        self.mode_settings = {
            "fast": {"chunk_size": 500, "top_k": 2, "max_tokens": 128},
            "balanced": {"chunk_size": 800, "top_k": 3, "max_tokens": 256},
            "quality": {"chunk_size": 1200, "top_k": 5, "max_tokens": 512}
        }
    
    def initialize(self):
        """Initialize models."""
        if self.emb_model is None:
            self.emb_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        
        if self.llm is None:
            self.llm = OllamaLLM(
                model=OLLAMA_MODEL,
                base_url="http://localhost:11434",
                temperature=0,
                num_predict=256
            )
    
    def get_document_hash(self, file_path: str) -> str:
        """Get hash for document."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()[:16]
    
    def process_document(self, file_path: str, file_name: str, file_type: str) -> Dict[str, Any]:
        """Process document and create vector store."""
        start_time = time.time()
        
        # Get document hash
        doc_hash = self.get_document_hash(file_path)
        index_path = INDEXES_DIR / doc_hash
        
        # Check cache
        if index_path.exists():
            vectorstore = VectorStore()
            vectorstore.load(str(index_path), self.emb_model)
            return {
                "id": doc_hash,
                "name": file_name,
                "type": file_type,
                "cached": True,
                "chunk_count": len(vectorstore.texts),
                "time": time.time() - start_time
            }
        
        # Process document
        text = DocumentProcessor.process(file_path, file_type)
        
        # Chunk text
        chunks = self.text_splitter.split_text(text)
        
        # Build vector store
        vectorstore = VectorStore()
        vectorstore.initialize(self.emb_model)
        vectorstore.build(chunks, {"name": file_name, "type": file_type})
        vectorstore.save(str(index_path))
        
        return {
            "id": doc_hash,
            "name": file_name,
            "type": file_type,
            "cached": False,
            "chunk_count": len(chunks),
            "time": time.time() - start_time
        }
    
    def ask(self, document_id: str, question: str, top_k: int = 3, mode: str = "balanced") -> Dict[str, Any]:
        """Answer question about document."""
        start_time = time.time()
        
        # Load vector store
        index_path = INDEXES_DIR / document_id
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")
        
        vectorstore = VectorStore()
        vectorstore.load(str(index_path), self.emb_model)
        
        # Search
        search_start = time.time()
        sources = vectorstore.search(question, k=top_k)
        search_time = time.time() - search_start
        
        if not sources:
            return {
                "answer": "No relevant information found in the document.",
                "sources": [],
                "timings": {"search": search_time, "total": time.time() - start_time}
            }
        
        # Build context
        max_chars = self.mode_settings.get(mode, {}).get("max_tokens", 256) * 4
        context = ""
        for s in sources:
            if len(context) + len(s["content"]) > max_chars:
                break
            context += s["content"] + "\n\n"
        
        # Generate answer
        prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the question.

Context:
{context}

Question: {question}

Answer:"""

        gen_start = time.time()
        try:
            answer = self.llm.invoke(prompt)
            if not isinstance(answer, str):
                answer = str(answer)
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        gen_time = time.time() - gen_start
        
        return {
            "answer": answer.strip(),
            "sources": sources,
            "timings": {
                "search": search_time,
                "generation": gen_time,
                "total": time.time() - start_time
            }
        }
    
    def quick_action(self, document_id: str, action: str) -> Dict[str, Any]:
        """Execute quick action on document."""
        action_prompts = {
            "summarize": "Provide a concise summary of this document in 3-5 sentences.",
            "key_points": "Extract the 5-7 most important key points from this document.",
            "explain_simple": "Explain this document in simple terms as if to a beginner.",
            "dates": "Find and list all important dates, numbers, and statistics mentioned in this document.",
            "questions": "Generate 5 good interview questions that can be answered using this document."
        }
        
        if action not in action_prompts:
            raise HTTPException(status_code=400, detail="Invalid action")
        
        return self.ask(document_id, action_prompts[action], top_k=5, mode="quality")


# ============================================================
# APP STATE
# ============================================================
class AppState:
    def __init__(self):
        self.rag = RAGPipeline()
        self.documents: Dict[str, Dict] = {}
    
    def ensure_initialized(self):
        if not self.rag.emb_model:
            self.rag.initialize()


app_state = AppState()


# ============================================================
# LIFESPAN
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state.ensure_initialized()
    yield


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="DocQuery API",
    description="Premium local AI document assistant",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/health")
async def health():
    return {"status": "ok", "service": "DocQuery API"}


@app.get("/status")
async def status():
    """Get system status."""
    return {
        "embeddings_loaded": app_state.rag.emb_model is not None,
        "ollama_model": OLLAMA_MODEL,
        "documents_count": len(app_state.documents)
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """Upload and process a document."""
    # Save file
    file_path = UPLOADS_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)
    
    # Get file type
    file_ext = file.filename.split(".")[-1].lower()
    
    # Process
    try:
        result = app_state.rag.process_document(str(file_path), file.filename, file_ext)
        
        # Store in memory
        app_state.documents[result["id"]] = {
            "id": result["id"],
            "name": result["name"],
            "type": result["type"],
            "size": len(content),
            "created_at": datetime.now().isoformat(),
            "chunk_count": result["chunk_count"],
            "indexed": True
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-text")
async def upload_text(
    text: str = Form(...),
    name: str = Form("Pasted Text")
):
    """Upload plain text."""
    # Save as txt file
    file_name = f"{name[:50]}_{int(time.time())}.txt"
    file_path = UPLOADS_DIR / file_name
    file_path.write_text(text)
    
    # Process
    try:
        result = app_state.rag.process_document(str(file_path), file_name, "txt")
        
        app_state.documents[result["id"]] = {
            "id": result["id"],
            "name": result["name"],
            "type": "txt",
            "size": len(text),
            "created_at": datetime.now().isoformat(),
            "chunk_count": result["chunk_count"],
            "indexed": True
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all documents."""
    return list(app_state.documents.values())


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document info."""
    if doc_id not in app_state.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    return app_state.documents[doc_id]


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document."""
    if doc_id not in app_state.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from memory
    del app_state.documents[doc_id]
    
    # Remove index
    index_path = INDEXES_DIR / doc_id
    if index_path.exists():
        import shutil
        shutil.rmtree(index_path)
    
    return {"status": "deleted", "id": doc_id}


@app.post("/ask")
async def ask_question(request: AskRequest):
    """Ask a question about a document."""
    if not request.document_id:
        raise HTTPException(status_code=400, detail="document_id required")
    
    try:
        result = app_state.rag.ask(
            request.document_id,
            request.question,
            request.top_k,
            request.mode
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quick-action")
async def quick_action(request: QuickActionRequest):
    """Execute a quick action."""
    try:
        result = app_state.rag.quick_action(request.document_id, request.action)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

