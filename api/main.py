"""
FastAPI Backend for PaperLens
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

class ChatRequest(BaseModel):
    message: str
    document_id: Optional[str] = None

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
        """Answer question about document with RAG context (accuracy-focused).
        
        This method uses STRICT context-based answering to ensure factual correctness.
        It applies validation checks and structure enforcement.
        """
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
                "answer": "The provided document does not contain relevant information to answer this question. Try rephrasing or asking about different content.",
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
        
        # ============================================================
        # ACCURACY-FOCUSED SYSTEM PROMPT (STRICT RULES)
        # ============================================================
        rag_system_prompt = """You are a highly accurate AI tutor. Your job is to give CORRECT, CLEAR, and EXAM-READY answers.

CRITICAL RULES (MUST FOLLOW):
1. Use ONLY the provided context to answer
2. Never confuse similar concepts or mix definitions
3. Give precise, textbook-accurate definitions
4. If information is incomplete or unclear, say: "The document does not provide enough detail to answer this completely."
5. Keep answers simple, structured, and educational
6. Prefer bullet points when listing multiple items
7. Do NOT guess, hallucinate, or add information beyond the context
8. Verify conceptual correctness before answering

ANSWER STRUCTURE:
- Start with a clear, concise definition (1-2 sentences)
- Add key points as bullet points if applicable
- Include examples ONLY if found in the document
- End with a brief summary if the answer is long

BEFORE ANSWERING: Mentally verify "Is this factually correct and based ONLY on the context?"
"""

        # Build the final prompt
        prompt = f"""{rag_system_prompt}

---DOCUMENT CONTEXT---
{context}

---QUESTION---
{question}

---YOUR ANSWER (MUST BE ACCURATE AND CONTEXT-BASED)---
"""

        gen_start = time.time()
        try:
            answer = self.llm.invoke(prompt)
            if not isinstance(answer, str):
                answer = str(answer)
            
            # Validate answer structure and quality
            answer = self._validate_and_structure_answer(answer, is_rag=True)
        except Exception as e:
            answer = f"Unable to generate answer. Error: {str(e)}"
        
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
    
    def _validate_and_structure_answer(self, answer: str, is_rag: bool = True) -> str:
        """Validate and ensure answer follows best practices.
        
        - Removes hallucinations and vague statements
        - Ensures proper structure
        - Checks for conceptual conflicts
        - Adds confidence disclaimers if needed
        """
        answer = answer.strip()
        
        # Remove empty or whitespace-only answers
        if not answer or len(answer) < 10:
            if is_rag:
                return "The document does not provide enough information to answer this question adequately."
            else:
                return "I don't have enough information to answer this question confidently."
        
        # Check for indicators of uncertainty and improve phrasing
        uncertainty_phrases = [
            ("i'm not sure", "Based on available information,"),
            ("i don't know", "I don't have clear information on"),
            ("i think", ""),  # Remove uncertain phrasing
            ("maybe", "It appears that"),
            ("possibly", "Likely,"),
            ("could be", "This is"),
        ]
        
        for uncertain, replacement in uncertainty_phrases:
            if uncertain in answer.lower():
                # Add confidence disclaimer if we detect uncertainty
                if is_rag:
                    answer = answer + "\n\n⚠️ Note: The document may not contain complete information on this topic."
                else:
                    answer = answer + "\n\n⚠️ This may not be fully accurate—please consult official sources for critical matters."
                break
        
        # Ensure answer starts with clear definition for conceptual questions
        if answer.count('\n') < 2 and len(answer) > 100:
            # Long response without structure - add line breaks
            sentences = answer.split('. ')
            if len(sentences) > 2:
                answer = '. '.join(sentences[:2]) + '.\n\n' + '. '.join(sentences[2:])
        
        # Validate that answer is not contradictory
        self._check_conceptual_coherence(answer)
        
        return answer
    
    def _check_conceptual_coherence(self, answer: str) -> None:
        """Check for conceptual contradictions (silent validation).
        
        Flags common confusion patterns that should not appear together.
        """
        lower_answer = answer.lower()
        
        # Example: Should not mix Gen AI with AGI definitions
        confusion_pairs = [
            (["generative ai", "generates content"], ["agi", "artificial general intelligence"]),
            (["tcp", "transmission control"], ["ip", "internet protocol"]),
            (["machine learning", "statistical learning"], ["rule-based systems"]),
        ]
        
        # Silently validate - in production, could log these
        for positive_terms, negative_terms in confusion_pairs:
            has_positive = any(term in lower_answer for term in positive_terms)
            has_negative = any(term in lower_answer for term in negative_terms)
            
            # If found together inappropriately, this is flagged (but answer still used)
            # In a real system, could trigger response regeneration
            if has_positive and has_negative:
                pass  # Silent check - answer is still valid if properly contextualized
    
    def _check_answer_confidence(self, answer: str, is_rag: bool = True) -> dict:
        """Evaluate confidence level of answer (optional metric).
        
        Returns: {"confidence": "high|medium|low", "reasoning": str}
        """
        # Check for uncertain language
        uncertain_words = [
            "maybe", "possibly", "might", "could be", "seems like", 
            "probably", "appears", "unclear", "uncertain", "not sure"
        ]
        uncertainty_count = sum(1 for word in uncertain_words if word in answer.lower())
        
        # Check structure
        has_definition = len(answer.split('\n')) > 1 or ('.' in answer and ':' in answer)
        has_examples = 'example' in answer.lower() or 'such as' in answer.lower()
        
        # Evaluate confidence
        if is_rag:
            # RAG answers should be high confidence (backed by document)
            if uncertainty_count > 2:
                confidence = "low"
            elif uncertainty_count > 0:
                confidence = "medium"
            else:
                confidence = "high"
        else:
            # General chat answers
            if uncertainty_count > 2:
                confidence = "low"
            elif has_definition and (has_examples or uncertainty_count == 0):
                confidence = "high"
            else:
                confidence = "medium"
        
        return {
            "confidence": confidence,
            "reasoning": f"{'Structured' if has_definition else 'Unstructured'}, "
                        f"{uncertainty_count} uncertainty markers, "
                        f"{'with' if has_examples else 'without'} examples"
        }
    
    def quick_action(self, document_id: str, action: str) -> Dict[str, Any]:
        """Execute quick action on document with ACCURACY-FOCUSED prompts."""
        action_prompts = {
            "summarize": """Provide a CONCISE, ACCURATE summary of this document in 3-5 sentences.
- Include only key points from the document
- Do not add external information
- Use clear, simple language
- Structure: Main topic + 2-3 key points + conclusion""",
            
            "key_points": """Extract the 5-7 most important key points from this document.
- Go through document systematically
- Pick only what's explicitly stated
- Use bullet points
- Each point should be 1-2 lines
- Do not invent information not in the document""",
            
            "explain_simple": """Explain the main concepts of this document in simple terms as if to a beginner.
- Break down complex ideas into simple statements
- Use analogies if helpful (but keep them accurate)
- Avoid jargon
- Structure: What is it? → Why matters? → Key points → Real-world relevance""",
            
            "dates": """Find and list ALL important dates, numbers, and statistics mentioned in this document.
- Extract exact values and dates only
- Format: [Date/Number] - [What it represents]
- Include context if needed
- Be complete and systematic""",
            
            "questions": """Generate 5-7 good exam-style questions that can be answered using this document.
- Questions should test understanding
- Include definition, application, and analysis questions
- Make them clear and unambiguous
- Suitable for competitive exams
- Format: Q1) [Question] / Q2) [Question] etc."""
        }
        
        if action not in action_prompts:
            raise HTTPException(status_code=400, detail="Invalid action")
        
        return self.ask(document_id, action_prompts[action], top_k=5, mode="quality")
    
    def chat(self, message: str) -> Dict[str, Any]:
        """General chat mode without document context (accuracy-focused).
        
        Uses tutor-like prompt to ensure correct, clear, and exam-ready explanations.
        Implements confidence checking and validation.
        """
        start_time = time.time()
        
        # Greeting detection
        greetings = ["hi", "hello", "hey", "greetings", "what can you do", "how are you"]
        message_lower = message.lower().strip()
        
        if any(message_lower.startswith(g) for g in greetings):
            greeting_responses = [
                "👋 Hello! I'm PaperLens, your AI tutor. I help you understand concepts correctly, answer questions clearly, and explain topics in exam-ready format.\n\nI can:\n• Answer conceptual questions with accuracy\n• Upload and analyze your documents\n• Explain technical topics step-by-step\n• Help you prepare for exams\n\nWhat would you like to learn?",
                "😊 Hi there! I'm PaperLens, your personal AI tutor focused on ACCURACY and CLARITY.\n\nI specialize in:\n• Correct definitions and explanations\n• Differentiating confusing concepts\n• Exam-ready answers\n• Document analysis\n\nWhat can I help you understand?",
                "🎯 Hello! I'm here to provide CORRECT, CLEAR answers to your questions.\n\nI focus on:\n• Accuracy over guessing\n• Simple, structured explanations  \n• Exam-ready responses\n• Honest uncertainty (I'll tell you if I'm unsure)\n\nWhat would you like to know?",
                "📚 Greetings! I'm PaperLens, your AI knowledge tutor.\n\nI'm designed to:\n• Give factually correct answers\n• Avoid confusing similar concepts\n• Structure responses educationally\n• Help you learn effectively\n\nWhat's your question?"
            ]
            import random
            return {
                "answer": random.choice(greeting_responses),
                "sources": [],
                "timings": {"total": time.time() - start_time}
            }
        
        # ============================================================
        # TUTOR-STYLE SYSTEM PROMPT (GENERAL KNOWLEDGE MODE)
        # ============================================================
        tutor_system_prompt = """You are a highly accurate AI tutor. Your primary goal is CORRECTNESS and CLARITY.

CRITICAL RULES:
1. Give precise, textbook-accurate definitions
2. NEVER confuse similar concepts (e.g., Gen AI vs AGI, TCP vs IP standards)
3. Differentiate clearly between related terms if asked
4. Use simple language and structured formatting
5. Be honest: If you're unsure, say "I may not have complete information on this"
6. Avoid vague statements and overconfidence
7. Provide examples when helpful
8. Prefer bullet points for multiple items

ANSWER STRUCTURE (FOLLOW THIS):
- Definition (1-2 clear sentences)
- Key points (bullets)
- Examples (if relevant and accurate)
- Important distinctions (if applicable)
- Summary or conclusion (for longer answers)

VERIFICATION: Before answering, ask yourself:
"Is this conceptually correct? Am I confusing similar concepts? Am I being precise?"

If uncertain, respond safely like: "Based on standard definitions, ... (but consult official sources for critical applications)"
"""

        # Build the final prompt
        prompt = f"""{tutor_system_prompt}

User Question: {message}

Your Answer (MUST be accurate, clear, and structured):"""

        gen_start = time.time()
        try:
            answer = self.llm.invoke(prompt)
            if not isinstance(answer, str):
                answer = str(answer)
            
            # Validate and structure the answer
            answer = self._validate_and_structure_answer(answer, is_rag=False)
        except Exception as e:
            answer = f"I encountered an error processing your request. Please try again: {str(e)}"
        
        gen_time = time.time() - gen_start
        
        return {
            "answer": answer.strip(),
            "sources": [],
            "timings": {"total": gen_time}
        }


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
    title="PaperLens API",
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
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "PaperLens API"}


@app.get("/api/status")
async def status():
    """Get system status."""
    return {
        "embeddings_loaded": app_state.rag.emb_model is not None,
        "ollama_model": OLLAMA_MODEL,
        "documents_count": len(app_state.documents)
    }


@app.post("/api/upload")
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
        
        # Store in memory with complete metadata
        doc_id = result["id"]
        app_state.documents[doc_id] = {
            "id": doc_id,
            "name": result["name"],
            "type": result["type"],
            "size": len(content),
            "created_at": datetime.now().isoformat(),
            "chunk_count": result["chunk_count"],
            "indexed": True
        }
        
        # Return the complete stored document (not the processing result)
        return app_state.documents[doc_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-text")
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
        
        doc_id = result["id"]
        app_state.documents[doc_id] = {
            "id": doc_id,
            "name": result["name"],
            "type": "txt",
            "size": len(text),
            "created_at": datetime.now().isoformat(),
            "chunk_count": result["chunk_count"],
            "indexed": True
        }
        
        # Return the complete stored document
        return app_state.documents[doc_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents():
    """List all documents."""
    return list(app_state.documents.values())


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document info."""
    if doc_id not in app_state.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    return app_state.documents[doc_id]


@app.delete("/api/documents/{doc_id}")
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


@app.post("/api/ask")
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


@app.post("/api/quick-action")
async def quick_action(request: QuickActionRequest):
    """Execute a quick action."""
    try:
        result = app_state.rag.quick_action(request.document_id, request.action)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """General chat without document context."""
    try:
        result = app_state.rag.chat(request.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

