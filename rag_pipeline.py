import os
import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import pypdf
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM


class SimpleVectorStore:
    """Minimal FAISS-backed vector store for local use.

    Stores normalized vectors and texts, supports similarity_search(question, k).
    """
    def __init__(self, index: faiss.IndexFlatIP, texts: List[str], vectors: np.ndarray, emb_model: SentenceTransformer):
        self.index = index
        self.texts = texts
        self.vectors = vectors
        self.emb_model = emb_model

    def similarity_search(self, query: str, k: int = 4) -> List[dict]:
        qvec = self.emb_model.encode([query], convert_to_numpy=True)
        qvec = qvec / (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12)
        D, I = self.index.search(qvec, k)
        results = []
        for pos, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            score = float(D[0][pos])
            results.append({"page_content": self.texts[idx], "score": score})
        return results

    def save_local(self, folder: str) -> None:
        Path(folder).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "texts.pkl"), "wb") as f:
            pickle.dump({"texts": self.texts, "vectors": self.vectors}, f)

    @classmethod
    def load_local(cls, folder: str, emb_model: SentenceTransformer):
        idx_file = os.path.join(folder, "index.faiss")
        txt_file = os.path.join(folder, "texts.pkl")
        if not os.path.exists(idx_file) or not os.path.exists(txt_file):
            return None
        index = faiss.read_index(idx_file)
        with open(txt_file, "rb") as f:
            data = pickle.load(f)
        return cls(index=index, texts=data.get("texts", []), vectors=data.get("vectors", None), emb_model=emb_model)


class DocQueryPipeline:
    def __init__(self, logs_dir: str = "logs", indexes_dir: str = "indexes"):
        """Initialize RAG pipeline components."""
        self.logs_dir = logs_dir
        self.indexes_dir = indexes_dir
        self.logger = self._setup_logging()

        # Ensure directories exist
        Path(self.logs_dir).mkdir(exist_ok=True)
        Path(self.indexes_dir).mkdir(exist_ok=True)

        # Configuration
        self.chunk_size = 800
        self.chunk_overlap = 120
        self.top_k = 3
        self.model_name = "all-MiniLM-L6-v2"
        self.ollama_model = "llama3"

        # Initialize components
        self.emb_model: Optional[SentenceTransformer] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.llm = None

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("DocQuery")
        logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

        # File handler
        log_file = os.path.join(self.logs_dir, "error.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def get_file_hash(self, file_path: str) -> str:
        """Generate SHA256 hash of PDF file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def initialize_embeddings(self) -> bool:
        """Initialize embeddings model with error handling."""
        try:
            # If an embedding model already exists (set by app cache), reuse it
            if getattr(self, "emb_model", None) is None:
                self.emb_model = SentenceTransformer(self.model_name, device="cpu")
            self.logger.info("Embeddings model initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings: {str(e)}")
            return False

    def initialize_llm(self) -> bool:
        """Initialize Ollama LLM with error handling and tuned params."""
        try:
            # Try preferred params first
            try:
                self.llm = OllamaLLM(model=self.ollama_model, base_url="http://localhost:11434", temperature=0, num_predict=256)
            except TypeError:
                # Fallback to model_kwargs if direct args unsupported
                self.llm = OllamaLLM(model=self.ollama_model, base_url="http://localhost:11434", model_kwargs={"temperature": 0, "num_predict": 256})
            # Don't perform a blocking test call here; defer connectivity errors to invocation time.
            self.logger.info("Ollama LLM object created (connectivity will be checked on first invoke)")
            return True
        except Exception as e:
            # Detect memory-related errors from Ollama and attempt smaller fallback models
            msg = str(e)
            self.logger.error(f"Failed to initialize Ollama LLM with model '{self.ollama_model}': {msg}")

            # Common Ollama memory error message contains 'requires more system memory'
            if "memory" in msg.lower() or "requires more system memory" in msg.lower():
                fallback_models = [
                    # try a smaller variant - adjust to models available in your Ollama install
                    f"{self.ollama_model}-mini",
                    "llama2-mini",
                    "llama2",
                    "llama3-small",
                ]
                for fm in fallback_models:
                    try:
                        self.logger.info(f"Attempting fallback Ollama model: {fm}")
                        try:
                            self.llm = OllamaLLM(model=fm, base_url="http://localhost:11434", temperature=0, num_predict=128)
                        except TypeError:
                            self.llm = OllamaLLM(model=fm, base_url="http://localhost:11434", model_kwargs={"temperature": 0, "num_predict": 128})
                        self.ollama_model = fm
                        self.logger.info(f"Fell back to Ollama model: {fm}")
                        return True
                    except Exception as e2:
                        self.logger.warning(f"Fallback model {fm} failed: {str(e2)}")

                # If no fallback succeeded, provide actionable guidance.
                self.logger.error(
                    "Ollama model initialization failed due to insufficient system memory. "
                    "Options: 1) Run a smaller model in Ollama (e.g. a '*-mini' variant), "
                    "2) Increase available system memory, or 3) Configure a remote Ollama server with more RAM."
                )
            return False

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            pdf_reader = pypdf.PdfReader(file_path)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")

            if not text.strip():
                raise ValueError("No text could be extracted from PDF")

            self.logger.info(f"Extracted {len(text)} characters from {file_path}")
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def create_vectorstore(self, text: str) -> Optional[SimpleVectorStore]:
        """Create FAISS vectorstore from text using sentence-transformers + faiss."""
        try:
            if not self.emb_model:
                if not self.initialize_embeddings():
                    return None

            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            self.logger.info(f"Created {len(chunks)} chunks from text")

            if len(chunks) == 0:
                self.logger.error("No chunks created from text")
                return None

            # Compute embeddings using sentence-transformers
            vectors = self.emb_model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)

            # Normalize vectors for cosine similarity
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

            dim = vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)

            vectorstore = SimpleVectorStore(index=index, texts=chunks, vectors=vectors, emb_model=self.emb_model)
            self.logger.info("FAISS vectorstore created successfully")
            return vectorstore
        except Exception as e:
            self.logger.error(f"Error creating vectorstore: {str(e)}")
            return None

    def save_vectorstore(self, vectorstore: SimpleVectorStore, index_path: str) -> bool:
        """Save FAISS vectorstore to disk."""
        try:
            vectorstore.save_local(index_path)
            self.logger.info(f"Vectorstore saved to {index_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving vectorstore: {str(e)}")
            return False

    def load_vectorstore(self, index_path: str) -> Optional[SimpleVectorStore]:
        """Load FAISS vectorstore from disk."""
        try:
            if not self.emb_model:
                if not self.initialize_embeddings():
                    return None

            vectorstore = SimpleVectorStore.load_local(index_path, emb_model=self.emb_model)
            if vectorstore:
                self.logger.info(f"Vectorstore loaded from {index_path}")
            return vectorstore
        except Exception as e:
            self.logger.error(f"Error loading vectorstore: {str(e)}")
            return None

    def get_index_path(self, file_hash: str) -> str:
        """Get index path based on file hash."""
        return os.path.join(self.indexes_dir, f"index_{file_hash}")

    def answer_question(
        self,
        vectorstore: SimpleVectorStore,
        question: str
    ) -> tuple[str, dict]:
        """Answer question using RAG pipeline.

        Returns (answer, timings)
        """
        try:
            if not self.llm:
                if not self.initialize_llm():
                    return "Error: Unable to initialize language model", {}

            # Retrieve relevant chunks with timing
            try:
                t0 = time.time()
                docs = vectorstore.similarity_search(question, k=self.top_k)
                sim_time = time.time() - t0
                self.logger.info(f"Similarity search time: {sim_time:.3f}s")
            except Exception:
                self.logger.error("Vectorstore does not support similarity_search().")
                return "Error: incompatible vectorstore", {}

            # Build context from retrieved docs, limit to 4000 chars
            max_context_chars = 4000
            context_parts = []
            current_len = 0
            for d in docs:
                chunk = d.get("page_content", "")
                if not chunk:
                    continue
                if current_len + len(chunk) > max_context_chars:
                    # Add portion that fits, then break
                    remaining = max_context_chars - current_len
                    if remaining > 0:
                        context_parts.append(chunk[:remaining])
                        current_len += remaining
                    break
                context_parts.append(chunk)
                current_len += len(chunk)

            context_text = "\n\n".join(context_parts)

            # Create prompt with concise instruction
            prompt = (
                "You are a helpful assistant. Use ONLY the provided context to answer the question.\n"
                "If the answer is not in the context, respond with EXACTLY: \"Not available in document.\"\n"
                "Answer concisely in 3-5 sentences.\n\n"
                f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
            )

            # Query Ollama LLM with timing
            try:
                t1 = time.time()
                raw = self.llm.invoke(prompt)
                llm_time = time.time() - t1
                self.logger.info(f"LLM generation time: {llm_time:.3f}s")
                answer = raw if isinstance(raw, str) else str(raw)
            except Exception as e:
                self.logger.error(f"LLM invocation failed: {str(e)}")
                return f"Error processing question: {str(e)}", {"similarity_time": sim_time}

            # If LLM returned empty or irrelevant, ensure fallback exact message check
            if not answer or "not available" in answer.lower():
                return "Not available in document.", {"similarity_time": sim_time, "llm_time": llm_time}

            self.logger.info(f"Generated answer for question: {question[:50]}...")
            return answer, {"similarity_time": sim_time, "llm_time": llm_time}
        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}")
            return f"Error processing question: {str(e)}", {}

    def process_pdf(self, file_path: str) -> tuple[Optional[SimpleVectorStore], str]:
        """
        Main processing pipeline: load PDF, extract text, create/load vectorstore.
        Returns tuple of (vectorstore, status_message)
        """
        try:
            # Get file hash for index caching
            file_hash = self.get_file_hash(file_path)
            index_path = self.get_index_path(file_hash)

            # Check if index exists
            if os.path.exists(index_path):
                vectorstore = self.load_vectorstore(index_path)
                if vectorstore:
                    return vectorstore, "Using cached embeddings index"

            # Extract text from PDF
            text = self.extract_text_from_pdf(file_path)

            # Create vectorstore
            vectorstore = self.create_vectorstore(text)
            if not vectorstore:
                return None, "Failed to create vectorstore"

            # Save vectorstore
            self.save_vectorstore(vectorstore, index_path)

            return vectorstore, "Successfully processed PDF and created embeddings"
        except Exception as e:
            error_msg = f"Error processing PDF: {str(e)}"
            self.logger.error(error_msg)
            return None, error_msg
