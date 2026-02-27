import streamlit as st
import subprocess
import sys
import os
from pathlib import Path
import time

from rag_pipeline import DocQueryPipeline


# Page configuration
st.set_page_config(
    page_title="DocQuery - Offline PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
    }
    .stSpinner {
        color: #FF6B6B;
    }
    .diagnostic-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success {
        color: #28a745;
    }
    .error {
        color: #dc3545;
    }
    .warning {
        color: #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def initialize_pipeline():
    """Initialize RAG pipeline."""
    return DocQueryPipeline(logs_dir="logs", indexes_dir="indexes")

@st.cache_resource
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load and cache sentence-transformers embedding model (CPU).

    Returns a SentenceTransformer instance cached for the session.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device="cpu")


def check_ollama_status():
    """Check if Ollama is running and llama3 model exists."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            models = result.stdout
            return "running" if "llama3" in models else "no_model"
        return "not_running"
    except FileNotFoundError:
        return "not_installed"
    except Exception:
        return "error"


def get_system_diagnostics():
    """Gather system and package diagnostic information."""
    try:
        import langchain
        import langchain_core
        import langchain_community
        import langchain_huggingface
        import langchain_ollama
        import sentence_transformers
        import faiss
        import pypdf
        import streamlit
        
        diagnostics = {
            "Python Version": f"{sys.version.split()[0]}",
            "streamlit": streamlit.__version__,
            "langchain": langchain.__version__,
            "langchain-core": langchain_core.__version__,
            "langchain-community": getattr(langchain_community, "__version__", "installed"),
            "langchain-huggingface": getattr(langchain_huggingface, "__version__", "installed"),
            "langchain-ollama": getattr(langchain_ollama, "__version__", "installed"),
            "sentence-transformers": sentence_transformers.__version__,
            "faiss": faiss.__version__,
            "pypdf": pypdf.__version__,
        }
        return diagnostics
    except ImportError as e:
        return {"Error": f"Import error: {str(e)}"}


def display_diagnostics():
    """Display diagnostics section in sidebar."""
    with st.sidebar:
        st.markdown("---")
        with st.expander("🔧 Diagnostics", expanded=False):
            # System info
            st.write("**System Information:**")
            diagnostics = get_system_diagnostics()
            for key, value in diagnostics.items():
                st.code(f"{key}: {value}", language="text")
            
            # Ollama status
            st.write("**Ollama Status:**")
            ollama_status = check_ollama_status()
            
            if ollama_status == "running":
                st.markdown(
                    '<span class="success">✓ Ollama running with llama3 model</span>',
                    unsafe_allow_html=True
                )
            elif ollama_status == "no_model":
                st.markdown(
                    '<span class="warning">⚠ Ollama running but llama3 model not found</span>',
                    unsafe_allow_html=True
                )
            elif ollama_status == "not_running":
                st.markdown(
                    '<span class="error">✗ Ollama not running</span>',
                    unsafe_allow_html=True
                )
            elif ollama_status == "not_installed":
                st.markdown(
                    '<span class="error">✗ Ollama not installed</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<span class="error">✗ Error checking Ollama</span>',
                    unsafe_allow_html=True
                )
            
            # Directories
            st.write("**Directories:**")
            data_dir = Path("data")
            indexes_dir = Path("indexes")
            logs_dir = Path("logs")
            
            st.code(f"data/: {data_dir.exists()}", language="text")
            st.code(f"indexes/: {indexes_dir.exists()}", language="text")
            st.code(f"logs/: {logs_dir.exists()}", language="text")


def main():
    """Main Streamlit application."""
    st.title("📄 DocQuery - Offline PDF Question Answering")
    st.markdown("""
    Upload a PDF and ask questions about its content. 
    All processing is done locally using Ollama (llama3) and sentence-transformers.
    """)
    
    # Initialize pipeline
    pipeline = initialize_pipeline()
    
    # Display diagnostics
    display_diagnostics()
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Upload PDF")
        uploaded_file = st.file_uploader(
            "Select a PDF file",
            type="pdf"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            file_path = data_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✓ File uploaded: {uploaded_file.name}")
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### Process Document")
            
            if st.button("Process PDF", type="primary", use_container_width=True):
                with st.spinner("Processing PDF..."):
                    # Ensure cached embedding model is used
                    emb = get_embedding_model(pipeline.model_name)
                    pipeline.emb_model = emb

                    # Compute file hash and index path (for UI/storage)
                    try:
                        file_hash = pipeline.get_file_hash(str(file_path))
                        index_dir = pipeline.get_index_path(file_hash)
                    except Exception:
                        file_hash = None
                        index_dir = None

                    vectorstore, status = pipeline.process_pdf(str(file_path))

                if vectorstore:
                    st.success(f"✓ {status}")
                    st.session_state.db = vectorstore
                    st.session_state.index_dir = index_dir
                    st.session_state.document_hash = file_hash
                    st.session_state.pdf_loaded = True
                else:
                    st.error(f"✗ {status}")
                    st.session_state.pdf_loaded = False
    
    # Question answering section
    if "pdf_loaded" in st.session_state and st.session_state.pdf_loaded:
        st.markdown("---")
        st.markdown("### Ask Questions")
        
        question = st.text_input(
            "Enter your question:",
            placeholder="What is the main topic of this document?",
            label_visibility="collapsed"
        )
        
        if question:
            col1, col2 = st.columns([1, 5])
            with col1:
                ask_button = st.button("Get Answer", type="primary", use_container_width=True)
            
            if ask_button:
                with st.spinner("Generating answer..."):
                    db = st.session_state.get("db")
                    if not db:
                        st.error("No document indexed. Click 'Process PDF' first.")
                    else:
                        answer, timings = pipeline.answer_question(db, question)

                st.markdown("### Answer")
                # Show the answer (no stack traces)
                try:
                    st.write(answer)
                except Exception:
                    st.write(str(answer))

                # Timing debug (temporary)
                try:
                    sim_t = timings.get("similarity_time")
                    llm_t = timings.get("llm_time")
                    st.info(f"Similarity search: {sim_t:.3f}s — LLM generation: {llm_t:.3f}s")
                except Exception:
                    pass
        
        # Clear cache option
        if st.button("Clear Document", use_container_width=False):
            st.session_state.pdf_loaded = False
            st.session_state.db = None
            st.session_state.index_dir = None
            st.session_state.document_hash = None
            st.rerun()
    
    else:
        if uploaded_file is None:
            st.info("👆 Upload a PDF file to get started")
        else:
            st.info("👆 Click 'Process PDF' to process the uploaded document")


if __name__ == "__main__":
    main()
