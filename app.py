"""
DocQuery - Offline PDF Q&A Application
A fast, beautiful, production-ready local AI app for PDF question answering.

Version 2.0 - Complete rewrite with performance optimizations and premium UI
"""
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import streamlit as st
from streamlit_option_menu import option_menu

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.rag_pipeline import DocQueryPipeline, SimpleVectorStore
from backend.config import PERFORMANCE_MODES, DEFAULT_PERFORMANCE_MODE
from frontend.styles import get_premium_css
from frontend.components import (
    PremiumComponents,
    SidebarComponents,
    ChatComponents,
    ChatMessage,
    init_session_state
)
from frontend.sections import render_sidebar, render_main_content


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="DocQuery - Offline PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "DocQuery v2.0 - Offline PDF Q&A with local AI"
    }
)


# =============================================================================
# CACHED RESOURCES
# =============================================================================
@st.cache_resource
def get_pipeline() -> DocQueryPipeline:
    """Get or create the RAG pipeline instance."""
    return DocQueryPipeline(
        logs_dir="logs",
        indexes_dir="indexes",
        performance_mode=st.session_state.get("performance_mode", DEFAULT_PERFORMANCE_MODE)
    )


@st.cache_resource
def get_embedding_model():
    """Pre-load embedding model for faster first use."""
    from sentence_transformers import SentenceTransformer
    from backend.config import EMBEDDING_MODEL, EMBEDDING_DEVICE
    
    try:
        model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "vectorstore": None,
        "pdf_name": None,
        "pdf_hash": None,
        "messages": [],
        "last_timings": {},
        "last_sources": [],
        "show_sources": True,
        "show_timings": True,
        "performance_mode": DEFAULT_PERFORMANCE_MODE,
        "top_k": 3,
        "processing": False,
        "page": "Chat",
        "ollama_connected": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# CALLBACKS
# =============================================================================
def handle_performance_mode_change(mode: str):
    """Handle performance mode change."""
    st.session_state.performance_mode = mode
    pipeline = get_pipeline()
    pipeline.set_performance_mode(mode)
    st.rerun()


def handle_sources_toggle(show: bool):
    """Handle sources toggle."""
    st.session_state.show_sources = show


def handle_timings_toggle(show: bool):
    """Handle timings toggle."""
    st.session_state.show_timings = show


def handle_top_k_change(k: int):
    """Handle top-k change."""
    st.session_state.top_k = k


def handle_clear_chat():
    """Clear chat history."""
    st.session_state.messages = []
    st.session_state.last_sources = []
    st.session_state.last_timings = {}
    st.rerun()


def handle_reset_pdf():
    """Reset PDF and clear all state."""
    st.session_state.vectorstore = None
    st.session_state.pdf_name = None
    st.session_state.pdf_hash = None
    st.session_state.messages = []
    st.session_state.last_sources = []
    st.session_state.last_timings = {}
    st.rerun()


def get_file_hash(file_path: str) -> str:
    """Generate a quick hash for file identification."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(8192), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:16]


# =============================================================================
# PDF PROCESSING
# =============================================================================
def process_uploaded_pdf(uploaded_file) -> Tuple[Optional[SimpleVectorStore], str]:
    """Process an uploaded PDF file."""
    try:
        # Save to temp directory
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        temp_path = temp_dir / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getvalue())
        
        # Get pipeline and process
        pipeline = get_pipeline()
        
        with st.spinner("Processing PDF..."):
            vectorstore, message = pipeline.process_pdf(str(temp_path))
            
            if vectorstore:
                st.session_state.pdf_hash = pipeline.get_file_hash(str(temp_path))
            
            return vectorstore, message
            
    except Exception as e:
        return None, f"Error: {str(e)}"


# =============================================================================
# QUESTION ANSWERING
# =============================================================================
def answer_question(question: str) -> Tuple[str, List[Dict], Dict]:
    """Answer a question using the RAG pipeline."""
    try:
        vectorstore = st.session_state.vectorstore
        if vectorstore is None:
            return "No PDF uploaded. Please upload a PDF first.", [], {}
        
        pipeline = get_pipeline()
        top_k = st.session_state.top_k
        
        # Start timing
        start_time = time.perf_counter()
        
        # Retrieve sources
        try:
            sources = vectorstore.similarity_search(question, k=top_k)
        except Exception as e:
            return f"Error retrieving sources: {str(e)}", [], {}
        
        retrieval_time = time.perf_counter() - start_time
        
        # Get answer from pipeline
        answer, timings = pipeline.answer_question(vectorstore, question, top_k=top_k)
        
        # Add retrieval time
        timings["retrieval"] = retrieval_time
        timings["total"] = sum(v for k, v in timings.items() if isinstance(v, (int, float)))
        
        # Store for display
        st.session_state.last_sources = sources
        st.session_state.last_timings = timings
        
        return answer, sources, timings
        
    except Exception as e:
        return f"Error: {str(e)}", [], {}


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application entry point."""
    
    # Initialize
    init_session_state()
    
    # Inject premium CSS
    st.markdown(get_premium_css(), unsafe_allow_html=True)
    
    # Check Ollama connection
    if st.session_state.ollama_connected is None:
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            st.session_state.ollama_connected = response.status_code == 200
        except:
            st.session_state.ollama_connected = False
    
    # Pre-load embedding model
    if st.session_state.vectorstore is None:
        get_embedding_model()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:10px 0 20px 0;">
            <div style="font-size:2.2rem;margin-bottom:8px;">📄</div>
            <div style="font-size:1.4rem;font-weight:700;color:#E6E8F2;">DocQuery</div>
            <div style="font-size:0.8rem;color:rgba(229,232,242,0.6);margin-top:4px;">
                Offline PDF Q&A
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Status indicators
        col1, col2 = st.columns(2)
        with col1:
            ollama_status = "🟢" if st.session_state.ollama_connected else "🔴"
            st.markdown(f"<div style='text-align:center;font-size:0.8rem;'>{ollama_status} Ollama</div>", unsafe_allow_html=True)
        with col2:
            indexed_status = "✅" if st.session_state.vectorstore else "⏳"
            st.markdown(f"<div style='text-align:center;font-size:0.8rem;'>{indexed_status} Indexed</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        page = option_menu(
            menu_title=None,
            options=["Chat", "About"],
            icons=["chat-dots-fill", "info-circle-fill"],
            default_index=0 if st.session_state.page == "Chat" else 1
        )
        st.session_state.page = page
        
        st.markdown("---")
        
        # Settings (only show on Chat page)
        if page == "Chat":
            # Performance mode selector
            st.markdown("#### ⚡ Mode")
            
            modes = [
                ("fast", "🚀 Fast", "Quick responses"),
                ("balanced", "⚖️ Balanced", "Speed + Quality"),
                ("quality", "🎯 Quality", "Best answers")
            ]
            
            current_mode = st.session_state.performance_mode
            mode_cols = st.columns(3)
            
            for i, (mode, label, desc) in enumerate(modes):
                with mode_cols[i]:
                    if st.button(
                        label,
                        key=f"mode_{mode}",
                        use_container_width=True,
                        type="primary" if mode == current_mode else "secondary"
                    ):
                        handle_performance_mode_change(mode)
            
            st.caption(f"Current: {[d[2] for d in modes if d[0] == current_mode][0]}")
            
            st.markdown("---")
            
            # Display toggles
            st.markdown("#### 👁️ Display")
            
            new_sources = st.toggle(
                "Show Sources",
                value=st.session_state.show_sources,
                key="toggle_sources_v2"
            )
            if new_sources != st.session_state.show_sources:
                handle_sources_toggle(new_sources)
            
            new_timings = st.toggle(
                "Show Timings", 
                value=st.session_state.show_timings,
                key="toggle_timings_v2"
            )
            if new_timings != st.session_state.show_timings:
                handle_timings_toggle(new_timings)
            
            st.markdown("---")
            
            # Retrieval settings
            st.markdown("#### 🔧 Retrieval")
            
            new_top_k = st.slider(
                "Top-K Chunks",
                min_value=1,
                max_value=10,
                value=st.session_state.top_k,
                step=1,
                key="top_k_slider"
            )
            if new_top_k != st.session_state.top_k:
                handle_top_k_change(new_top_k)
            
            st.markdown("---")
            
            # Action buttons
            st.markdown("#### 🧹 Actions")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    handle_clear_chat()
            with col_b:
                if st.button("🔄 Reset PDF", use_container_width=True):
                    handle_reset_pdf()
    
    # Main content area
    if st.session_state.page == "About":
        render_about_page()
    else:
        render_chat_page()


def render_about_page():
    """Render the About page."""
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">About DocQuery</h1>
        <p class="hero-subtitle">A fast, beautiful, offline PDF Q&A application</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    col1, col2 = st.columns(2)
    
    with col1:
        PremiumComponents.glass_card(
            title="🧠 AI Features",
            icon="",
            children=lambda: st.markdown("""
            - **Local LLM** - Uses Ollama for offline AI
            - **Smart Retrieval** - FAISS vector search
            - **Fast Embeddings** - SentenceTransformers
            - **Smart Caching** - Reuses processed PDFs
            - **Performance Modes** - Fast/Balanced/Quality
            """)
        )
    
    with col2:
        PremiumComponents.glass_card(
            title="🎨 UI Features",
            icon="",
            children=lambda: st.markdown("""
            - **Premium Dark Theme** - Beautiful glassmorphism
            - **Real-time Timings** - See performance
            - **Source Citations** - Know what AI used
            - **Chat History** - Conversation continuity
            - **Responsive Design** - Works on all screens
            """)
        )
    
    # Tech stack
    st.markdown("### 🛠️ Tech Stack")
    
    st.markdown("""
    | Component | Technology |
    |-----------|------------|
    | Frontend | Streamlit |
    | Vector Store | FAISS |
    | Embeddings | SentenceTransformers |
    | LLM | Ollama (llama3) |
    | PDF Processing | pypdf |
    """)
    
    # Getting started
    st.markdown("### 🚀 Getting Started")
    
    PremiumComponents.glass_card(
        title="Quick Start Guide",
        children=lambda: st.markdown("""
        1. **Install dependencies**: `pip install -r requirements.txt`
        2. **Start Ollama**: Run `ollama serve` in terminal
        3. **Run app**: `streamlit run app.py`
        4. **Upload PDF**: Drag and drop any PDF
        5. **Ask questions**: Type in the chat input
        
        ---
        
        **Tips:**
        - First PDF processing takes longer (downloading embedding model)
        - Same PDFs are cached for instant loading
        - Use "Fast" mode for quickest responses
        - Check "Show Timings" to see performance metrics
        """)
    )


def render_chat_page():
    """Render the main chat page."""
    
    # Status bar
    render_status_bar()
    
    # Hero
    render_hero()
    
    # Step indicator
    render_step_indicator()
    
    # Main content based on state
    if st.session_state.vectorstore is None:
        render_upload_section()
    else:
        render_chat_section()


def render_status_bar():
    """Render the status bar."""
    if st.session_state.vectorstore is not None:
        status = "ready"
        status_text = "Ready"
    elif st.session_state.processing:
        status = "processing"
        status_text = "Processing..."
    else:
        status = "idle"
        status_text = "Idle"
    
    pdf_info = f"📄 {st.session_state.pdf_name}" if st.session_state.pdf_name else ""
    
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;flex-wrap:wrap;">
        <span class="status-badge {status}">
            <span class="status-dot {'online' if status == 'ready' else ''}"></span>
            {status_text}
        </span>
        {f'<span style="color:rgba(229,232,242,0.6);font-size:0.85rem;">{pdf_info}</span>' if pdf_info else ''}
    </div>
    """, unsafe_allow_html=True)


def render_hero():
    """Render the hero section."""
    badges = [
        {"text": "⚡ Fast", "type": "primary"},
        {"text": "🔒 Offline", "type": "success"},
        {"text": "🧠 Local AI", "type": "info"}
    ]
    
    if st.session_state.vectorstore is not None:
        badges.append({"text": "✅ Indexed", "type": "success"})
    
    badges_html = "".join([
        f'<span class="badge badge-{b["type"]}">{b["text"]}</span>'
        for b in badges
    ])
    
    st.markdown(f"""
    <div class="hero-section">
        <div style="position: relative; z-index: 1;">
            <h1 class="hero-title">DocQuery</h1>
            <p class="hero-subtitle">Upload a PDF → Ask questions → Get instant answers (fully offline)</p>
            <div class="badge-container">{badges_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_step_indicator():
    """Render step indicator."""
    if st.session_state.vectorstore is None:
        steps = ["1) Upload PDF", "2) Ask Questions"]
        current = 0
    else:
        steps = ["✓ Uploaded", "2) Ask Questions"]
        current = 1
    
    html = '<div class="step-indicator">'
    for i, step in enumerate(steps):
        if i < current:
            status = "completed"
            number = "✓"
        elif i == current:
            status = "active"
            number = str(i + 1)
        else:
            status = ""
            number = str(i + 1)
        
        html += f'<span class="step-pill {status}"><span class="step-number">{number}</span>{step}</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_upload_section():
    """Render the PDF upload section."""
    col1, col2 = st.columns([1.3, 1], gap="large")
    
    with col1:
        PremiumComponents.glass_card(
            title="📤 Upload Your PDF",
            icon="📄",
            children=lambda: render_upload_form()
        )
    
    with col2:
        PremiumComponents.glass_card(
            title="✨ Features",
            icon="🎯",
            children=lambda: render_features_list()
        )


def render_upload_form():
    """Render the upload form."""
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload a PDF document to start asking questions"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        process_btn = st.button(
            "🚀 Process PDF",
            use_container_width=True,
            disabled=(uploaded_file is None),
            type="primary"
        )
    
    with col2:
        demo_btn = st.button(
            "✨ Demo Mode",
            use_container_width=True
        )
    
    if process_btn and uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            vectorstore, message = process_uploaded_pdf(uploaded_file)
            
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_name = uploaded_file.name
                PremiumComponents.success_toast(f"PDF ready: {len(vectorstore.texts)} chunks")
                st.rerun()
            else:
                st.error(f"Error: {message}")
    
    if demo_btn:
        st.info("👆 Upload any PDF to get started! The demo shows the UI capabilities.")


def render_features_list():
    """Render the features list."""
    st.markdown("""
    <div style="color: rgba(229,232,242,0.7); font-size: 0.9rem;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <span style="color: #4ADE80;">✅</span> Chat-style Q&A
        </div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <span style="color: #4ADE80;">✅</span> Source citations
        </div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <span style="color: #4ADE80;">✅</span> Performance timings
        </div>
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <span style="color: #4ADE80;">✅</span> Smart caching
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="color: #4ADE80;">✅</span> Premium dark UI
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_chat_section():
    """Render the chat section."""
    # Chat header
    num_chunks = len(st.session_state.vectorstore.texts) if st.session_state.vectorstore else 0
    
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;flex-wrap:wrap;gap:12px;">
        <div>
            <h3 style="margin:0;font-weight:600;color:#E6E8F2;">💬 Chat</h3>
            <p style="margin:4px 0 0 0;font-size:0.85rem;color:rgba(229,232,242,0.6);">📄 {st.session_state.pdf_name}</p>
        </div>
        <span class="badge badge-info">📊 {num_chunks} chunks</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Show empty state if no messages
    if not st.session_state.messages:
        PremiumComponents.empty_state(
            icon="💬",
            title="Start a conversation",
            description="Ask questions about your PDF and get instant answers"
        )
    
    # Render message history
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f"""
        <div class="chat-message">
            <div class="chat-avatar {msg['role']}">{avatar}</div>
            <div class="chat-bubble {msg['role']}">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Chat input
    question = st.chat_input("Ask something about your PDF...")
    
    if question:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        
        # Get response
        with st.spinner("Thinking..."):
            answer, sources, timings = answer_question(question)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
        
        # Show response
        with st.chat_message("assistant"):
            st.markdown(f'<div class="chat-bubble assistant">{answer}</div>', unsafe_allow_html=True)
            
            # Show timings
            if st.session_state.show_timings and timings:
                render_timings(timings)
            
            # Show sources
            if st.session_state.show_sources and sources:
                with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                    for i, source in enumerate(sources):
                        content = source.get("page_content", "")
                        score = source.get("score")
                        preview = content[:600] + "..." if len(content) > 600 else content
                        
                        score_html = f'<span class="source-score">Score: {score:.3f}</span>' if score else ""
                        
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">
                                <span class="source-chunk-num">📄 Chunk {i + 1}</span>
                                {score_html}
                            </div>
                            <div class="source-preview">{preview.replace(chr(10), '<br>')}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.rerun()


def render_timings(timings: Dict):
    """Render timings panel."""
    if not timings:
        return
    
    # Filter numeric values
    timing_items = [(k, v) for k, v in timings.items() if isinstance(v, (int, float))]
    
    if not timing_items:
        return
    
    total = sum(v for _, v in timing_items)
    
    html = '<div class="timings-panel"><div style="font-weight:600;color:#E6E8F2;margin-bottom:12px;">⏱️ Performance</div>'
    
    for label, value in timing_items:
        label_formatted = label.replace("_", " ").title()
        html += f"""
        <div class="timing-row">
            <span class="timing-label">{label_formatted}</span>
            <span class="timing-value">{value:.3f}s</span>
        </div>
        """
    
    html += f"""
    <div class="timing-row">
        <span class="timing-label" style="font-weight:600;">Total</span>
        <span class="timing-value total">{total:.3f}s</span>
    </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()

