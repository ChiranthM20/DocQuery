import os
import time
from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu

from backend.rag_pipeline import DocQueryPipeline


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="DocQuery",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Premium CSS (cards + glow + responsive polish)
# ----------------------------
st.markdown(
    """
<style>
/* overall spacing */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* gradient header */
.hero {
  border-radius: 22px;
  padding: 18px 18px;
  background: radial-gradient(1200px circle at 10% 10%, rgba(124,58,237,.35), transparent 45%),
              radial-gradient(900px circle at 90% 30%, rgba(99,102,241,.25), transparent 40%),
              rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.10);
  box-shadow: 0 20px 60px rgba(0,0,0,.35);
}

/* glass cards */
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 16px;
  backdrop-filter: blur(10px);
  box-shadow: 0 14px 40px rgba(0,0,0,0.28);
  transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
}
.card:hover {
  transform: translateY(-2px);
  border-color: rgba(124,58,237,0.50);
  box-shadow: 0 18px 55px rgba(0,0,0,0.33);
}

/* small muted text */
.muted { opacity: .78; font-size: 0.92rem; }

/* tiny badge */
.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.06);
  margin-right: 8px;
  font-size: .86rem;
  opacity: .92;
}

/* answer bubble */
.bubble {
  background: rgba(124,58,237,0.12);
  border: 1px solid rgba(124,58,237,0.22);
  border-radius: 16px;
  padding: 14px 16px;
}

/* step pills */
.step-pill {
  display: inline-block;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.05);
  margin-right: 8px;
  font-size: .90rem;
}
.step-pill.active {
  border-color: rgba(124,58,237,.55);
  background: rgba(124,58,237,.14);
}

/* make buttons wider on mobile */
@media (max-width: 720px) {
  .block-container { padding-left: .8rem; padding-right: .8rem; }
}
</style>
""",
    unsafe_allow_html=True
)


# ----------------------------
# Cached pipeline
# ----------------------------
@st.cache_resource
def get_pipeline():
    return DocQueryPipeline(logs_dir="logs", indexes_dir="indexes")


# ----------------------------
# Session state init
# ----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []  # chat messages list of dicts

if "last_timings" not in st.session_state:
    st.session_state.last_timings = {}

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# ----------------------------
# Sidebar (navigation + settings)
# ----------------------------
with st.sidebar:
    st.markdown("### 📄 DocQuery")
    st.caption("Offline PDF Q&A (FAISS + Ollama)")

    page = option_menu(
        menu_title=None,
        options=["Chat", "About"],
        icons=["chat-dots-fill", "info-circle-fill"],
        default_index=0
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Settings")

    top_k = st.slider("Top-K Chunks", min_value=2, max_value=10, value=4, step=1)
    st.caption("Higher = more context, but slower.")

    show_sources = st.toggle("Show Sources", value=True)
    show_timings = st.toggle("Show Timings", value=True)

    st.markdown("---")
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🧼 Reset PDF", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.pdf_name = None
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_timings = {}
        st.rerun()


# ----------------------------
# Main UI
# ----------------------------
pipeline = get_pipeline()

# pre-initialize embeddings early so we can warn the user if the model
# download fails (common when truly offline). Without this the first PDF
# processing attempt might present the vague "Network Error" message.
if not pipeline.initialize_embeddings():
    st.warning(
        "⚠️ Embedding model could not be initialized. "
        "An internet connection is required the first time the model is downloaded. "
        "Please connect to the network and restart the app, or pre-download the model "
        "manually (see README)."
    )

st.markdown(
    """
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
    <div>
      <div style="font-size:1.6rem;font-weight:700;">DocQuery</div>
      <div class="muted">Upload a PDF → build local embeddings → ask questions like ChatGPT (offline).</div>
    </div>
    <div>
      <span class="badge">⚡ Fast UI</span>
      <span class="badge">🔒 Offline</span>
      <span class="badge">📌 Sources</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

st.write("")

# Step indicator
step_1 = "active" if st.session_state.vectorstore is None else ""
step_2 = "active" if st.session_state.vectorstore is not None else ""
st.markdown(
    f"""
<div style="margin: 8px 0 14px 0;">
  <span class="step-pill {step_1}">1) Upload PDF</span>
  <span class="step-pill {step_2}">2) Chat</span>
</div>
""",
    unsafe_allow_html=True
)

# ----------------------------
# Step 1: Upload + Process
# ----------------------------
if st.session_state.vectorstore is None:
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 1) Upload your PDF")
        pdf = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")

        colA, colB = st.columns([1, 1])
        with colA:
            process_btn = st.button("🚀 Process PDF", use_container_width=True, disabled=(pdf is None))
        with colB:
            demo_btn = st.button("✨ UI Demo Mode", use_container_width=True)

        st.caption("Tip: After processing once, it uses cached embeddings for the same PDF.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### What you’ll get")
        st.markdown(
            """
- ✅ Chat-style Q&A  
- ✅ Source chunks (expandable)  
- ✅ Timings (retrieval + LLM)  
- ✅ Clean, responsive UI  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if demo_btn:
        st.success("Demo mode: Upload any PDF to continue. The UI is ready ✨")

    if process_btn and pdf is not None:
        # Save uploaded PDF to temp folder
        tmp_dir = Path("temp_uploads")
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / pdf.name
        tmp_path.write_bytes(pdf.getvalue())

        with st.status("Processing your PDF…", expanded=True) as status:
            st.write("✅ Saving file")
            time.sleep(0.2)

            st.write("🧠 Building / loading embeddings index")
            t0 = time.time()
            vectorstore, msg = pipeline.process_pdf(str(tmp_path))
            t_process = time.time() - t0

            if vectorstore is None:
                status.update(label="Failed to process PDF", state="error", expanded=True)
                # if the error message looks network-related, add a tip
                if "network" in msg.lower() or "connection" in msg.lower():
                    st.error(msg)
                    st.error(
                        "⚠️ It looks like the embedding model could not be downloaded. "
                        "This usually happens when running offline for the first time. "
                        "Make sure you have an internet connection so the model can be fetched, "
                        "or manually download/copy the model to the HuggingFace cache before retrying."
                    )
                else:
                    st.error(msg)
            else:
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_name = pdf.name
                status.update(label=f"Done: {msg}", state="complete", expanded=False)
                st.toast(f"✅ PDF Ready in {t_process:.2f}s", icon="✅")
                st.rerun()


# ----------------------------
# Step 2: Chat UI
# ----------------------------
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 2) Chat with your PDF")
    st.caption(f"📌 Current PDF: **{st.session_state.pdf_name}**")
    st.markdown("</div>", unsafe_allow_html=True)

    # Show chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input
    question = st.chat_input("Ask something about the PDF…")

    if question:
        # store user msg
        st.session_state.messages.append({"role": "user", "content": question})

        # show assistant response area
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                # Retrieve sources once (for display)
                t_retr = time.time()
                docs = st.session_state.vectorstore.similarity_search(question, k=top_k)
                retr_time = time.time() - t_retr

                # Ask pipeline (LLM) (pipeline does its own retrieval again internally,
                # but we keep this for now to avoid changing your backend file)
                answer, timings = pipeline.answer_question(st.session_state.vectorstore, question)

                # Combine timings nicely
                st.session_state.last_timings = {
                    "retrieval_ui": retr_time,
                    **(timings or {})
                }
                st.session_state.last_sources = docs or []

                st.markdown(f'<div class="bubble">{answer}</div>', unsafe_allow_html=True)

                # timings
                if show_timings:
                    t = st.session_state.last_timings
                    st.write("")
                    st.markdown(
                        f"""
<span class="badge">🔎 Retrieval(UI): {t.get("retrieval_ui", 0):.2f}s</span>
<span class="badge">🔎 Similarity(Pipeline): {t.get("similarity_time", 0):.2f}s</span>
<span class="badge">🧠 LLM: {t.get("llm_time", 0):.2f}s</span>
""",
                        unsafe_allow_html=True
                    )

                # sources
                if show_sources:
                    with st.expander("📌 Sources used (top chunks)", expanded=False):
                        if not st.session_state.last_sources:
                            st.info("No sources found.")
                        else:
                            for i, d in enumerate(st.session_state.last_sources, start=1):
                                score = d.get("score", None)
                                chunk = d.get("page_content", "")
                                st.markdown(f"**Chunk {i}**" + (f" — similarity: `{score:.3f}`" if score is not None else ""))
                                st.write(chunk[:1200] + ("…" if len(chunk) > 1200 else ""))
                                st.markdown("---")

        # store assistant msg
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()


# ----------------------------
# About page
# ----------------------------
if page == "About":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## About")
    st.markdown(
        """
**DocQuery** is an offline PDF Q&A system:

- Extracts text from PDF  
- Splits into chunks  
- Embeds with SentenceTransformers (CPU)  
- Stores vectors in FAISS locally  
- Uses Ollama to answer using only retrieved context  

This UI is designed for **beautiful interactive demos**: chat UX, cards, hover effects, timings, and sources.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)