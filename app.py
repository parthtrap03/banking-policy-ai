"""
Banking Policy Intelligence Platform - Enterprise Streamlit UI

Features:
- Clean enterprise light theme (ChatGPT-style)
- Multi-session chat interface in sidebar
- Source citation cards with expandable details
- Suggested question chips centrally aligned
- Thumbs up/down feedback on every answer
"""

import streamlit as st
import time
import uuid
from pathlib import Path

# Must be the first Streamlit call
st.set_page_config(
    page_title="Ask FactorAvenue",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# PREMIUM ENTERPRISE LIGHT THEME CSS
# ============================================================================

CUSTOM_CSS = """
<style>
/* === IMPORTS === */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* === GLOBAL === */
.stApp {
    font-family: 'Inter', sans-serif;
    background-color: #ffffff;
    color: #172A47;
}

/* === HEADER === */
.main-header {
    padding: 1rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid #ECEFF4;
    text-align: center;
}

.main-header h1 {
    color: #1B2A4A;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}

.main-header p {
    color: #5E6E82;
    font-size: 0.95rem;
    margin: 0.5rem 0 0 0;
}

/* === SUGGESTED QUESTIONS === */
.suggestion-category {
    color: #38b5b1;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
}

/* Make st.button styling for suggestions cleaner */
div.stButton > button[kind="secondary"] {
    border-radius: 6px;
    border: 1px solid #1B2A4A;
    background-color: #ffffff;
    color: #1B2A4A;
    font-weight: 600;
    transition: all 0.2s ease;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: #1B2A4A;
    background-color: #1B2A4A;
    color: #ffffff;
}


/* === SOURCE CARDS === */
.source-card {
    background-color: #F4F7F9;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: background-color 0.2s ease;
}

.source-card:hover {
    background-color: #E2E8F0;
}

.source-card .source-type {
    color: #5E6E82;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.source-card .source-section {
    color: #172A47;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.3rem;
}

.source-card .source-authority {
    color: #48C3EB;
    font-size: 0.75rem;
    margin-top: 0.2rem;
    font-weight: 600;
}

.source-card .source-excerpt {
    color: #334155;
    font-size: 0.85rem;
    margin-top: 0.6rem;
    line-height: 1.5;
    border-top: 1px solid #E2E8F0;
    padding-top: 0.6rem;
}

/* === INTENT BADGE === */
.intent-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    margin-right: 0.4rem;
    border: 1px solid #E2E8F0;
    background-color: #ffffff;
    color: #1B2A4A;
}

/* === DIVIDER === */
.clean-divider {
    height: 1px;
    background: #ECEFF4;
    margin: 1.5rem 0;
}

/* === SIDEBAR OVERRIDES === */
[data-testid="stSidebar"] {
    background-color: #F4F7F9;
    border-right: 1px solid #E2E8F0;
}

/* Hide Streamlit default top padding */
.css-18e3th9 {
    padding-top: 2rem;
}

/* Chat bubble styling overrides to look more integrated */
div[data-testid="stChatMessage"] {
    background-color: transparent;
    padding: 1.5rem 0;
    border-bottom: 1px solid #ECEFF4;
}
div[data-testid="stChatMessage"]:nth-of-type(even) {
    background-color: #F8FAFC;
}

/* === CHAT INPUT (darker, visible on white bg) === */
[data-testid="stChatInput"] {
    background-color: #172A47 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(23, 42, 71, 0.18) !important;
}
[data-testid="stChatInput"] > div {
    background-color: #172A47 !important;
    border: 1px solid #2C3E5C !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #172A47 !important;
    color: #FFFFFF !important;
    caret-color: #FFFFFF !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #B8C4D6 !important;
    opacity: 1 !important;
}
[data-testid="stChatInput"] button {
    background-color: #2C3E5C !important;
    color: #FFFFFF !important;
}
[data-testid="stChatInput"] button:hover {
    background-color: #3A5278 !important;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# INITIALIZATION
# ============================================================================

@st.cache_resource
def load_rag_system():
    """Initialize the RAG system (cached across reruns).
    Uses FastRAG (pre-built FAISS index) if available — cold start ~3-5s.
    Falls back to full system if index not built — cold start ~50s.
    """
    from pathlib import Path

    if Path("./search_index/faiss_index.bin").exists():
        from fast_rag import FastRAG
        rag = FastRAG(enable_reranker=True)
        class _Stats:
            def get_retrieval_stats(self):
                return {
                    'total_chunks': rag.index.ntotal,
                    'retrieval_method': 'FAISS+BM25+Rerank',
                }
        rag.retriever = _Stats()
        return rag

    from legal_rag_system import BankingPolicyRAG, HybridStoreManager, BankingDocumentProcessor

    processor = BankingDocumentProcessor()
    documents = processor.load_documents()
    business_info = {
        'company_name': 'Banking Policy Intelligence',
        'industry': 'Banking / Financial Services',
    }
    documents = processor.enrich_metadata(documents, business_info)
    chunks = processor.smart_chunk(documents)

    store_manager = HybridStoreManager()
    persist_dir = store_manager.persist_dir

    if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        vectordb = store_manager.load_vector_store()
    else:
        vectordb = store_manager.create_vector_store(chunks)

    rag = BankingPolicyRAG(vectordb=vectordb, chunks=chunks, enable_analytics=True)
    return rag

# Load system
try:
    rag = load_rag_system()
    system_ready = True
except Exception as e:
    system_ready = False
    system_error = str(e)


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

if "chat_sessions" not in st.session_state:
    # Dictionary mapping session_id -> { "title": str, "messages": list, "rag_history": list }
    default_id = str(uuid.uuid4())
    st.session_state.chat_sessions = {
        default_id: {
            "title": "New Chat",
            "messages": [],
            "rag_history": []
        }
    }
    st.session_state.current_session_id = default_id

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = list(st.session_state.chat_sessions.keys())[0]

if "example_style" not in st.session_state:
    st.session_state.example_style = "Short example"

# Helper function to create a new session
def create_new_session():
    new_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_id] = {
        "title": "New Chat",
        "messages": [],
        "rag_history": []
    }
    st.session_state.current_session_id = new_id
    if system_ready:
        rag.clear_history()

# Select the active session object
active_session = st.session_state.chat_sessions[st.session_state.current_session_id]

# Sync RAG engine's internal history with the active session
if system_ready:
    rag.chat_history = active_session["rag_history"].copy()


# ============================================================================
# SIDEBAR (Multi-Session Tabs)
# ============================================================================

with st.sidebar:
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        create_new_session()
        st.rerun()

    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("<span style='color:#888; font-size:0.8rem; font-weight:600; margin-bottom:0.5rem; display:block;'>Recent Chats</span>", unsafe_allow_html=True)

    # Render History Tabs
    # Sort sessions by newest first based on their position in the dict (Python 3.7+ preserves insertion order)
    # Reversing gives us the newest first.
    for s_id in reversed(list(st.session_state.chat_sessions.keys())):
        session_data = st.session_state.chat_sessions[s_id]
        title = session_data["title"]
        
        # Highlight active session
        is_active = (s_id == st.session_state.current_session_id)
        icon = "💬" if is_active else "💭"
        btn_label = f"{icon} {title[:25]}{'...' if len(title) > 25 else ''}"
        
        if st.button(btn_label, key=f"session_{s_id}", use_container_width=True):
            st.session_state.current_session_id = s_id
            st.rerun()


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>Ask FactorAvenue</h1>
    <p>Ask questions about factoring, forfaiting, RBI guidelines, TReDS, and related banking regulation.</p>
</div>
""", unsafe_allow_html=True)

# Settings Row (Answer Style, Clear)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.session_state.example_style = st.selectbox(
        "Answer Details:",
        options=["None", "Short example", "Detailed example"],
        index=["None", "Short example", "Detailed example"].index(st.session_state.example_style),
        label_visibility="collapsed"
    )

with col3:
    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        active_session["messages"] = []
        active_session["rag_history"] = []
        if system_ready:
            rag.clear_history()
        st.rerun()

st.markdown('<div class="clean-divider"></div>', unsafe_allow_html=True)

# Suggested Questions Section (Only show if current chat is empty)
if not active_session["messages"] and system_ready:
    st.markdown("<h4 style='text-align: center; color:#555;'>💡 Suggestions</h4>", unsafe_allow_html=True)

    suggested = rag.get_suggested_questions()
    categories = list(suggested.items())
    
    # Render centrally aligned items
    for i in range(0, len(categories), 2):
        cols = st.columns([1, 3, 3, 1]) # Center 2 columns
        for j, col in enumerate([cols[1], cols[2]]):
            if i + j < len(categories):
                category_name, questions = categories[i + j]
                with col:
                    st.markdown(f"<div class='suggestion-category' style='text-align:center;'>{category_name}</div>", unsafe_allow_html=True)
                    for q in questions[:2]:
                        if st.button(f"{q}", key=f"suggest_{i+j}_{q[:30]}", use_container_width=True):
                            st.session_state.pending_question = q
                            st.rerun()


# Render Chat History
for idx, msg in enumerate(active_session["messages"]):
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🏛️"):
        st.markdown(msg["content"])

        # Show metadata and sources for assistant messages
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]

            intent = meta.get('intent', 'GENERAL').lower()
            confidence = meta.get('confidence', 'Medium').lower()

            st.markdown(f"""
            <div class="response-meta" style="margin-top:0.8rem; border-top:1px solid #eaeaea; padding-top:0.5rem;">
                <span class="intent-badge">Intent: {meta.get('intent', 'GENERAL')}</span>
                <span class="intent-badge">Confidence: {meta.get('confidence', 'Medium')}</span>
                <span style="color: #999; font-size: 0.7rem; margin-left: 0.5rem;">
                    ⚡ {meta.get('response_time_ms', 0)}ms
                </span>
            </div>
            """, unsafe_allow_html=True)

            sources = meta.get('sources', [])
            if sources:
                with st.expander(f"📄 View Sources ({len(sources)} documents)", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-type">{src.get('document_type', 'Document')}</div>
                            <div class="source-section">{src.get('section', 'General')}</div>
                            <div class="source-authority">📌 {src.get('issuing_authority', 'Unknown')}</div>
                            <div class="source-excerpt">{src.get('content', '')[:250]}...</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Feedback buttons
            query_id = meta.get('query_id')
            if query_id:
                feedback_key = f"feedback_{st.session_state.current_session_id}_{idx}"
                f_cols = st.columns([1, 1, 10])
                with f_cols[0]:
                    if st.button("👍", key=f"{feedback_key}_up", help="Helpful answer"):
                        rag.submit_feedback(query_id, 1)
                        st.toast("Thanks for the feedback!", icon="✅")
                with f_cols[1]:
                    if st.button("👎", key=f"{feedback_key}_down", help="Not helpful"):
                        rag.submit_feedback(query_id, -1)
                        st.toast("Feedback recorded.", icon="📝")


# ============================================================================
# CHAT INPUT PROCESSING
# ============================================================================

user_input = st.chat_input("Ask about banking policies, RBI guidelines, KYC norms...")
question = st.session_state.get("pending_question") or user_input

if question:
    # Clear pending question
    if "pending_question" in st.session_state:
        del st.session_state["pending_question"]
        
    if not system_ready:
        st.error(f"System is not ready: {system_error}")
    else:
        # Title updater for new chats
        if len(active_session["messages"]) == 0:
            active_session["title"] = question[:25] + "..." if len(question) > 25 else question

        # Display user message
        active_session["messages"].append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant", avatar="🏛️"):
            with st.spinner("Searching policy documents..."):
                result = rag.query(question, k=5, example_style=st.session_state.example_style)

            st.markdown(result["answer"])

            intent = result['intent'].lower()
            confidence = result['confidence'].lower()

            st.markdown(f"""
            <div class="response-meta" style="margin-top:0.8rem; border-top:1px solid #eaeaea; padding-top:0.5rem;">
                <span class="intent-badge">Intent: {result['intent']}</span>
                <span class="intent-badge">Confidence: {result['confidence']}</span>
                <span style="color: #999; font-size: 0.7rem; margin-left: 0.5rem;">
                    ⚡ {result['response_time_ms']}ms
                </span>
            </div>
            """, unsafe_allow_html=True)

            if result["sources"]:
                with st.expander(f"📄 View Sources ({result['num_retrieved']} documents)", expanded=False):
                    for i, src in enumerate(result["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-type">{src.get('document_type', 'Document')}</div>
                            <div class="source-section">{src.get('section', 'General')}</div>
                            <div class="source-authority">📌 {src.get('issuing_authority', 'Unknown')}</div>
                            <div class="source-excerpt">{src.get('content', '')[:250]}...</div>
                        </div>
                        """, unsafe_allow_html=True)

            query_id = result.get('query_id')
            if query_id:
                msg_idx = len(active_session["messages"])
                f_cols = st.columns([1, 1, 10])
                with f_cols[0]:
                    if st.button("👍", key=f"fb_{st.session_state.current_session_id}_{msg_idx}_up"):
                        rag.submit_feedback(query_id, 1)
                        st.toast("Thanks for the feedback!", icon="✅")
                with f_cols[1]:
                    if st.button("👎", key=f"fb_{st.session_state.current_session_id}_{msg_idx}_down"):
                        rag.submit_feedback(query_id, -1)
                        st.toast("Feedback recorded.", icon="📝")

        # Save to active session
        active_session["messages"].append({
            "role": "assistant",
            "content": result["answer"],
            "metadata": {
                "sources": result["sources"],
                "intent": result["intent"],
                "confidence": result["confidence"],
                "response_time_ms": result["response_time_ms"],
                "retrieval_method": result["retrieval_method"],
                "query_id": result.get("query_id"),
            }
        })
        
        # Save RAG memory back to session
        active_session["rag_history"] = rag.chat_history.copy()
        
        st.rerun()
