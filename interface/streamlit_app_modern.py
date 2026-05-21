#!/usr/bin/env python3
"""
Modern ChatGPT-style Web Interface for Armenian Labor Law Q&A
"""

import sys
import json
import os
import logging
from pathlib import Path
import streamlit as st
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_pipeline import RAGPipeline
from src.generation.generator import LLMGenerator

# Page config
st.set_page_config(
    page_title="🇦🇲 Armenian Labor Law AI",
    page_icon="🇦🇲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f7f7f8;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 18px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background: white;
        color: #374151;
        padding: 15px 20px;
        border-radius: 18px;
        margin: 10px 0;
        max-width: 80%;
        border-left: 4px solid #10b981;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .article-badge {
        display: inline-block;
        background: #ede9fe;
        color: #6d28d9;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        margin: 2px;
        font-weight: 600;
    }
    
    /* Selected option styling */
    .selected-option {
        background: #10b981 !important;
        color: white !important;
        border: 2px solid #059669 !important;
    }
    
    /* Option buttons */
    div[data-testid="stButton"] button {
        text-align: left !important;
        padding: 10px 15px !important;
    }
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 5px 0;
    }
    
    /* Header styling */
    .big-title {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2em;
    }
    
    .subtitle {
        color: #6b7280;
        font-size: 1.1em;
        margin-bottom: 1.5em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat management
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversations' not in st.session_state:
    st.session_state.conversations = []  # List of saved conversations

if 'current_conversation_id' not in st.session_state:
    st.session_state.current_conversation_id = 0

# Initialize settings in session state
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "Llama 3.3-70B (Fastest)"
if 'retrieval_method' not in st.session_state:
    st.session_state.retrieval_method = "Hybrid (Best)"
if 'top_k' not in st.session_state:
    st.session_state.top_k = 3
if 'show_sources' not in st.session_state:
    st.session_state.show_sources = True
if 'show_scores' not in st.session_state:
    st.session_state.show_scores = False
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None

# Sidebar - Chat History + New Conversation
with st.sidebar:
    st.markdown("### 💬 Conversations")
    
    # New Conversation button - saves current before creating new
    if st.button("➕ New Conversation", use_container_width=True, type="primary"):
        # Save current conversation if it has messages
        if st.session_state.messages:
            # Get first question as title
            title = "New Chat"
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    title = msg["content"][:40].replace('\n', ' ')
                    break
            
            # Save conversation
            st.session_state.conversations.append({
                'id': st.session_state.current_conversation_id,
                'title': title,
                'messages': st.session_state.messages.copy(),
                'timestamp': datetime.now().strftime("%H:%M")
            })
            
            # Start new conversation
            st.session_state.current_conversation_id += 1
            st.session_state.messages = []
        
        st.rerun()
    
    st.markdown("---")
    
    # Current Chat
    if st.session_state.messages:
        num_questions = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f"**💬 Current** ({num_questions} Q&A)")
        st.markdown("---")
    
    # Previous Conversations
    if st.session_state.conversations:
        st.markdown("**📚 History:**")
        
        for conv in reversed(st.session_state.conversations[-10:]):  # Show last 10
            conv_preview = f"{conv['title'][:35]}..."
            if st.button(
                f"🕐 {conv['timestamp']} - {conv_preview}",
                key=f"conv_{conv['id']}",
                use_container_width=True
            ):
                # Load this conversation
                st.session_state.messages = conv['messages'].copy()
                st.rerun()
    else:
        st.markdown("*No saved chats yet*")

# Use settings from session state
model_choice = st.session_state.model_choice
retrieval_method = st.session_state.retrieval_method
top_k = st.session_state.top_k
show_sources = st.session_state.show_sources
show_scores = st.session_state.show_scores

# Extract model config
if "405B" in model_choice:
    generation_model = "405B"
elif "DeepSeek" in model_choice:
    generation_model = "DeepSeek"
elif "Qwen" in model_choice:
    generation_model = "Qwen"
elif "3.3" in model_choice or "Fastest" in model_choice:
    generation_model = "3.3-70B"
else:
    generation_model = "70B"

def _get_nvidia_api_key() -> str:
    """Prefer NVIDIA_API_KEY from environment (.env or export)."""
    key = os.getenv("NVIDIA_API_KEY")
    if key:
        return key
    raise ValueError(
        "NVIDIA_API_KEY is not set. Run: export NVIDIA_API_KEY='nvapi-...' "
        "or add it to a .env file in the project root."
    )


# Load RAG pipeline (cached)
@st.cache_resource
def load_rag_pipeline(retrieval_method, generation_model):
    """Load RAG pipeline (cached)."""
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    chunks_file = project_root / "data" / "chunks" / "labor_law_chunks_hybrid.json"
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"Missing chunks file: {chunks_file}\n"
            "Run: python scripts/extract_pdf.py && python scripts/preprocess_data.py "
            "&& python scripts/create_hybrid_chunks.py"
        )
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    if "BM25" in retrieval_method:
        retriever = BM25Retriever(chunks)
        retriever.load_index(str(project_root / "data" / "indices" / "bm25_hybrid" / "bm25_index.pkl"))
    elif "Dense" in retrieval_method:
        retriever = DenseRetriever(chunks)
        retriever.load_index(str(project_root / "data" / "indices" / "dense_hybrid"))
        retriever.warmup()
    else:
        retriever = HybridRetriever(chunks, bm25_weight=0.5, dense_weight=0.5)
        retriever.load_index(str(project_root / "data" / "indices" / "hybrid_v2"))
        retriever.warmup()
    
    if generation_model == "405B":
        model_name = "meta/llama-3.1-405b-instruct"
        max_tokens = 2000
    elif generation_model == "DeepSeek":
        model_name = "deepseek-ai/deepseek-v3.1"
        max_tokens = 2000
    elif generation_model == "Qwen":
        model_name = "qwen/qwen3-next-80b-a3b-instruct"
        max_tokens = 6000
    elif "3.3" in generation_model or "Fastest" in generation_model:
        model_name = "meta/llama-3.3-70b-instruct"
        max_tokens = 2000
    else:
        model_name = "meta/llama-3.1-70b-instruct"
        max_tokens = 2000
    
    generator = LLMGenerator(
        model_name=model_name,
        provider="nvidia",
        api_key=_get_nvidia_api_key(),
        max_tokens=max_tokens,
        temperature=0.1
    )
    
    return RAGPipeline(retriever=retriever, generator=generator)


try:
    with st.spinner("Բեռնում եմ գիտելիքի բազան և embedding մոդելը (առաջին անգամ 1–2 րոպե)..."):
        rag_pipeline = load_rag_pipeline(retrieval_method, generation_model)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Չհաջողվեց բեռնել RAG pipeline: {e}")
    st.exception(e)
    st.stop()

# Main header
st.markdown('<h1 class="big-title">🇦🇲 Հայաստանի Աշխատանքային Օրենսգիրք</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI Assistant powered by Advanced Retrieval & Generation</p>', unsafe_allow_html=True)

# Display chat history
chat_container = st.container()

with chat_container:
    for msg_idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>👤 Դուք:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            answer_html = f"""
            <div class="assistant-message">
                <strong>🤖 Օգնական:</strong><br>
                {message["content"]}
            </div>
            """
            
            st.markdown(answer_html, unsafe_allow_html=True)
            
            # Show clickable article sources
            sources = message.get("sources") or []
            if show_sources and sources:
                st.markdown("**📚 Աղբյուրներ:**")
                
                num_cols = min(len(sources), 3)
                cols = st.columns(num_cols)
                
                for idx, art_num in enumerate(sources[:5]):
                    with cols[idx % num_cols]:
                        # Clickable button for each article
                        if st.button(
                            f"📄 Հոդված {art_num}",
                            key=f"art_{msg_idx}_{idx}_{art_num}",
                            use_container_width=True
                        ):
                            # Show article text in expander
                            if "retrieved_chunks" in message:
                                for chunk in message["retrieved_chunks"]:
                                    if chunk.get('article_number') == art_num:
                                        with st.expander(f"📖 Հոդված {art_num} - Ամբողջ տեքստ", expanded=True):
                                            st.markdown(chunk['text'])
                                        break
            
            scores = message.get("scores") or []
            msg_sources = message.get("sources") or []
            if show_scores and scores and msg_sources:
                st.markdown("**🎯 Relevance Scores:**")
                cols = st.columns(len(scores))
                for i, (art, score) in enumerate(zip(msg_sources, scores)):
                    with cols[i]:
                        st.metric(f"Հոդված {art}", f"{score:.2f}", delta=f"#{i+1}")

# Question input area (ChatGPT-style)
st.markdown("---")

# Settings popup (ChatGPT-style dropdown menu)
with st.expander("➕ Models & Settings", expanded=False):
    st.markdown("**🤖 Select Model:**")
    
    model_options = [
        ("Llama 3.3-70B", "Llama 3.3-70B (Fastest)"),
        ("Llama 3.1-70B", "Llama 3.1-70B (Balanced)"),
        ("Llama 3.1-405B", "Llama 3.1-405B (Best Quality)"),
        ("DeepSeek V3.1", "DeepSeek V3.1 (Best Reasoning)"),
        ("Qwen 3 Next 80B", "Qwen 3 Next 80B (Advanced)")
    ]
    
    current_model_short = st.session_state.model_choice.split(" (")[0]
    
    for i, (model_short, model_full) in enumerate(model_options):
        is_selected = current_model_short == model_short
        
        # Clear visual indicator with emoji
        if is_selected:
            button_label = f"✅ {model_short} (ACTIVE)"
            button_type = "primary"
        else:
            button_label = f"⚪ {model_short}"
            button_type = "secondary"
        
        if st.button(button_label, key=f"model_{i}", use_container_width=True, type=button_type):
            st.session_state.model_choice = model_full
            # Clear cache to reload pipeline with new model
            load_rag_pipeline.clear()
            st.rerun()
    
    st.markdown("---")
    st.markdown("**🔍 Select Retrieval:**")
    
    retrieval_options = [
        "BM25 (Keywords)",
        "Dense (Semantic)",
        "Hybrid (Best)"
    ]
    
    for i, method in enumerate(retrieval_options):
        is_selected = st.session_state.retrieval_method == method
        
        # Clear visual indicator
        if is_selected:
            button_label = f"✅ {method} (ACTIVE)"
            button_type = "primary"
        else:
            button_label = f"⚪ {method}"
            button_type = "secondary"
        
        if st.button(button_label, key=f"retr_{i}", use_container_width=True, type=button_type):
            st.session_state.retrieval_method = method
            # Clear cache to reload pipeline with new retrieval method
            load_rag_pipeline.clear()
            st.rerun()
    
    st.markdown("---")
    st.markdown("**⚙️ Advanced:**")
    st.session_state.top_k = st.slider("Articles to retrieve", 1, 10, st.session_state.top_k)
    st.session_state.show_sources = st.checkbox("Show sources", st.session_state.show_sources)

# Example questions (set pending question, then rerun)
with st.expander("💡 Օրինակ հարցեր (սեղմեք օգտագործելու համար)"):
    example_questions = [
        "Քանի՞ արձակուրդային օր կա։",
        "Ինչպե՞ս է սահմանվում գործուղման օրապահիկը։",
        "Ի՞նչ իրավունքներ ունի աշխատողը երբ իրեն կրճատում են։",
        "Որո՞նք են նվազագույն աշխատավարձի կանոնները։",
        "Ի՞նչ է ասում Հոդված 145-րդը։"
    ]
    cols = st.columns(2)
    for i, example in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_question = example
                st.rerun()

# Chat form (Enter key submits)
with st.form("chat_form", clear_on_submit=True):
    question = st.text_input(
        "💬 Message",
        placeholder="Հարցրեք Աշխատանքային Օրենսգրքի մասին...",
        label_visibility="collapsed",
    )
    send_button = st.form_submit_button("📤 Send", use_container_width=True, type="primary")

# Resolve which question to process this run
question_to_ask = None
if st.session_state.pending_question:
    question_to_ask = st.session_state.pending_question.strip()
    st.session_state.pending_question = None
elif send_button and question:
    question_to_ask = question.strip()

if question_to_ask:
    st.session_state.messages.append({"role": "user", "content": question_to_ask})

    with st.spinner('🔍 Փնտրում եմ հոդվածները և գեներացնում պատասխանը (30–90 վրկ)...'):
        try:
            result = rag_pipeline.answer_question(
                question_to_ask,
                top_k=top_k,
                return_context=True
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": result['answer'],
                "sources": result.get('article_numbers', [])[:top_k],
                "scores": result.get('scores', [])[:top_k],
                "retrieved_chunks": result.get('retrieved_chunks', []),
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()

        except Exception as e:
            error_msg = str(e)
            logger.exception("Error in RAG pipeline")

            if "meta tensor" in error_msg or "to_empty" in error_msg:
                err_display = "DeepSeek model temporarily unavailable. Please select Llama or Qwen."
            elif "NVIDIA_API_KEY" in error_msg or "api_key" in error_msg.lower() or "401" in error_msg:
                err_display = f"API key error: {error_msg}. Set export NVIDIA_API_KEY='nvapi-...'"
            else:
                err_display = error_msg

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Սխալ: {err_display}",
                "sources": [],
                "scores": [],
                "retrieved_chunks": [],
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.error(f"❌ {err_display}")
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #9ca3af; font-size: 0.9em; padding: 20px;'>
    <p>🇦🇲 <strong>Armenian Labor Law AI Assistant</strong></p>
    <p>Powered by NVIDIA API | 286 Articles | Retrieval-Augmented Generation</p>
    <p style='font-size: 0.8em; margin-top: 10px;'>
        Built with ❤️ for Armenian Legal Research
    </p>
</div>
""", unsafe_allow_html=True)
