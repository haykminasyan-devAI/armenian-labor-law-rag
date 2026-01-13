#!/usr/bin/env python3
"""
Streamlit Web Interface for Armenian Labor Law Q&A
"""

import sys
import json
from pathlib import Path
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.rag_pipeline import RAGPipeline
from src.generation.generator import LLMGenerator


# Page config
st.set_page_config(
    page_title="Armenian Labor Law Q&A",
    page_icon="🇦🇲",
    layout="wide"
)

# Title
st.title("🇦🇲 Հայաստանի Աշխատանքային օրենսգիրք")
st.subheader("Հարցեր և Պատասխաններ (RAG System)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Կարգավորումներ")
    
    retrieval_method = st.selectbox(
        "Որոնման մեթոդ:",
        ["BM25 (Keywords)", "Dense (Armenian Embeddings)", "Hybrid (Both)"],
        index=0
    )
    
    top_k = st.slider("Հոդվածների քանակ:", min_value=1, max_value=10, value=3)
    
    st.markdown("---")
    st.markdown("### 📊 Համակարգի տեղեկություններ")
    st.markdown("- **Հոդվածներ:** 286")
    st.markdown("- **Մոդել:** NVIDIA Llama 3.1-70B")
    st.markdown("- **Լեզու:** Հայերեն")


@st.cache_resource
def load_rag_pipeline(method):
    """Load RAG pipeline (cached)."""
    # Load chunks
    chunks_file = project_root / "data" / "chunks" / "labor_law_chunks.json"
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Initialize retriever
    if "BM25" in method:
        retriever = BM25Retriever(chunks)
        index_path = project_root / "indices" / "bm25" / "bm25_index.pkl"
        retriever.load_index(str(index_path))
    elif "Dense" in method:
        retriever = DenseRetriever(chunks)
        index_path = project_root / "indices" / "dense"
        retriever.load_index(str(index_path))
    else:  # Hybrid
        # TODO: Implement hybrid
        retriever = BM25Retriever(chunks)
        index_path = project_root / "indices" / "bm25" / "bm25_index.pkl"
        retriever.load_index(str(index_path))
    
    # Initialize generator
    api_key = "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"
    generator = LLMGenerator(
        model_name="meta/llama-3.1-70b-instruct",
        provider="nvidia",
        api_key=api_key,
        max_tokens=1000,
        temperature=0.1
    )
    
    # Create RAG pipeline
    return RAGPipeline(retriever=retriever, generator=generator)


# Load pipeline
rag_pipeline = load_rag_pipeline(retrieval_method)

# Main chat interface
st.markdown("### 💬 Հարցրեք ինձ աշխատանքային իրավունքի մասին:")

# Example questions
with st.expander("📝 Օրինակ հարցեր"):
    st.markdown("""
    - Որո՞նք են նվազագույն աշխատավարձի կանոնները։
    - Քանի՞ արձակուրդային օր կա։
    - Ինչպե՞ս է սահմանվում գործուղման օրապահիկը։
    - Ի՞նչ իրավունքներ ունի աշխատողը երբ իրեն կրճատում են։
    - Ինչ է ասում Հոդված 1-ին հոդվածը։
    """)

# Question input
question = st.text_input(
    "Ձեր հարցը:",
    placeholder="Գրեք ձեր հարցը հայերեն...",
    key="question_input"
)

# Search button
if st.button("🔍 Փնտրել", type="primary") or question:
    if question:
        with st.spinner('🔍 Փնտրում եմ համապատասխան հոդվածներ...'):
            try:
                # Get answer
                result = rag_pipeline.answer_question(
                    question,
                    top_k=top_k,
                    return_context=True
                )
                
                # Display answer
                st.markdown("---")
                st.markdown("### 💡 ՊԱՏԱՍԽԱՆ:")
                st.success(result['answer'])
                
                # Display retrieved articles
                st.markdown("---")
                st.markdown("### 📊 Գտնված հոդվածներ:")
                
                cols = st.columns(3)
                for i, (article, score) in enumerate(zip(result['article_numbers'], result['scores'])):
                    with cols[i % 3]:
                        st.metric(
                            label=f"Հոդված {article}",
                            value=f"{score:.2f}",
                            delta=f"#{i+1}"
                        )
                
                # Show context in expander
                with st.expander("📄 Դիտել գտնված հոդվածների տեքստը"):
                    for i, chunk in enumerate(result['retrieved_chunks'], 1):
                        st.markdown(f"**Հոդված {chunk.get('article_number')}:**")
                        st.text(chunk['text'][:500] + "...")
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Սխալ: {e}")
    else:
        st.warning("⚠️ Խնդրում ենք մուտքագրել հարց")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🇦🇲 Armenian Labor Law RAG System | Powered by NVIDIA Llama 3.1-70B</p>
    <p>Using {method} | 286 Articles Indexed</p>
</div>
""".format(method=retrieval_method), unsafe_allow_html=True)
