#!/usr/bin/env python3
"""
Test the complete RAG pipeline with Dense retriever (Armenian embeddings).
"""

import sys
import json
import os
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.dense_retriever import DenseRetriever
from src.generation.rag_pipeline import RAGPipeline
from src.generation.generator import LLMGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test RAG pipeline with Dense retriever."""
    
    logger.info("=" * 80)
    logger.info("TESTING RAG PIPELINE (Dense + Llama 3.1-70B)")
    logger.info("=" * 80)
    
    # Paths
    chunks_file = project_root / "data" / "chunks" / "labor_law_chunks.json"
    index_dir = project_root / "indices" / "dense"
    
    # Step 1: Load chunks and Dense index
    logger.info("\n📖 Loading chunks and Dense index...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    retriever = DenseRetriever(chunks, model_name='Metric-AI/armenian-text-embeddings-1')
    retriever.load_index(str(index_dir))
    logger.info("✅ Dense retriever loaded (Armenian embeddings)")
    
    # Step 2: Initialize RAG pipeline with NVIDIA Llama
    logger.info("\n🤖 Initializing RAG pipeline with Llama 3.1-70B (NVIDIA Build)...")
    
    api_key = "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"
    
    generator = LLMGenerator(
        model_name="meta/llama-3.1-70b-instruct",
        provider="nvidia",
        api_key=api_key,
        max_tokens=1000,
        temperature=0.1
    )
    rag_pipeline = RAGPipeline(retriever=retriever, generator=generator)
    logger.info("✅ RAG pipeline ready with Dense + Llama 3.1-70B")
    
    # Step 3: Test questions (same 5 as BM25 for comparison)
    test_questions = [
        "Ինչ է կարգավորում Աշխատանքային օրենսգրքի 1-ին հոդվածը։",
        "Քանի՞ արձակուրդային օր կա։",
        "ինչ է ասում Հոդված 159-րդը",
        "Ինչպե՞ս է սահմանվում գործուղման օրապահիկը։",
        "Ի՞նչ իրավունքներ ունի աշխատողը երբ իրեն կրճատում են։"
    ]
    
    # Step 4: Answer questions
    for i, question in enumerate(test_questions, 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"QUESTION {i}: {question}")
        logger.info("=" * 80)
        
        result = rag_pipeline.answer_question(question, top_k=3, return_context=True)
        
        logger.info(f"\n📊 Retrieved Articles: {result['article_numbers']}")
        logger.info(f"📊 Similarity Scores: {[f'{s:.4f}' for s in result['scores']]}")
        logger.info(f"📊 Retrieval Method: {result['retrieval_method']}")
        logger.info(f"\n💡 ANSWER:\n{result['answer']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ DENSE RAG PIPELINE TEST COMPLETE!")
    logger.info("=" * 80)
    logger.info("\n💡 Compare these results with BM25 RAG results!")
    logger.info("   Dense finds articles by SEMANTIC MEANING")
    logger.info("   BM25 finds articles by KEYWORD MATCHING")


if __name__ == "__main__":
    main()
