#!/usr/bin/env python3
"""
Test the BM25 retriever with sample queries.
"""

import sys
import json
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.bm25_retriever import BM25Retriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_retrieval():
    """Test BM25 retrieval with sample queries."""
    
    # Paths
    chunks_file = project_root / "data" / "chunks" / "labor_law_chunks.json"
    index_file = project_root / "indices" / "bm25" / "bm25_index.pkl"
    
    logger.info("=" * 80)
    logger.info("TESTING BM25 RETRIEVAL")
    logger.info("=" * 80)
    
    # Load chunks
    logger.info(f"\n📖 Loading chunks...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Initialize retriever
    retriever = BM25Retriever(chunks)
    
    # Load index
    logger.info(f"\n📇 Loading BM25 index...")
    retriever.load_index(str(index_file))
    
    # Test queries (Armenian labor law queries)
    test_queries = [
        "աշխատանքային օրենսգիրք",      # Labor law
        "աշխատանքային պայմանագիր",     # Employment contract
        "նվազագույն աշխատավարձ",      # Minimum wage
        "արձակուրդ",                  # Vacation
        "աշխատանքային ժամեր"          # Working hours
    ]
    
    # Test each query
    for i, query in enumerate(test_queries, 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"Query {i}: '{query}'")
        logger.info("=" * 80)
        
        # Search
        results = retriever.search(query, top_k=3)
        
        # Display results
        for j, result in enumerate(results, 1):
            logger.info(f"\n📄 Result {j}:")
            logger.info(f"   Score: {result['score']:.4f}")
            logger.info(f"   Article: {result.get('article_number', 'N/A')}")
            logger.info(f"   Chunk ID: {result['chunk_id']}")
            logger.info(f"   Text preview: {result['text'][:150]}...")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TESTING COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_retrieval()