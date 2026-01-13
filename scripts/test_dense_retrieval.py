#!/usr/bin/env python3
"""
Test the Dense retriever with Armenian embeddings.
"""

import sys
import json
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.dense_retriever import DenseRetriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_dense_retrieval():
    """Test Dense retrieval with Armenian embeddings."""
    
    # Paths
    chunks_file = project_root / "data" / "chunks" / "labor_law_chunks.json"
    index_dir = project_root / "indices" / "dense"
    
    logger.info("=" * 80)
    logger.info("TESTING DENSE RETRIEVAL (Armenian Embeddings)")
    logger.info("=" * 80)
    
    # Load chunks
    logger.info(f"\n📖 Loading chunks...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Initialize retriever
    retriever = DenseRetriever(chunks, model_name='Metric-AI/armenian-text-embeddings-1')
    
    # Load index
    logger.info(f"\n📇 Loading Dense index...")
    retriever.load_index(str(index_dir))
    
    # Test queries (same as BM25 for comparison)
    test_queries = [
        "Ինչ է կարգավորում Աշխատանքային օրենսգրքի 1-ին հոդվածը։",
        "Քանի՞ արձակուրդային օր կա։",
        "Ինչպե՞ս է սահմանվում գործուղման օրապահիկը։",
        "Ի՞նչ իրավունքներ ունի աշխատողը երբ իրեն կրճատում են։"
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
            logger.info(f"   Similarity Score: {result['score']:.4f}")
            logger.info(f"   Article: {result.get('article_number', 'N/A')}")
            logger.info(f"   Chunk ID: {result['chunk_id']}")
            logger.info(f"   Text preview: {result['text'][:150]}...")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TESTING COMPLETE!")
    logger.info("=" * 80)
    logger.info("\n💡 Compare these results with BM25 results!")
    logger.info("   Dense retrieval finds by MEANING, BM25 finds by KEYWORDS")


if __name__ == "__main__":
    test_dense_retrieval()
