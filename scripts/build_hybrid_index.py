"""
Build Hybrid Index (BM25 + Dense) for Armenian Labor Law corpus.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import json
from src.retrieval.hybrid_retriever import HybridRetriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Build and save hybrid index."""
    
    # Paths
    chunks_path = project_root / "data" / "chunks" / "labor_law_chunks.json"
    index_path = project_root / "data" / "indices" / "hybrid"
    
    logger.info("="*60)
    logger.info("Building Hybrid Index (BM25 + Dense)")
    logger.info("="*60)
    
    # Load chunks
    logger.info(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Initialize hybrid retriever
    logger.info("\nInitializing Hybrid Retriever...")
    logger.info("  - BM25 weight: 0.5 (keyword matching)")
    logger.info("  - Dense weight: 0.5 (semantic similarity)")
    logger.info("  - Fusion method: Reciprocal Rank Fusion (RRF)")
    
    retriever = HybridRetriever(
        chunks=chunks,
        bm25_weight=0.5,
        dense_weight=0.5,
        rrf_k=60  # Standard RRF constant
    )
    
    # Build index
    logger.info("\nBuilding index (this will take a few minutes)...")
    logger.info("  Step 1/2: Building BM25 index (fast)")
    logger.info("  Step 2/2: Building Dense index (downloading embeddings if needed)")
    
    retriever.build_index()
    
    # Save index
    logger.info(f"\nSaving index to {index_path}...")
    retriever.save_index(str(index_path))
    
    logger.info("\n" + "="*60)
    logger.info("✅ Hybrid index built and saved successfully!")
    logger.info("="*60)
    logger.info(f"\nIndex location: {index_path}")
    logger.info(f"  - BM25 index: {index_path}/bm25.pkl")
    logger.info(f"  - Dense index: {index_path}/dense/")
    logger.info("\nNext steps:")
    logger.info("  1. Test retrieval: python scripts/test_hybrid_retrieval.py")
    logger.info("  2. Run RAG with hybrid: python scripts/test_rag.py --method hybrid")
    logger.info("  3. Use in Streamlit: Select 'Hybrid' from the dropdown")


if __name__ == "__main__":
    main()
