#!/usr/bin/env python3
"""
Build retrieval indices from chunks.
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


def main():
    """Build BM25 index."""
    
    # Paths
    chunks_file = project_root / "data" / "chunks" / "labor_law_chunks.json"
    index_dir = project_root / "indices" / "bm25"
    index_file = index_dir / "bm25_index.pkl"
    
    logger.info("=" * 80)
    logger.info("BUILDING BM25 INDEX")
    logger.info("=" * 80)
    
    # Step 1: Load chunks
    logger.info(f"\n📖 Loading chunks from {chunks_file}...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"   Loaded {len(chunks)} chunks")
    
    # Step 2: Initialize retriever
    logger.info(f"\n🔧 Initializing BM25 retriever...")
    retriever = BM25Retriever(chunks)
    
    # Step 3: Build index
    logger.info(f"\n📇 Building BM25 index...")
    retriever.build_index()
    
    # Step 4: Save index
    logger.info(f"\n💾 Saving index to {index_file}...")
    retriever.save_index(str(index_file))
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ BM25 INDEX BUILT SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"\n📊 Index Statistics:")
    logger.info(f"   - Total documents: {len(chunks)}")
    logger.info(f"   - Index location: {index_file}")
    logger.info(f"\n🔍 Next: Test the retriever with scripts/test_retrieval.py")


if __name__ == "__main__":
    main()