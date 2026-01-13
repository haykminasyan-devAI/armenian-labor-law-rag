#!/usr/bin/env python3
"""
Build dense retrieval index using Armenian embeddings.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Build Dense index with Armenian embeddings."""
    
    # Paths
    chunks_file = project_root / "data" / "chunks" / "labor_law_chunks.json"
    index_dir = project_root / "indices" / "dense"
    
    logger.info("=" * 80)
    logger.info("BUILDING DENSE INDEX (Armenian Embeddings - LOCAL)")
    logger.info("=" * 80)
    
    # Step 1: Load chunks
    logger.info(f"\n📖 Loading chunks from {chunks_file}...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"   Loaded {len(chunks)} chunks")
    
    # Step 2: Initialize retriever
    logger.info(f"\n🔧 Initializing Dense retriever with Armenian embeddings...")
    retriever = DenseRetriever(
        chunks,
        model_name='Metric-AI/armenian-text-embeddings-1'
    )
    
    # Step 3: Build index
    logger.info(f"\n📇 Building Dense index...")
    logger.info("   First run will download model (~500MB), then embed 286 chunks...")
    logger.info("   This will take ~2-3 minutes...")
    retriever.build_index()
    
    # Step 4: Save index
    logger.info(f"\n💾 Saving index to {index_dir}...")
    retriever.save_index(str(index_dir))
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ DENSE INDEX BUILT SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"\n📊 Index Statistics:")
    logger.info(f"   - Total documents: {len(chunks)}")
    logger.info(f"   - Embedding dimension: {retriever.embedding_dim}")
    logger.info(f"   - Index location: {index_dir}")
    logger.info(f"\n🔍 Next: Test with scripts/test_dense_retrieval.py")


if __name__ == "__main__":
    main()
