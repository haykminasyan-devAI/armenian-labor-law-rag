#!/usr/bin/env python3
"""
Build indices for hybrid chunks (with token-limited subsections).
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
import json

def main():
    print("🔨 Building Indices for Hybrid Chunks")
    print("=" * 70)
    
    # Load hybrid chunks
    chunks_file = project_root / "data" / "chunks" / "labor_law_chunks_hybrid.json"
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"✓ Loaded {len(chunks)} hybrid chunks")
    
    # Build BM25 index
    print("\n1️⃣  Building BM25 index...")
    bm25_retriever = BM25Retriever(chunks)
    bm25_retriever.build_index()
    bm25_index_path = project_root / "data" / "indices" / "bm25_hybrid"
    bm25_index_path.mkdir(parents=True, exist_ok=True)
    bm25_retriever.save_index(str(bm25_index_path / "bm25_index.pkl"))
    print(f"   ✓ Saved to: {bm25_index_path}")
    
    # Build Dense index
    print("\n2️⃣  Building Dense index (this will take a few minutes)...")
    dense_retriever = DenseRetriever(chunks)
    dense_retriever.build_index()
    dense_index_path = project_root / "data" / "indices" / "dense_hybrid"
    dense_index_path.mkdir(parents=True, exist_ok=True)
    dense_retriever.save_index(str(dense_index_path))
    print(f"   ✓ Saved to: {dense_index_path}")
    
    # Build Hybrid index
    print("\n3️⃣  Building Hybrid index...")
    hybrid_retriever = HybridRetriever(chunks, bm25_weight=0.5, dense_weight=0.5)
    hybrid_retriever.build_index()
    hybrid_index_path = project_root / "data" / "indices" / "hybrid_v2"
    hybrid_index_path.mkdir(parents=True, exist_ok=True)
    hybrid_retriever.save_index(str(hybrid_index_path))
    print(f"   ✓ Saved to: {hybrid_index_path}")
    
    print("\n✅ All indices built successfully!")
    print("\n🎯 Next Steps:")
    print("  Update Streamlit app to use hybrid chunks:")
    print(f"    - chunks file: {chunks_file}")
    print(f"    - BM25 index: {bm25_index_path}")
    print(f"    - Dense index: {dense_index_path}")
    print(f"    - Hybrid index: {hybrid_index_path}")

if __name__ == "__main__":
    main()
