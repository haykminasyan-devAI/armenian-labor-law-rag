"""
Dense Retriever - Semantic embedding-based retrieval using Armenian embeddings.
"""

from typing import List, Dict, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path

from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """Dense retrieval using Armenian-specific embeddings (local model)."""
    
    def __init__(self, 
                 chunks: List[Dict], 
                 model_name: str = 'Metric-AI/armenian-text-embeddings-1'):
        """
        Initialize dense retriever with Armenian embeddings.
        
        Args:
            chunks: List of chunk dictionaries
            model_name: HuggingFace model name for Armenian embeddings
        """
        super().__init__(chunks)
        self.model_name = model_name
        self.model = None
        self.index = None
        self.embeddings = None
        self.embedding_dim = None
    
    def build_index(self):
        """Build dense embedding index using local Armenian embeddings."""
        logger.info(f"Building dense index with Armenian embeddings: {self.model_name}")
        logger.info("Loading embedding model locally (first run will download ~500MB)...")
        
        # Load embedding model locally
        self.model = SentenceTransformer(self.model_name)
        logger.info("✅ Embedding model loaded")
        
        # Extract texts from chunks
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Generate embeddings for all chunks (batched for efficiency)
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32  # Process in batches
        )
        
        self.embeddings = self.embeddings.astype('float32')
        self.embedding_dim = self.embeddings.shape[1]
        
        logger.info(f"Generated {len(self.embeddings)} embeddings of dimension {self.embedding_dim}")
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        self.is_indexed = True
        logger.info(f"✅ Dense index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search using dense embeddings.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of chunks with similarity scores
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index() first.")
        
        if self.model is None:
            # Load model if not already loaded (e.g., after loading from disk)
            self.model = SentenceTransformer(self.model_name)
        
        # Get query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        query_embedding = query_embedding.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Safety check
                result = self.chunks[idx].copy()
                result['score'] = float(score)
                result['retrieval_method'] = 'dense_armenian'
                results.append(result)
        
        logger.debug(f"Dense search for '{query}': found {len(results)} results")
        return results
    
    def save_index(self, path: str):
        """Save dense index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        # Save embeddings and metadata
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim
            }, f)
        
        logger.info(f"Dense index saved to {path}")
    
    def load_index(self, path: str):
        """Load dense index from disk."""
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss.index"))
        
        # Load metadata
        with open(path / "metadata.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.model_name = data['model_name']
        self.embedding_dim = data['embedding_dim']
        
        # Model will be loaded on first search
        self.model = None
        
        self.is_indexed = True
        logger.info(f"Dense index loaded from {path}")
