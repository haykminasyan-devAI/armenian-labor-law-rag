"""
Re-ranker for improving retrieval precision using cross-encoder models.

Two-stage retrieval:
1. First stage: Fast retrieval (BM25/Dense/Hybrid) → Get top-100 candidates
2. Second stage: Re-ranking → Use cross-encoder to re-score → Return top-K

Cross-encoders are more accurate but slower than bi-encoders (used in dense retrieval).
"""

from typing import List, Dict, Optional
import logging
from sentence_transformers import CrossEncoder
import numpy as np

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder based re-ranker for improving retrieval precision.
    
    Uses a pre-trained cross-encoder model to re-score retrieved candidates.
    Cross-encoders jointly encode query+document, making them more accurate
    than bi-encoders at the cost of speed.
    """
    
    def __init__(self, 
                 model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                 device: Optional[str] = None):
        """
        Initialize re-ranker.
        
        Args:
            model_name: HuggingFace cross-encoder model name
                Popular options:
                - 'cross-encoder/ms-marco-MiniLM-L-6-v2' (fast, English-focused)
                - 'cross-encoder/ms-marco-MiniLM-L-12-v2' (better quality)
                - 'BAAI/bge-reranker-base' (good multilingual support)
                - 'BAAI/bge-reranker-large' (best quality, slower)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.model_name = model_name
        
        logger.info(f"Loading re-ranker model: {model_name}")
        logger.info("First run will download ~100-400MB...")
        
        try:
            self.model = CrossEncoder(model_name, device=device)
            logger.info(f"✅ Re-ranker loaded on {self.model.device}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            logger.info("Falling back to ms-marco-MiniLM-L-6-v2...")
            self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
            self.model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            logger.info("✅ Re-ranker loaded (fallback model)")
    
    def rerank(self, 
               query: str, 
               candidates: List[Dict], 
               top_k: int = 5) -> List[Dict]:
        """
        Re-rank candidates using cross-encoder scoring.
        
        Args:
            query: Search query
            candidates: List of candidate chunks from first-stage retrieval
            top_k: Number of top results to return after re-ranking
            
        Returns:
            Top-K re-ranked results with updated scores
        """
        if not candidates:
            return []
        
        if len(candidates) <= top_k:
            # If we have fewer candidates than top_k, just re-score them all
            logger.debug(f"Candidates ({len(candidates)}) <= top_k ({top_k}), re-scoring all")
        
        logger.debug(f"Re-ranking {len(candidates)} candidates for top-{top_k}")
        
        # Prepare query-document pairs for cross-encoder
        pairs = []
        for candidate in candidates:
            text = candidate.get('text', '')
            # Truncate very long texts to avoid model limits
            if len(text) > 2000:
                text = text[:2000] + "..."
            pairs.append([query, text])
        
        # Score all pairs
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            # Fallback: return original candidates
            return candidates[:top_k]
        
        # Combine candidates with new scores
        reranked = []
        for candidate, score in zip(candidates, scores):
            result = candidate.copy()
            result['rerank_score'] = float(score)
            result['original_score'] = result.get('score', 0.0)
            result['score'] = float(score)  # Update main score
            result['retrieval_method'] = result.get('retrieval_method', 'unknown') + '_reranked'
            reranked.append(result)
        
        # Sort by re-rank score and return top-k
        reranked = sorted(reranked, key=lambda x: x['rerank_score'], reverse=True)
        top_results = reranked[:top_k]
        
        logger.debug(f"Re-ranking complete: returned top-{len(top_results)} results")
        
        return top_results


class RerankedRetriever:
    """
    Wrapper that combines any base retriever with re-ranking.
    
    Two-stage retrieval pipeline:
    1. Base retriever (BM25/Dense/Hybrid) → top-N candidates (e.g., N=20-100)
    2. Re-ranker (cross-encoder) → top-K final results (e.g., K=3-5)
    """
    
    def __init__(self, 
                 base_retriever,
                 reranker: Optional[Reranker] = None,
                 first_stage_k: int = 20):
        """
        Initialize re-ranked retriever.
        
        Args:
            base_retriever: Any retriever (BM25/Dense/Hybrid)
            reranker: Reranker instance (if None, creates default)
            first_stage_k: How many candidates to retrieve in first stage
                          (more = better recall but slower re-ranking)
        """
        self.base_retriever = base_retriever
        self.reranker = reranker if reranker else Reranker()
        self.first_stage_k = first_stage_k
        
        logger.info(f"RerankedRetriever initialized:")
        logger.info(f"  - Base retriever: {type(base_retriever).__name__}")
        logger.info(f"  - Re-ranker: {self.reranker.model_name}")
        logger.info(f"  - First stage K: {first_stage_k}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Two-stage search with re-ranking.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            
        Returns:
            Top-K re-ranked results
        """
        # Stage 1: Fast retrieval of candidates
        logger.debug(f"Stage 1: Retrieving {self.first_stage_k} candidates...")
        candidates = self.base_retriever.search(query, top_k=self.first_stage_k)
        
        logger.debug(f"Stage 1: Retrieved {len(candidates)} candidates")
        
        # Stage 2: Re-rank candidates
        logger.debug(f"Stage 2: Re-ranking to top-{top_k}...")
        final_results = self.reranker.rerank(query, candidates, top_k=top_k)
        
        logger.debug(f"Stage 2: Returned {len(final_results)} results")
        
        return final_results
    
    def build_index(self):
        """Build index for base retriever."""
        if hasattr(self.base_retriever, 'build_index'):
            self.base_retriever.build_index()
    
    def save_index(self, path: str):
        """Save base retriever index."""
        if hasattr(self.base_retriever, 'save_index'):
            self.base_retriever.save_index(path)
    
    def load_index(self, path: str):
        """Load base retriever index."""
        if hasattr(self.base_retriever, 'load_index'):
            self.base_retriever.load_index(path)
    
    @property
    def is_indexed(self):
        """Check if base retriever is indexed."""
        if hasattr(self.base_retriever, 'is_indexed'):
            return self.base_retriever.is_indexed
        return True
