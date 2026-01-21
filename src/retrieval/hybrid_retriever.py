"""
Hybrid Retriever - Combines BM25 (keyword) and Dense (semantic) retrieval.
Uses Reciprocal Rank Fusion (RRF) to merge results.
"""

from typing import List, Dict, Optional
import logging
from pathlib import Path

from .base_retriever import BaseRetriever
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval combining BM25 (keyword-based) and Dense (semantic) methods.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both retrievers,
    which has been shown to outperform weighted score fusion in many cases.
    """
    
    def __init__(self, 
                 chunks: List[Dict],
                 bm25_weight: float = 0.5,
                 dense_weight: float = 0.5,
                 rrf_k: int = 60):
        """
        Initialize hybrid retriever.
        
        Args:
            chunks: List of chunk dictionaries
            bm25_weight: Weight for BM25 scores (0-1), used if RRF disabled
            dense_weight: Weight for Dense scores (0-1), used if RRF disabled
            rrf_k: RRF constant (default 60, from original paper)
        """
        super().__init__(chunks)
        
        # Initialize sub-retrievers
        self.bm25_retriever = BM25Retriever(chunks)
        self.dense_retriever = DenseRetriever(chunks)
        
        # Fusion parameters
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        
        # Validate weights
        if not (0 <= bm25_weight <= 1 and 0 <= dense_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        
        logger.info(f"Hybrid retriever initialized (BM25: {bm25_weight}, Dense: {dense_weight}, RRF_k: {rrf_k})")
    
    def build_index(self):
        """Build indices for both BM25 and Dense retrievers."""
        logger.info("Building hybrid index (BM25 + Dense)...")
        
        # Build BM25 index
        logger.info("1/2 Building BM25 component...")
        self.bm25_retriever.build_index()
        
        # Build Dense index
        logger.info("2/2 Building Dense component...")
        self.dense_retriever.build_index()
        
        self.is_indexed = True
        logger.info("✅ Hybrid index built successfully")
    
    def _reciprocal_rank_fusion(self, 
                                 bm25_results: List[Dict], 
                                 dense_results: List[Dict],
                                 top_k: int) -> List[Dict]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score(d) = Σ 1 / (k + rank(d))
        where k is a constant (typically 60) and rank starts at 1.
        
        Args:
            bm25_results: Results from BM25 retriever
            dense_results: Results from Dense retriever
            top_k: Number of results to return
            
        Returns:
            Fused and re-ranked results
        """
        # Create rank maps for each retriever
        rrf_scores = {}
        
        # Add BM25 scores (weighted)
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result.get('chunk_id', id(result))  # Use chunk_id or object id
            rrf_score = self.bm25_weight / (self.rrf_k + rank)
            
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    'chunk': result,
                    'score': 0.0,
                    'bm25_rank': rank,
                    'dense_rank': None
                }
            rrf_scores[chunk_id]['score'] += rrf_score
        
        # Add Dense scores (weighted)
        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result.get('chunk_id', id(result))
            rrf_score = self.dense_weight / (self.rrf_k + rank)
            
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    'chunk': result,
                    'score': 0.0,
                    'bm25_rank': None,
                    'dense_rank': rank
                }
            else:
                rrf_scores[chunk_id]['dense_rank'] = rank
            
            rrf_scores[chunk_id]['score'] += rrf_score
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]
        
        # Prepare final results
        final_results = []
        for item in sorted_results:
            result = item['chunk'].copy()
            result['score'] = float(item['score'])
            result['retrieval_method'] = 'hybrid_rrf'
            result['bm25_rank'] = item['bm25_rank']
            result['dense_rank'] = item['dense_rank']
            final_results.append(result)
        
        return final_results
    
    def search(self, query: str, top_k: int = 5, use_rrf: bool = True) -> List[Dict]:
        """
        Search using hybrid retrieval (BM25 + Dense).
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_rrf: If True, use Reciprocal Rank Fusion; else use weighted scores
            
        Returns:
            List of chunks ranked by hybrid scoring
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Retrieve from both methods (get more candidates for fusion)
        retrieval_k = top_k * 3  # Retrieve 3x more for better fusion
        
        logger.debug(f"Hybrid search: retrieving {retrieval_k} from each method")
        
        bm25_results = self.bm25_retriever.search(query, top_k=retrieval_k)
        dense_results = self.dense_retriever.search(query, top_k=retrieval_k)
        
        logger.debug(f"BM25: {len(bm25_results)} results, Dense: {len(dense_results)} results")
        
        # Fuse results
        if use_rrf:
            final_results = self._reciprocal_rank_fusion(bm25_results, dense_results, top_k)
            logger.debug(f"RRF fusion: {len(final_results)} final results")
        else:
            # Simple weighted score fusion (alternative method)
            final_results = self._weighted_score_fusion(bm25_results, dense_results, top_k)
            logger.debug(f"Weighted fusion: {len(final_results)} final results")
        
        return final_results
    
    def _weighted_score_fusion(self, 
                                bm25_results: List[Dict], 
                                dense_results: List[Dict],
                                top_k: int) -> List[Dict]:
        """
        Alternative fusion: weighted combination of normalized scores.
        
        Args:
            bm25_results: Results from BM25
            dense_results: Results from Dense
            top_k: Number of results to return
            
        Returns:
            Fused results
        """
        combined_scores = {}
        
        # Normalize BM25 scores
        if bm25_results:
            max_bm25 = max(r['score'] for r in bm25_results)
            min_bm25 = min(r['score'] for r in bm25_results)
            bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
            
            for result in bm25_results:
                chunk_id = result.get('chunk_id', id(result))
                normalized_score = (result['score'] - min_bm25) / bm25_range
                weighted_score = normalized_score * self.bm25_weight
                
                combined_scores[chunk_id] = {
                    'chunk': result,
                    'score': weighted_score
                }
        
        # Normalize Dense scores (already cosine similarity 0-1)
        for result in dense_results:
            chunk_id = result.get('chunk_id', id(result))
            weighted_score = result['score'] * self.dense_weight
            
            if chunk_id in combined_scores:
                combined_scores[chunk_id]['score'] += weighted_score
            else:
                combined_scores[chunk_id] = {
                    'chunk': result,
                    'score': weighted_score
                }
        
        # Sort and return top_k
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]
        
        final_results = []
        for item in sorted_results:
            result = item['chunk'].copy()
            result['score'] = float(item['score'])
            result['retrieval_method'] = 'hybrid_weighted'
            final_results.append(result)
        
        return final_results
    
    def save_index(self, path: str):
        """Save both BM25 and Dense indices."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving hybrid index to {path}...")
        
        # Save BM25 index
        self.bm25_retriever.save_index(str(path / "bm25.pkl"))
        
        # Save Dense index
        self.dense_retriever.save_index(str(path / "dense"))
        
        logger.info(f"✅ Hybrid index saved to {path}")
    
    def load_index(self, path: str):
        """Load both BM25 and Dense indices."""
        path = Path(path)
        
        logger.info(f"Loading hybrid index from {path}...")
        
        # Load BM25 index
        self.bm25_retriever.load_index(str(path / "bm25.pkl"))
        
        # Load Dense index
        self.dense_retriever.load_index(str(path / "dense"))
        
        self.is_indexed = True
        logger.info(f"✅ Hybrid index loaded from {path}")
