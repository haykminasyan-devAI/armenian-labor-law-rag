"""
Evaluation module for RAG system.

Provides metrics for:
- Retrieval quality (Recall@K, Precision@K, MRR, NDCG)
- Answer quality (Citation accuracy, Hallucination detection)
"""

from .metrics import (
    # Retrieval metrics
    recall_at_k,
    precision_at_k,
    hit_at_k,
    mrr,
    average_precision,
    ndcg_at_k,
    evaluate_retrieval,
    
    # Answer quality metrics
    article_citation_accuracy,
    detect_hallucination,
    evaluate_answer,
    
    # Evaluator class
    RAGEvaluator
)

__all__ = [
    # Retrieval
    'recall_at_k',
    'precision_at_k',
    'hit_at_k',
    'mrr',
    'average_precision',
    'ndcg_at_k',
    'evaluate_retrieval',
    
    # Answer quality
    'article_citation_accuracy',
    'detect_hallucination',
    'evaluate_answer',
    
    # Evaluator
    'RAGEvaluator'
]
