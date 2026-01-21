"""
Evaluation metrics for RAG system.

Two categories:
1. Retrieval metrics: How well did we retrieve the right articles?
2. Generation metrics: How good is the generated answer?
"""

from typing import List, Set, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# RETRIEVAL METRICS
# =============================================================================

def recall_at_k(gold_articles: Set[int], retrieved_articles: List[int], k: int) -> float:
    """
    Recall@K: What fraction of relevant articles did we retrieve in top-K?
    
    Formula: |relevant ∩ retrieved[:k]| / |relevant|
    
    Args:
        gold_articles: Set of ground truth article numbers
        retrieved_articles: List of retrieved article numbers (in rank order)
        k: Cutoff position
        
    Returns:
        Recall value (0.0-1.0), or NaN if no gold articles
        
    Example:
        >>> recall_at_k({145, 146, 147}, [145, 150, 146], k=3)
        0.667  # Found 2 out of 3 relevant articles
    """
    if not gold_articles:
        return float("nan")
    
    retrieved_set = set(retrieved_articles[:k])
    hits = retrieved_set & gold_articles
    
    return len(hits) / len(gold_articles)


def precision_at_k(gold_articles: Set[int], retrieved_articles: List[int], k: int) -> float:
    """
    Precision@K: What fraction of retrieved articles are relevant?
    
    Formula: |relevant ∩ retrieved[:k]| / k
    
    Args:
        gold_articles: Set of ground truth article numbers
        retrieved_articles: List of retrieved article numbers (in rank order)
        k: Cutoff position
        
    Returns:
        Precision value (0.0-1.0), or NaN if k <= 0
        
    Example:
        >>> precision_at_k({145, 146}, [145, 150, 146], k=3)
        0.667  # 2 out of 3 retrieved are relevant
    """
    if k <= 0:
        return float("nan")
    
    # Use actual retrieved count if fewer than k
    actual_k = min(k, len(retrieved_articles)) if retrieved_articles else k
    if actual_k == 0:
        return 0.0
    
    retrieved_set = set(retrieved_articles[:k])
    hits = retrieved_set & gold_articles
    
    return len(hits) / actual_k


def hit_at_k(gold_articles: Set[int], retrieved_articles: List[int], k: int) -> int:
    """
    Hit@K (Success@K): Did we retrieve at least one relevant article in top-K?
    
    Returns 1 if yes, 0 if no. Useful for binary success rate.
    
    Args:
        gold_articles: Set of ground truth article numbers
        retrieved_articles: List of retrieved article numbers
        k: Cutoff position
        
    Returns:
        1 if hit, 0 otherwise
        
    Example:
        >>> hit_at_k({145, 146}, [150, 145, 160], k=3)
        1  # Found at least one relevant article (145)
    """
    retrieved_set = set(retrieved_articles[:k])
    hits = retrieved_set & gold_articles
    return int(len(hits) > 0)


def mrr(gold_articles: Set[int], retrieved_articles: List[int]) -> float:
    """
    Mean Reciprocal Rank (MRR): Rank of the first relevant article.
    
    Formula: 1 / rank_of_first_relevant_article
    
    MRR focuses on the rank of the FIRST correct result.
    Higher is better (max = 1.0 if first result is relevant).
    
    Args:
        gold_articles: Set of ground truth article numbers
        retrieved_articles: List of retrieved article numbers (in rank order)
        
    Returns:
        MRR score (0.0-1.0)
        
    Example:
        >>> mrr({145, 146}, [150, 145, 146])
        0.5  # First relevant article is at rank 2 → 1/2 = 0.5
        
        >>> mrr({145, 146}, [145, 150, 146])
        1.0  # First relevant article is at rank 1 → 1/1 = 1.0
    """
    if not gold_articles:
        return 0.0
    
    for rank, article in enumerate(retrieved_articles, start=1):
        if article in gold_articles:
            return 1.0 / rank
    
    return 0.0  # No relevant article found


def average_precision(gold_articles: Set[int], retrieved_articles: List[int]) -> float:
    """
    Average Precision (AP): Average of precision values at each relevant position.
    
    Takes into account both precision and recall at each position.
    
    Formula: (Σ Precision@k * relevance@k) / |relevant|
    
    Args:
        gold_articles: Set of ground truth article numbers
        retrieved_articles: List of retrieved article numbers (in rank order)
        
    Returns:
        AP score (0.0-1.0)
        
    Example:
        >>> average_precision({145, 146}, [145, 150, 146, 160])
        # At rank 1: 145 is relevant, P@1 = 1/1 = 1.0
        # At rank 3: 146 is relevant, P@3 = 2/3 = 0.667
        # AP = (1.0 + 0.667) / 2 = 0.833
    """
    if not gold_articles:
        return 0.0
    
    score = 0.0
    num_hits = 0
    
    for rank, article in enumerate(retrieved_articles, start=1):
        if article in gold_articles:
            num_hits += 1
            precision = num_hits / rank
            score += precision
    
    return score / len(gold_articles) if gold_articles else 0.0


def ndcg_at_k(gold_articles: Set[int], retrieved_articles: List[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K).
    
    Measures ranking quality, with higher weight on top positions.
    Assumes binary relevance (article is relevant or not).
    
    Args:
        gold_articles: Set of ground truth article numbers
        retrieved_articles: List of retrieved article numbers (in rank order)
        k: Cutoff position
        
    Returns:
        NDCG score (0.0-1.0)
    """
    if not gold_articles or k <= 0:
        return 0.0
    
    # DCG: sum of (relevance / log2(rank+1))
    dcg = 0.0
    for rank, article in enumerate(retrieved_articles[:k], start=1):
        relevance = 1.0 if article in gold_articles else 0.0
        dcg += relevance / np.log2(rank + 1)
    
    # IDCG: best possible DCG (all relevant articles at top)
    idcg = 0.0
    for rank in range(1, min(k, len(gold_articles)) + 1):
        idcg += 1.0 / np.log2(rank + 1)
    
    return dcg / idcg if idcg > 0 else 0.0


# =============================================================================
# GENERATION METRICS (Answer Quality)
# =============================================================================

def compute_f1_score(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 score between prediction and reference.
    
    Treats both as bags of tokens and computes F1 based on overlap.
    Standard metric for QA evaluation.
    
    Args:
        prediction: Generated answer text
        reference: Gold/reference answer text
        
    Returns:
        F1 score (0.0-1.0)
        
    Example:
        >>> compute_f1_score(
        ...     "Արձակուրդը 20 օր է",
        ...     "Արձակուրդի տևողությունը 20 օր է"
        ... )
        0.667  # 2 matching tokens out of 3 predicted, 2 out of 4 reference
    """
    # Tokenize (simple split by whitespace and lowercase)
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # Calculate overlap
    common_tokens = pred_tokens & ref_tokens
    
    if not common_tokens:
        return 0.0
    
    # Precision = common / predicted
    precision = len(common_tokens) / len(pred_tokens)
    
    # Recall = common / reference
    recall = len(common_tokens) / len(ref_tokens)
    
    # F1 = harmonic mean
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def semantic_similarity(text1: str, text2: str, model_name: str = 'Metric-AI/armenian-text-embeddings-1') -> float:
    """
    Compute semantic similarity between two texts using embeddings.
    
    Better than F1 for Armenian because it handles:
    - Morphological variations (արձակուրդի vs արձակուրդը)
    - Synonyms (աշխատող vs աշխատակից)
    - Paraphrasing (different words, same meaning)
    
    Args:
        text1: First text
        text2: Second text  
        model_name: Embedding model (default: Armenian embeddings)
        
    Returns:
        Similarity score (0.0-1.0), higher = more similar
        
    Example:
        >>> semantic_similarity(
        ...     "Արձակուրդի տևողությունը 20 օր է",
        ...     "Արձակուրդը 20 աշխատանքային օր է"
        ... )
        0.92  # Very similar meaning!
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    try:
        # Load embedding model (cached after first use)
        model = SentenceTransformer(model_name)
        
        # Encode both texts
        emb1 = model.encode([text1], show_progress_bar=False)
        emb2 = model.encode([text2], show_progress_bar=False)
        
        # Compute cosine similarity
        similarity = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Error computing semantic similarity: {e}")
        return 0.0


# =============================================================================
# OLD GENERATION METRICS (kept for reference)
# =============================================================================

def article_citation_accuracy(answer: str, retrieved_articles: List[int]) -> Dict[str, float]:
    """
    Check if cited articles in the answer are actually in retrieved context.
    
    Detects hallucinated article citations.
    
    Args:
        answer: Generated answer text
        retrieved_articles: List of article numbers that were retrieved
        
    Returns:
        Dict with citation metrics
    """
    import re
    
    # Extract cited articles from answer (Armenian: Հոդված 145)
    cited_pattern = r'Հոդված\s+(\d+)'
    cited_articles = [int(num) for num in re.findall(cited_pattern, answer)]
    
    if not cited_articles:
        return {
            'cited_count': 0,
            'correct_citations': 0,
            'hallucinated_citations': 0,
            'citation_accuracy': 1.0  # No citations = no errors
        }
    
    retrieved_set = set(retrieved_articles)
    correct = sum(1 for art in cited_articles if art in retrieved_set)
    hallucinated = len(cited_articles) - correct
    
    return {
        'cited_count': len(cited_articles),
        'correct_citations': correct,
        'hallucinated_citations': hallucinated,
        'citation_accuracy': correct / len(cited_articles) if cited_articles else 1.0
    }


def detect_hallucination(answer: str, 
                        retrieved_articles: List[int],
                        context: str = "") -> Dict[str, any]:
    """
    Detect hallucinations in the generated answer.
    
    Hallucination types:
    1. Citation hallucination: Cites articles not in retrieved context
    2. Content hallucination: Contains specific claims not found in context
    
    Args:
        answer: Generated answer text
        retrieved_articles: Article numbers that were retrieved
        context: Retrieved context text
        
    Returns:
        Dict with hallucination metrics
    """
    import re
    
    hallucination_report = {
        'has_hallucination': False,
        'citation_hallucination': False,
        'hallucinated_articles': [],
        'hallucination_count': 0,
        'hallucination_severity': 'none'  # none, low, high
    }
    
    # 1. Citation hallucination: Check if cited articles are in retrieved set
    cited_pattern = r'Հոդված\s+(\d+)'
    cited_articles = [int(num) for num in re.findall(cited_pattern, answer)]
    
    if cited_articles:
        retrieved_set = set(retrieved_articles)
        hallucinated = [art for art in cited_articles if art not in retrieved_set]
        
        if hallucinated:
            hallucination_report['citation_hallucination'] = True
            hallucination_report['has_hallucination'] = True
            hallucination_report['hallucinated_articles'] = hallucinated
            hallucination_report['hallucination_count'] = len(hallucinated)
            
            # Severity based on percentage of hallucinated citations
            hallucination_rate = len(hallucinated) / len(cited_articles)
            if hallucination_rate >= 0.5:
                hallucination_report['hallucination_severity'] = 'high'
            else:
                hallucination_report['hallucination_severity'] = 'low'
    
    # 2. Content hallucination detection (basic heuristics)
    if context:
        # Check for specific numbers/dates in answer that don't appear in context
        # Extract numbers from answer and context
        answer_numbers = set(re.findall(r'\b\d+\b', answer))
        context_numbers = set(re.findall(r'\b\d+\b', context))
        
        # Numbers in answer but not in context (potential hallucination)
        novel_numbers = answer_numbers - context_numbers
        
        # Filter out article citations (already checked) and small numbers (likely safe)
        novel_numbers = {num for num in novel_numbers 
                        if int(num) not in cited_articles and int(num) > 10}
        
        if novel_numbers and not hallucination_report['has_hallucination']:
            hallucination_report['has_hallucination'] = True
            hallucination_report['hallucination_severity'] = 'low'
            hallucination_report['novel_numbers'] = list(novel_numbers)
    
    return hallucination_report


# =============================================================================
# AGGREGATED METRICS
# =============================================================================

def evaluate_retrieval(gold_articles: Set[int], 
                       retrieved_articles: List[int],
                       k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
    """
    Compute all retrieval metrics at once.
    
    Args:
        gold_articles: Set of ground truth article numbers
        retrieved_articles: List of retrieved article numbers (in rank order)
        k_values: List of K values to evaluate at
        
    Returns:
        Dict with all retrieval metrics
    """
    metrics = {}
    
    # Metrics at different K values
    for k in k_values:
        metrics[f'recall@{k}'] = recall_at_k(gold_articles, retrieved_articles, k)
        metrics[f'precision@{k}'] = precision_at_k(gold_articles, retrieved_articles, k)
        metrics[f'hit@{k}'] = hit_at_k(gold_articles, retrieved_articles, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(gold_articles, retrieved_articles, k)
    
    # Rank-based metrics (no K cutoff)
    metrics['mrr'] = mrr(gold_articles, retrieved_articles)
    metrics['average_precision'] = average_precision(gold_articles, retrieved_articles)
    
    return metrics


def evaluate_answer(answer: str, 
                    retrieved_articles: List[int],
                    context: str = "",
                    reference_answer: str = "") -> Dict[str, any]:
    """
    Compute all answer quality metrics.
    
    Args:
        answer: Generated answer text
        retrieved_articles: Article numbers that were retrieved
        context: Retrieved context text (optional)
        reference_answer: Gold/reference answer for comparison (optional)
        
    Returns:
        Dict with answer quality metrics
    """
    metrics = {}
    
    # Citation accuracy
    citation_metrics = article_citation_accuracy(answer, retrieved_articles)
    metrics.update(citation_metrics)
    
    # Hallucination detection
    hallucination_metrics = detect_hallucination(answer, retrieved_articles, context)
    metrics.update(hallucination_metrics)
    
    # Answer similarity metrics (if reference provided)
    if reference_answer:
        # F1 token overlap
        metrics['f1_score'] = compute_f1_score(answer, reference_answer)
        
        # Semantic similarity (meaning-based)
        metrics['semantic_similarity'] = semantic_similarity(answer, reference_answer)
    
    # Context usage (if context provided)
    if context:
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(context_words & answer_words)
        metrics['context_overlap_ratio'] = overlap / len(answer_words) if answer_words else 0.0
    
    return metrics


# =============================================================================
# DATASET EVALUATION
# =============================================================================

class RAGEvaluator:
    """
    Comprehensive evaluator for RAG system.
    
    Evaluates both retrieval and generation quality.
    """
    
    def __init__(self):
        self.results = []
    
    def evaluate_single(self,
                       question: str,
                       gold_articles: Set[int],
                       retrieved_articles: List[int],
                       answer: str,
                       context: str = "",
                       reference_answer: str = "") -> Dict:
        """
        Evaluate a single question.
        
        Args:
            question: The question
            gold_articles: Ground truth relevant articles
            retrieved_articles: Articles retrieved by the system
            answer: Generated answer
            context: Retrieved context text
            
        Returns:
            Dict with all metrics for this question
        """
        result = {
            'question': question,
            'gold_articles': list(gold_articles),
            'retrieved_articles': retrieved_articles[:10],  # Top-10
        }
        
        # Retrieval metrics
        retrieval_metrics = evaluate_retrieval(gold_articles, retrieved_articles)
        result['retrieval'] = retrieval_metrics
        
        # Answer metrics
        answer_metrics = evaluate_answer(answer, retrieved_articles, context, reference_answer)
        result['answer'] = answer_metrics
        
        self.results.append(result)
        return result
    
    def get_aggregate_metrics(self) -> Dict:
        """
        Compute average metrics across all evaluated questions.
        
        Returns:
            Dict with averaged metrics
        """
        if not self.results:
            return {}
        
        # Aggregate retrieval metrics
        retrieval_keys = self.results[0]['retrieval'].keys()
        aggregated = {}
        
        for key in retrieval_keys:
            values = [r['retrieval'][key] for r in self.results]
            # Filter out NaNs
            values = [v for v in values if not np.isnan(v)]
            if values:
                aggregated[f'avg_{key}'] = np.mean(values)
                aggregated[f'std_{key}'] = np.std(values)
        
        # Aggregate answer metrics
        answer_keys = self.results[0]['answer'].keys()
        for key in answer_keys:
            values = [r['answer'][key] for r in self.results]
            if values and isinstance(values[0], (int, float)):
                aggregated[f'avg_{key}'] = np.mean(values)
        
        aggregated['total_questions'] = len(self.results)
        
        return aggregated
    
    def print_summary(self):
        """Print evaluation summary."""
        metrics = self.get_aggregate_metrics()
        
        print("="*60)
        print("RAG EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Questions: {metrics.get('total_questions', 0)}")
        print("\nRetrieval Metrics:")
        print(f"  MRR:           {metrics.get('avg_mrr', 0):.3f}")
        print(f"  Recall@3:      {metrics.get('avg_recall@3', 0):.3f}")
        print(f"  Precision@3:   {metrics.get('avg_precision@3', 0):.3f}")
        print(f"  Hit@3:         {metrics.get('avg_hit@3', 0):.3f}")
        print(f"  NDCG@3:        {metrics.get('avg_ndcg@3', 0):.3f}")
        print("\nAnswer Quality:")
        print(f"  Citation Accuracy:     {metrics.get('avg_citation_accuracy', 0):.3f}")
        print(f"  Hallucination Rate:    {metrics.get('avg_hallucination_count', 0):.2f}")
        
        # Count questions with hallucinations
        if self.results:
            hallucinated_count = sum(1 for r in self.results if r['answer'].get('has_hallucination', False))
            hallucination_pct = (hallucinated_count / len(self.results)) * 100
            print(f"  Questions w/ Halluc:   {hallucinated_count}/{len(self.results)} ({hallucination_pct:.1f}%)")
        
        # Answer similarity metrics (if available)
        if 'avg_f1_score' in metrics:
            print(f"\n  F1 Token Score:        {metrics.get('avg_f1_score', 0):.3f}")
            print(f"  Semantic Similarity:   {metrics.get('avg_semantic_similarity', 0):.3f}")
        
        print("="*60)
