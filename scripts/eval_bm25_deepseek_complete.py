"""
Complete Evaluation: BM25 + DeepSeek V3.1
Includes: MRR, Recall, Hit@K, F1 Score, Semantic Similarity
Optimized to avoid slowdowns.
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.bm25_retriever import BM25Retriever
from src.generation.rag_pipeline import RAGPipeline
from src.generation.generator import LLMGenerator
from src.evaluation.metrics import (
    mrr, recall_at_k, precision_at_k, hit_at_k,
    compute_f1_score, article_citation_accuracy, detect_hallucination
)

print("="*80)
print("COMPLETE EVALUATION: BM25 + DeepSeek V3.1")
print("All Metrics: MRR, Recall, Hit@K, F1, Semantic Similarity")
print("="*80)

# Load data
print("\n[1/5] Loading test dataset...")
with open('data/evaluation/test_set.json', 'r', encoding='utf-8') as f:
    test_set = json.load(f)
print(f"      ✅ {len(test_set)} questions")

print("\n[2/5] Loading chunks...")
with open('data/chunks/labor_law_chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)
print(f"      ✅ {len(chunks)} chunks")

# Initialize BM25
print("\n[3/5] Loading BM25 retriever...")
retriever = BM25Retriever(chunks)
retriever.load_index('indices/bm25/bm25_index.pkl')
print("      ✅ BM25 loaded")

# Initialize DeepSeek
print("\n[4/5] Initializing DeepSeek V3.1...")
generator = LLMGenerator(
    model_name="deepseek-ai/deepseek-v3.1",
    provider="nvidia",
    api_key="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",
    max_tokens=2000,
    temperature=0.1
)
print("      ✅ DeepSeek ready")

# Pre-load semantic model ONCE (key optimization!)
print("\n[5/5] Pre-loading semantic model...")
from sentence_transformers import SentenceTransformer
import numpy as np
sem_model = SentenceTransformer('Metric-AI/armenian-text-embeddings-1')
print("      ✅ Ready")

# Create pipeline
pipeline = RAGPipeline(retriever=retriever, generator=generator)

# Evaluate
print("\n" + "="*80)
print("EVALUATING")
print("="*80)
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print(f"Questions: {len(test_set)}")
print("="*80)

start_time = time.time()
results = []

for i, test_case in enumerate(test_set, 1):
    q_short = test_case['question'][:45].replace('\n', ' ')
    elapsed = (time.time() - start_time) / 60
    
    print(f"\n[{i}/{len(test_set)}] ({elapsed:.1f}min) {q_short}...")
    
    try:
        # Run RAG
        result = pipeline.answer_question(test_case['question'], top_k=5, return_context=True)
        
        gold = set(test_case['gold_articles'])
        retrieved = result['article_numbers']
        generated = result['answer']
        reference = test_case.get('reference_answer', '')
        context = "\n".join([c['text'] for c in result.get('retrieved_chunks', [])])
        
        # Retrieval metrics
        mrr_val = mrr(gold, retrieved)
        hit1 = hit_at_k(gold, retrieved, 1)
        hit3 = hit_at_k(gold, retrieved, 3)
        hit5 = hit_at_k(gold, retrieved, 5)
        recall3 = recall_at_k(gold, retrieved, 3)
        
        # Answer quality
        citation = article_citation_accuracy(generated, retrieved)
        halluc = detect_hallucination(generated, retrieved, context)
        
        # F1 and Semantic (only if reference exists)
        f1_val = 0.0
        sem_val = 0.0
        
        if reference:
            f1_val = compute_f1_score(generated, reference)
            
            # Semantic similarity with pre-loaded model
            emb1 = sem_model.encode([generated], show_progress_bar=False)
            emb2 = sem_model.encode([reference], show_progress_bar=False)
            sem_val = float(np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0])))
        
        # Store all data
        results.append({
            'question_id': test_case.get('id', i),
            'question': test_case['question'],
            'gold_articles': test_case['gold_articles'],
            'retrieved_articles': retrieved[:5],
            'generated_answer': generated,
            'reference_answer': reference,
            'metrics': {
                'mrr': mrr_val,
                'recall@3': recall3,
                'hit@1': hit1,
                'hit@3': hit3,
                'hit@5': hit5,
                'citation_accuracy': citation['citation_accuracy'],
                'has_hallucination': halluc['has_hallucination'],
                'f1_score': f1_val,
                'semantic_similarity': sem_val
            }
        })
        
        # Display
        halluc_str = "✗" if halluc['has_hallucination'] else "✓"
        print(f"  MRR={mrr_val:.2f} Hit@3={hit3} | F1={f1_val:.3f} Sem={sem_val:.3f} | Halluc={halluc_str}")
        
        # Rate limit
        time.sleep(2)
        
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)[:70]}")
        continue

# Calculate averages
elapsed = (time.time() - start_time) / 60

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Completed: {len(results)}/90")
print(f"Time: {elapsed:.1f} min")
print("="*80)

if results:
    # Averages
    avg_mrr = np.mean([r['metrics']['mrr'] for r in results])
    avg_recall3 = np.mean([r['metrics']['recall@3'] for r in results])
    avg_hit1 = np.mean([r['metrics']['hit@1'] for r in results])
    avg_hit3 = np.mean([r['metrics']['hit@3'] for r in results])
    
    # Only for questions with references
    with_ref = [r for r in results if r['reference_answer']]
    avg_f1 = np.mean([r['metrics']['f1_score'] for r in with_ref]) if with_ref else 0
    avg_sem = np.mean([r['metrics']['semantic_similarity'] for r in with_ref]) if with_ref else 0
    
    # Hallucinations
    halluc_count = sum(1 for r in results if r['metrics']['has_hallucination'])
    
    print("\nRETRIEVAL:")
    print(f"  MRR:     {avg_mrr:.3f}")
    print(f"  Recall@3: {avg_recall3:.3f}")
    print(f"  Hit@1:   {avg_hit1:.3f}")
    print(f"  Hit@3:   {avg_hit3:.3f}")
    
    print("\nANSWER QUALITY:")
    print(f"  F1 Score:            {avg_f1:.3f}")
    print(f"  Semantic Similarity: {avg_sem:.3f}")
    print(f"  Hallucinations:      {halluc_count}/{len(results)} ({halluc_count/len(results)*100:.1f}%)")
    
    print("="*80)

# Save
output = {
    'configuration': 'BM25 + DeepSeek V3.1',
    'completed': len(results),
    'time_minutes': round(elapsed, 1),
    'average_metrics': {
        'mrr': avg_mrr if results else 0,
        'recall@3': avg_recall3 if results else 0,
        'hit@1': avg_hit1 if results else 0,
        'hit@3': avg_hit3 if results else 0,
        'f1_score': avg_f1,
        'semantic_similarity': avg_sem,
        'hallucination_rate': halluc_count / len(results) if results else 0
    },
    'detailed_results': results
}

with open('results/evaluation/bm25_deepseek_complete.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n✅ Complete results saved to: results/evaluation/bm25_deepseek_complete.json")
print("="*80)
