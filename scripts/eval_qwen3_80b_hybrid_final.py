"""
Comprehensive Evaluation: Hybrid Retriever (New Chunks) + Qwen 3 Next 80B (API)
All metrics: Retrieval (MRR, Recall, Precision) + Answer (F1, Semantic, Hallucination)
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_pipeline import RAGPipeline
from src.generation.generator import LLMGenerator
from src.evaluation.metrics import RAGEvaluator

print("="*80)
print("COMPREHENSIVE EVALUATION")
print("Configuration: Hybrid Retriever (New Chunks) + Qwen 3 Next 80B (API)")
print("All Metrics: Retrieval + Answer Quality + Semantic Similarity + Sentence Completion")
print("="*80)

# Load test dataset
print("\n[1/5] Loading test dataset...")
with open('data/evaluation/test_set.json', 'r', encoding='utf-8') as f:
    test_set = json.load(f)

print(f"      ✅ {len(test_set)} questions loaded")

# Load NEW HYBRID chunks  
print("\n[2/5] Loading hybrid chunks (token-optimized)...")
with open('data/chunks/labor_law_chunks_hybrid.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)
avg_tokens = sum(c.get('tokens', 0) for c in chunks) / len(chunks)
print(f"      ✅ {len(chunks)} chunks loaded (avg {avg_tokens:.0f} tokens)")

# Initialize Hybrid retriever with NEW index
print("\n[3/5] Initializing Hybrid retriever...")
retriever = HybridRetriever(chunks, bm25_weight=0.5, dense_weight=0.5)
hybrid_index = Path('data/indices/hybrid_v2')

if hybrid_index.exists():
    print("      Loading index...")
    retriever.load_index(str(hybrid_index))
    print("      ✅ Hybrid index loaded (BM25 + Dense, RRF fusion)")
else:
    print("      ❌ Index not found!")
    sys.exit(1)

# Initialize Qwen 3 Next 80B via NVIDIA API
print("\n[4/5] Initializing Qwen 3 Next 80B (NVIDIA API)...")
print("      Connecting to API...")

api_key = "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"
generator = LLMGenerator(
    model_name="qwen/qwen3-next-80b-a3b-instruct",
    provider="nvidia",
    api_key=api_key,
    max_tokens=2000,
    temperature=0.1
)
print("      ✅ Qwen 3 Next 80B ready")

# Create pipeline
pipeline = RAGPipeline(retriever=retriever, generator=generator)

# Initialize evaluator
evaluator = RAGEvaluator()

# Sentence completion checker
def is_sentence_complete(text):
    """Check if text ends with complete sentence."""
    text = text.strip()
    complete_endings = ('։', '.', '?', '!', '»', '"', '."', '։"', '!»', '?»')
    return text.endswith(complete_endings)

# Run evaluation
print("\n[5/5] Running Comprehensive Evaluation...")
print("="*80)
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print(f"Questions: {len(test_set)}")
print(f"Estimated time: ~{len(test_set) * 10 / 60:.0f} minutes (API is fast)")
print("="*80)

start_time = time.time()
errors = 0
incomplete_sentences = 0

for i, test_case in enumerate(test_set, 1):
    question_short = test_case['question'][:50].replace('\n', ' ')
    elapsed = (time.time() - start_time) / 60
    
    print(f"\n[{i}/{len(test_set)}] ({elapsed:.1f}min)")
    print(f"  Q: {question_short}...")
    print(f"  Gold: {test_case['gold_articles']}")
    
    try:
        # Run RAG
        result = pipeline.answer_question(
            test_case['question'],
            top_k=5,
            return_context=True
        )
        
        print(f"  Retrieved: {result['article_numbers'][:3]}")
        
        # Check sentence completion
        answer = result['answer']
        is_complete = is_sentence_complete(answer)
        if not is_complete:
            incomplete_sentences += 1
            print(f"  ⚠️  Incomplete sentence! Ends with: ...{answer[-50:]}")
        else:
            print(f"  ✅ Complete sentence")
        
        # Extract context
        context_text = "\n".join([c['text'] for c in result.get('retrieved_chunks', [])])
        reference_answer = test_case.get('reference_answer', '')
        
        # Evaluate with ALL metrics
        metrics = evaluator.evaluate_single(
            question=test_case['question'],
            gold_articles=set(test_case['gold_articles']),
            retrieved_articles=result['article_numbers'],
            answer=result['answer'],
            context=context_text,
            reference_answer=reference_answer
        )
        
        # Display metrics
        print(f"\n  📊 RETRIEVAL:")
        print(f"     MRR: {metrics['retrieval']['mrr']:.3f} | "
              f"Recall@3: {metrics['retrieval']['recall@3']:.3f} | "
              f"Precision@3: {metrics['retrieval']['precision@3']:.3f}")
        
        print(f"  📝 ANSWER QUALITY:")
        print(f"     Citation Acc: {metrics['answer']['citation_accuracy']:.3f} | "
              f"Hallucination: {'No ✓' if not metrics['answer']['has_hallucination'] else 'Yes ✗'}")
        
        # Show similarity metrics if available
        if reference_answer and 'f1_score' in metrics['answer']:
            print(f"  🎯 SIMILARITY TO REFERENCE:")
            print(f"     F1 Score: {metrics['answer']['f1_score']:.3f} | "
                  f"Semantic: {metrics['answer']['semantic_similarity']:.3f}")
        
    except Exception as e:
        errors += 1
        print(f"  ✗ ERROR: {str(e)[:80]}")
        continue

# Final summary
elapsed_total = time.time() - start_time

print("\n" + "="*80)
print("🎉 FINAL RESULTS - Qwen 3 Next 80B + Hybrid + New Chunks")
print("="*80)
print(f"Configuration: Hybrid (New Chunks) + Qwen 3 Next 80B (API)")
print(f"Questions processed: {len(evaluator.results)}/{len(test_set)}")
print(f"Errors: {errors}")
print(f"Total time: {elapsed_total/60:.1f} minutes")
print(f"📝 Sentence Completion: {len(evaluator.results) - incomplete_sentences}/{len(evaluator.results)} complete ({(1-incomplete_sentences/len(evaluator.results))*100:.1f}%)")
print("="*80)

# Print detailed summary
evaluator.print_summary()

# Save results
results = {
    'configuration': 'Hybrid (New Chunks) + Qwen 3 Next 80B (API)',
    'retrieval_method': 'Hybrid (BM25 + Dense, RRF fusion)',
    'chunks': 'Hybrid (token-optimized, avg 444 tokens)',
    'generation_model': 'qwen/qwen3-next-80b-a3b-instruct',
    'num_questions': len(test_set),
    'num_completed': len(evaluator.results),
    'num_errors': errors,
    'incomplete_sentences': incomplete_sentences,
    'elapsed_time_minutes': round(elapsed_total / 60, 1),
    'aggregate_metrics': evaluator.get_aggregate_metrics(),
    'detailed_results': evaluator.results
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'results/evaluation/qwen3_80b_hybrid_chunks_{timestamp}.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ Detailed results saved to: {output_file}")

# Create summary report
summary_file = f'results/evaluation/qwen3_80b_hybrid_chunks_summary_{timestamp}.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("QWEN 3 NEXT 80B + HYBRID (NEW CHUNKS) EVALUATION SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    agg = evaluator.get_aggregate_metrics()
    
    f.write("RETRIEVAL METRICS:\n")
    f.write(f"  MRR:           {agg.get('avg_mrr', 0):.3f}\n")
    f.write(f"  Recall@3:      {agg.get('avg_recall@3', 0):.3f}\n")
    f.write(f"  Precision@3:   {agg.get('avg_precision@3', 0):.3f}\n")
    f.write(f"  Hit@3:         {agg.get('avg_hit@3', 0):.3f}\n")
    f.write(f"  NDCG@3:        {agg.get('avg_ndcg@3', 0):.3f}\n")
    
    f.write("\nANSWER QUALITY:\n")
    f.write(f"  Citation Accuracy:     {agg.get('avg_citation_accuracy', 0):.3f}\n")
    f.write(f"  Hallucination Count:   {agg.get('avg_hallucination_count', 0):.2f}\n")
    f.write(f"  Sentence Completion:   {(1-incomplete_sentences/len(evaluator.results))*100:.1f}%\n")
    
    if 'avg_f1_score' in agg:
        f.write(f"\nANSWER SIMILARITY:\n")
        f.write(f"  F1 Token Score:        {agg.get('avg_f1_score', 0):.3f}\n")
        f.write(f"  Semantic Similarity:   {agg.get('avg_semantic_similarity', 0):.3f}\n")
    
    f.write(f"\nPERFORMANCE:\n")
    f.write(f"  Total Time:            {elapsed_total/60:.1f} minutes\n")
    f.write(f"  Avg per Question:      {elapsed_total/len(test_set):.1f}s\n")
    f.write(f"  Errors:                {errors}/{len(test_set)}\n")
    
    f.write("\n" + "="*80 + "\n")

print(f"✅ Summary saved to: {summary_file}")
print("="*80)
