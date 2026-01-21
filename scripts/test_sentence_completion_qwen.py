"""
Test sentence completion with Qwen 3 Next 80B on OrinAi data
"""

import sys
from pathlib import Path
import json
import re

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_pipeline import RAGPipeline
from src.generation.generator import LLMGenerator

def is_sentence_complete_armenian(text):
    """Check if Armenian text ends with complete sentence.
    In Armenian, sentences ALWAYS end with ։ (Armenian full stop)
    """
    text = text.strip()
    # Armenian sentence must end with ։
    # Can also have quotes or other punctuation after it
    complete_endings = ('։')
    return text.endswith(complete_endings)

def parse_orinai_data(file_path):
    """Parse OrinAi_1 (1).txt format into questions."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by numbered questions
    questions = []
    pattern = r'\d+\.\s*Հարց\s*\n\n(.*?)\n\nՊատասխան\n(.*?)(?=\n\nՀղումներ|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for i, (question, answer) in enumerate(matches[:10], 1):  # Test on first 10
        questions.append({
            'id': i,
            'question': question.strip(),
            'reference_answer': answer.strip()
        })
    
    return questions

print("="*80)
print("🧪 SENTENCE COMPLETION TEST - Qwen 3 Next 80B")
print("Testing Armenian sentence endings (։ . ? !)")
print("="*80)

# Load chunks
print("\n[1/4] Loading hybrid chunks...")
with open('data/chunks/labor_law_chunks_hybrid.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)
print(f"      ✅ {len(chunks)} chunks loaded")

# Initialize retriever
print("\n[2/4] Loading Hybrid retriever...")
retriever = HybridRetriever(chunks, bm25_weight=0.5, dense_weight=0.5)
retriever.load_index('data/indices/hybrid_v2')
print("      ✅ Hybrid index loaded")

# Initialize Qwen 3 Next 80B
print("\n[3/4] Initializing Qwen 3 Next 80B (API)...")
api_key = "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"
generator = LLMGenerator(
    model_name="qwen/qwen3-next-80b-a3b-instruct",
    provider="nvidia",
    api_key=api_key,
    max_tokens=6000,
    temperature=0.1
)
print("      ✅ Qwen 3 Next 80B ready")

# Create pipeline
pipeline = RAGPipeline(retriever=retriever, generator=generator)

# Parse test questions
print("\n[4/4] Loading test questions from OrinAi_1 (1).txt...")
test_questions = parse_orinai_data('OrinAi_1 (1).txt')
print(f"      ✅ {len(test_questions)} questions loaded")

print("\n" + "="*80)
print("RUNNING TESTS...")
print("="*80)

complete_count = 0
incomplete_count = 0

for test_case in test_questions:
    print(f"\n[Question {test_case['id']}/10]")
    print(f"Q: {test_case['question'][:60]}...")
    
    # Get answer
    result = pipeline.answer_question(
        test_case['question'],
        top_k=5,
        return_context=False
    )
    
    answer = result['answer']
    is_complete = is_sentence_complete_armenian(answer)
    
    # Show last 80 chars
    last_chars = answer[-80:].replace('\n', ' ')
    
    if is_complete:
        complete_count += 1
        print(f"  ✅ COMPLETE - Ends: ...{last_chars}")
    else:
        incomplete_count += 1
        print(f"  ❌ INCOMPLETE - Ends: ...{last_chars}")
    
    print(f"  Last char: '{answer[-1]}' (ord: {ord(answer[-1])})")

print("\n" + "="*80)
print("📊 FINAL RESULTS")
print("="*80)
print(f"Total Questions: {len(test_questions)}")
print(f"✅ Complete sentences: {complete_count}/{len(test_questions)} ({complete_count/len(test_questions)*100:.1f}%)")
print(f"❌ Incomplete sentences: {incomplete_count}/{len(test_questions)} ({incomplete_count/len(test_questions)*100:.1f}%)")
print("="*80)

if complete_count < len(test_questions) * 0.8:
    print("\n⚠️  WARNING: Less than 80% sentence completion!")
    print("   Suggestion: Update prompt to emphasize shorter, complete answers")
else:
    print("\n🎉 SUCCESS: Good sentence completion rate!")
