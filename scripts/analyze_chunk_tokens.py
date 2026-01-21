#!/usr/bin/env python3
"""
Analyze token lengths of chunks in the dataset.
Helps understand token distribution and optimize retrieval.
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import tiktoken
    USE_TIKTOKEN = True
    print("✓ Using tiktoken (accurate token counting)")
except ImportError:
    USE_TIKTOKEN = False
    print("⚠ tiktoken not installed, using approximation")
    print("  Install: pip install tiktoken")


def count_tokens_tiktoken(text: str, model: str = "gpt-4") -> int:
    """Count tokens using tiktoken (accurate)."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count.
    Rule of thumb: 1 token ≈ 0.75 words (English)
    For Armenian: slightly more per word
    """
    words = len(text.split())
    # Armenian words are often longer, use 1.3 multiplier
    return int(words * 1.3)


def analyze_chunks(chunks_file: Path):
    """Analyze token distribution in chunks."""
    
    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"\n📊 Analyzing {len(chunks)} chunks from {chunks_file.name}")
    print("=" * 70)
    
    # Count tokens for each chunk
    token_counts = []
    article_tokens = Counter()
    
    for chunk in chunks:
        text = chunk.get('text', '')
        article_num = chunk.get('article_number', 'Unknown')
        
        if USE_TIKTOKEN:
            tokens = count_tokens_tiktoken(text)
        else:
            tokens = count_tokens_approximate(text)
        
        token_counts.append(tokens)
        article_tokens[article_num] += tokens
    
    # Statistics
    total_tokens = sum(token_counts)
    avg_tokens = total_tokens / len(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    median_tokens = sorted(token_counts)[len(token_counts) // 2]
    
    print("\n📈 Token Statistics:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average per chunk: {avg_tokens:.1f} tokens")
    print(f"  Median per chunk: {median_tokens} tokens")
    print(f"  Min per chunk: {min_tokens} tokens")
    print(f"  Max per chunk: {max_tokens} tokens")
    
    # Distribution
    print("\n📊 Token Distribution:")
    bins = [0, 50, 100, 200, 300, 500, 1000, 10000]
    bin_labels = ["0-50", "51-100", "101-200", "201-300", "301-500", "501-1000", "1000+"]
    
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        count = sum(1 for t in token_counts if low < t <= high)
        pct = (count / len(token_counts)) * 100
        bar = "█" * int(pct / 2)
        print(f"  {bin_labels[i]:>10}: {count:>4} chunks ({pct:>5.1f}%) {bar}")
    
    # Context window analysis
    print("\n🔍 Context Window Analysis:")
    for k in [1, 3, 5, 10]:
        # Simulate retrieving top-k chunks
        top_k_tokens = sorted(token_counts, reverse=True)[:k]
        total = sum(top_k_tokens)
        print(f"  Top-{k} chunks: ~{total:,} tokens (avg {total/k:.0f} per chunk)")
    
    # Percentiles
    print("\n📐 Percentiles:")
    sorted_tokens = sorted(token_counts)
    for p in [25, 50, 75, 90, 95, 99]:
        idx = int(len(sorted_tokens) * p / 100)
        print(f"  {p}th percentile: {sorted_tokens[idx]} tokens")
    
    # Top articles by token count
    print("\n📄 Top 10 Articles by Total Tokens:")
    top_articles = article_tokens.most_common(10)
    for article, tokens in top_articles:
        print(f"  Article {article}: {tokens:,} tokens")
    
    # Model context analysis
    print("\n🤖 Model Context Budget Analysis:")
    print("  Model input budget (typical):")
    print(f"    - Llama 3.x: ~8,000 tokens context")
    print(f"    - Your chunks (top-3): ~{sorted(token_counts, reverse=True)[:3]} tokens")
    print(f"    - Prompt template: ~200 tokens")
    print(f"    - Available for answer: ~{8000 - sum(sorted(token_counts, reverse=True)[:3]) - 200:,} tokens")
    
    # Optimization suggestions
    print("\n💡 Optimization Suggestions:")
    if avg_tokens > 300:
        print("  ⚠ Average chunk size is large (>300 tokens)")
        print("    Consider splitting long articles further")
    if max_tokens > 1000:
        print(f"  ⚠ Largest chunk is {max_tokens} tokens")
        print("    Very large chunks may dominate context window")
    
    long_chunks = sum(1 for t in token_counts if t > 500)
    if long_chunks > len(chunks) * 0.1:
        print(f"  ⚠ {long_chunks} chunks exceed 500 tokens ({long_chunks/len(chunks)*100:.1f}%)")
        print("    Consider max_chunk_size parameter in preprocessing")
    
    print("\n✅ Analysis complete!")
    
    return {
        'total_chunks': len(chunks),
        'total_tokens': total_tokens,
        'avg_tokens': avg_tokens,
        'token_counts': token_counts
    }


def main():
    """Main function."""
    chunks_file = project_root / "data" / "chunks" / "labor_law_chunks.json"
    
    if not chunks_file.exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        return
    
    analyze_chunks(chunks_file)


if __name__ == "__main__":
    main()
