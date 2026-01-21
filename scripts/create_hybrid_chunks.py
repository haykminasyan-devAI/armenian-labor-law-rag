#!/usr/bin/env python3
"""
Create hybrid chunks: Article-based but split long articles into sub-sections.

Strategy:
- Keep articles together if < 800 tokens
- Split long articles (>800 tokens) into logical sub-sections
- Preserve article number metadata for all chunks
- Maintain context and readability
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import tiktoken
    encoding = tiktoken.encoding_for_model("gpt-4")
    def count_tokens(text):
        return len(encoding.encode(text))
    print("✓ Using tiktoken for accurate token counting")
except ImportError:
    def count_tokens(text):
        return int(len(text.split()) * 1.3)
    print("⚠ Using approximate token counting (install tiktoken for accuracy)")


def split_article_into_subsections(article_text: str, article_num: str, max_tokens: int = 800) -> List[Dict]:
    """
    Split a long article into logical sub-sections.
    
    Args:
        article_text: Full article text
        article_num: Article number
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    # Check if article needs splitting
    article_tokens = count_tokens(article_text)
    
    if article_tokens <= max_tokens:
        # Article is small enough, keep as-is
        return [{
            'text': article_text.strip(),
            'article_number': article_num,
            'subsection': 'full',
            'tokens': article_tokens
        }]
    
    # Article is too long, need to split
    print(f"  Splitting Article {article_num} ({article_tokens} tokens) into subsections...")
    
    # Strategy 1: Split by numbered/lettered sections (մ), ա), բ), etc.)
    section_pattern = r'(\n[ա-ֆ]\)|\n\d+\)|\n[ա-ֆ]\.|\n\d+\.)'
    sections = re.split(section_pattern, article_text)
    
    if len(sections) > 3:  # Found meaningful sections
        current_chunk = []
        current_tokens = 0
        chunk_num = 1
        
        for i, section in enumerate(sections):
            section_tokens = count_tokens(section)
            
            # If single section is too large, split by sentences
            if section_tokens > max_tokens:
                # Add accumulated chunks first
                if current_chunk:
                    chunk_text = ''.join(current_chunk).strip()
                    chunks.append({
                        'text': f"[Հոդված {article_num}, մաս {chunk_num}]\n{chunk_text}",
                        'article_number': article_num,
                        'subsection': f'part_{chunk_num}',
                        'tokens': current_tokens
                    })
                    current_chunk = []
                    current_tokens = 0
                    chunk_num += 1
                
                # Split this large section by sentences
                sentences = re.split(r'([։\.])\s+', section)
                temp_chunk = []
                temp_tokens = 0
                
                for j in range(0, len(sentences), 2):
                    if j + 1 < len(sentences):
                        sent = sentences[j] + sentences[j+1]
                    else:
                        sent = sentences[j]
                    
                    sent_tokens = count_tokens(sent)
                    
                    if temp_tokens + sent_tokens > max_tokens and temp_chunk:
                        # Save accumulated sentences
                        chunk_text = ' '.join(temp_chunk).strip()
                        chunks.append({
                            'text': f"[Հոդված {article_num}, մաս {chunk_num}]\n{chunk_text}",
                            'article_number': article_num,
                            'subsection': f'part_{chunk_num}',
                            'tokens': temp_tokens
                        })
                        temp_chunk = []
                        temp_tokens = 0
                        chunk_num += 1
                    
                    temp_chunk.append(sent)
                    temp_tokens += sent_tokens
                
                # Add remaining
                if temp_chunk:
                    chunk_text = ' '.join(temp_chunk).strip()
                    chunks.append({
                        'text': f"[Հոդված {article_num}, մաս {chunk_num}]\n{chunk_text}",
                        'article_number': article_num,
                        'subsection': f'part_{chunk_num}',
                        'tokens': temp_tokens
                    })
                    chunk_num += 1
                
            elif current_tokens + section_tokens > max_tokens and current_chunk:
                # Save current chunk
                chunk_text = ''.join(current_chunk).strip()
                chunks.append({
                    'text': f"[Հոդված {article_num}, մաս {chunk_num}]\n{chunk_text}",
                    'article_number': article_num,
                    'subsection': f'part_{chunk_num}',
                    'tokens': current_tokens
                })
                current_chunk = [section]
                current_tokens = section_tokens
                chunk_num += 1
            else:
                current_chunk.append(section)
                current_tokens += section_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ''.join(current_chunk).strip()
            chunks.append({
                'text': f"[Հոդված {article_num}, մաս {chunk_num}]\n{chunk_text}",
                'article_number': article_num,
                'subsection': f'part_{chunk_num}',
                'tokens': current_tokens
            })
    
    else:
        # No clear sections, split by sentences
        sentences = re.split(r'([։\.])\s+', article_text)
        current_chunk = []
        current_tokens = 0
        chunk_num = 1
        
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sent = sentences[i] + sentences[i+1]
            else:
                sent = sentences[i]
            
            sent_tokens = count_tokens(sent)
            
            if current_tokens + sent_tokens > max_tokens and current_chunk:
                # Save chunk
                chunk_text = ' '.join(current_chunk).strip()
                chunks.append({
                    'text': f"[Հոդված {article_num}, մաս {chunk_num}]\n{chunk_text}",
                    'article_number': article_num,
                    'subsection': f'part_{chunk_num}',
                    'tokens': current_tokens
                })
                current_chunk = []
                current_tokens = 0
                chunk_num += 1
            
            current_chunk.append(sent)
            current_tokens += sent_tokens
        
        # Final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            chunks.append({
                'text': f"[Հոդված {article_num}, մաս {chunk_num}]\n{chunk_text}",
                'article_number': article_num,
                'subsection': f'part_{chunk_num}',
                'tokens': current_tokens
            })
    
    print(f"    → Split into {len(chunks)} chunks")
    return chunks


def create_hybrid_chunks(input_file: Path, output_file: Path, max_tokens: int = 800):
    """
    Create hybrid chunks from existing article-based chunks.
    
    Args:
        input_file: Path to existing chunks
        output_file: Path to save new chunks
        max_tokens: Maximum tokens per chunk
    """
    print(f"\n🔨 Creating Hybrid Chunks")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Max tokens per chunk: {max_tokens}")
    print("=" * 70)
    
    # Load existing chunks
    with open(input_file, 'r', encoding='utf-8') as f:
        original_chunks = json.load(f)
    
    print(f"\n📥 Loaded {len(original_chunks)} original chunks")
    
    # Process each article
    new_chunks = []
    articles_split = 0
    articles_kept = 0
    
    for chunk in original_chunks:
        article_text = chunk.get('text', '')
        article_num = chunk.get('article_number', 'Unknown')
        
        # Split if needed
        subsections = split_article_into_subsections(article_text, article_num, max_tokens)
        
        if len(subsections) > 1:
            articles_split += 1
        else:
            articles_kept += 1
        
        # Add metadata
        for subsection in subsections:
            subsection['source'] = chunk.get('source', 'labor_law')
            subsection['language'] = chunk.get('language', 'Armenian')
            subsection['char_count'] = len(subsection['text'])
            subsection['word_count'] = len(subsection['text'].split())
        
        new_chunks.extend(subsections)
    
    # Save new chunks
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_chunks, f, ensure_ascii=False, indent=2)
    
    # Statistics
    total_tokens = sum(c['tokens'] for c in new_chunks)
    avg_tokens = total_tokens / len(new_chunks)
    max_chunk_tokens = max(c['tokens'] for c in new_chunks)
    min_chunk_tokens = min(c['tokens'] for c in new_chunks)
    
    print(f"\n✅ Hybrid Chunks Created!")
    print(f"  Original chunks: {len(original_chunks)}")
    print(f"  New chunks: {len(new_chunks)}")
    print(f"  Articles kept whole: {articles_kept}")
    print(f"  Articles split: {articles_split}")
    print(f"\n📊 Token Statistics:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average per chunk: {avg_tokens:.1f} tokens")
    print(f"  Min: {min_chunk_tokens} tokens")
    print(f"  Max: {max_chunk_tokens} tokens")
    print(f"\n💾 Saved to: {output_file}")
    
    # Context window analysis
    print(f"\n🔍 Context Window Impact:")
    for k in [1, 3, 5]:
        top_k_tokens = sorted([c['tokens'] for c in new_chunks], reverse=True)[:k]
        total = sum(top_k_tokens)
        print(f"  Top-{k} chunks: ~{total:,} tokens (avg {total/k:.0f} per chunk)")
    
    available = 8000 - sum(sorted([c['tokens'] for c in new_chunks], reverse=True)[:3]) - 200
    print(f"  Available for answer (top-3): ~{available:,} tokens")


def main():
    """Main function."""
    input_file = project_root / "data" / "chunks" / "labor_law_chunks.json"
    output_file = project_root / "data" / "chunks" / "labor_law_chunks_hybrid.json"
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    create_hybrid_chunks(input_file, output_file, max_tokens=800)
    
    print("\n🎯 Next Steps:")
    print("  1. Build indices: python scripts/build_hybrid_chunks_index.py")
    print("  2. Test in interface with the new chunks")


if __name__ == "__main__":
    main()
