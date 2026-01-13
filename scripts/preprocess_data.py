import sys
import json
from pathlib import Path
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import DocumentCleaner
from src.preprocessing import ArmenianLegalChunker, ChunkingStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():

    input_file = project_root / "data" / "processed" / "labor_law.txt"
    output_file = project_root / "data" / "chunks" / "labor_law_chunks.json"
    stats_file = project_root / "data" / "chunks" / "chunking_stats.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    
    logger.info("PREPROCESSING PIPELINE: Armenian Labor Law")
    

    logger.info("\n📖 Step 1: Loading text...")

    with open(input_file, 'r', encoding = 'utf-8') as f:
        raw_text = f.read()

    logger.info(f"Loaded {len(raw_text):,} characters")


    cleaner = DocumentCleaner(
        remove_extra_whitespace=True,
        remove_page_numbers=True,
        normalize_unicode=True
    )
    clean_text = cleaner.clean_text(raw_text)
    logger.info(f"   Cleaned text: {len(clean_text):,} characters")
    logger.info(f"   Removed: {len(raw_text) - len(clean_text):,} characters")
    

    logger.info("\n📦 Step 3: Chunking text...")
    chunker = ArmenianLegalChunker(
        strategy=ChunkingStrategy.ARTICLE_BASED,
        chunk_size=512,
        overlap=50
    )

    chunks = chunker.chunk_text(
        clean_text,
        metadata={'source': 'labor_law.pdf', 'language': 'Armenian'}
    )
    
    # Calculate statistics
    total_chars = sum(c['char_count'] for c in chunks)
    total_words = sum(c['word_count'] for c in chunks)
    avg_chars = total_chars / len(chunks) if chunks else 0
    avg_words = total_words / len(chunks) if chunks else 0
    
    stats = {
        'total_chunks': len(chunks),
        'total_chars': total_chars,
        'total_words': total_words,
        'avg_chars_per_chunk': avg_chars,
        'avg_words_per_chunk': avg_words,
        'chunking_strategy': chunker.strategy.value,
        'chunk_size': chunker.chunk_size,
        'overlap': chunker.overlap
    }
    
    logger.info(f"\n📊 Statistics:")
    logger.info(f"   Total chunks: {stats['total_chunks']}")
    logger.info(f"   Avg chars/chunk: {stats['avg_chars_per_chunk']:.0f}")
    logger.info(f"   Avg words/chunk: {stats['avg_words_per_chunk']:.0f}")

    # Step 4: Save chunks
    logger.info(f"\n💾 Step 4: Saving chunks...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"   Saved to: {output_file}")
    
    # Save statistics
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"   Stats saved to: {stats_file}")
    
    # Show sample chunks
    logger.info(f"\n📄 Sample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        logger.info(f"\n   Chunk {i}:")
        logger.info(f"   - Article: {chunk.get('article_number', 'N/A')}")
        logger.info(f"   - Words: {chunk['word_count']}")
        logger.info(f"   - Preview: {chunk['text'][:100]}...")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ PREPROCESSING COMPLETE!")
    logger.info("=" * 80)
    
    logger.info(f"   Run: python scripts/build_index.py\n")


if __name__ == "__main__":
    main()

    





