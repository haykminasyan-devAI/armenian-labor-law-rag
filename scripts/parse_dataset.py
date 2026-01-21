"""
Parse the uploaded dataset from .txt to JSON format for evaluation.
"""

import sys
from pathlib import Path
import json
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_armenian_dataset(txt_path):
    """
    Parse dataset with format:
    N. Հարց
    [question text]
    Պատասխան
    [answer text]
    Հղումներ՝ Հոդված X, Հոդված Y
    """
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by question numbers
    # Pattern: number followed by ". Հարց"
    question_blocks = re.split(r'\n\d+\.\s+Հարց\n', content)
    question_blocks = [b.strip() for b in question_blocks if b.strip()]
    
    dataset = []
    
    for i, block in enumerate(question_blocks, 1):
        # Split into question and answer parts
        parts = block.split('\nՊատասխան\n')
        if len(parts) != 2:
            print(f"Warning: Could not parse block {i}")
            continue
        
        question_part = parts[0].strip()
        answer_part = parts[1].strip()
        
        # Extract question (everything before "Պատասխան")
        question = question_part
        
        # Extract reference answer (between "Պատասխան" and "Հղումներ")
        # Split answer_part to get the actual answer text
        answer_text_match = re.split(r'\nՀղումներ[՝`]\s*', answer_part)
        reference_answer = answer_text_match[0].strip() if answer_text_match else ""
        
        # Extract gold articles from references
        # Pattern: Հղումներ՝ Հոդված 13, Հոդված 14, etc.
        # Handle both ՝ and ` apostrophes
        references_match = re.search(r'Հղումներ[՝`]\s*(.+?)(?:\n|$)', answer_part)
        
        gold_articles = []
        if references_match:
            refs_text = references_match.group(1)
            # Extract all article numbers
            article_numbers = re.findall(r'[Հհ]ոդված\s+(\d+)', refs_text)
            gold_articles = [int(num) for num in article_numbers]
        
        if not gold_articles:
            print(f"Warning: No gold articles found for question {i}: {question[:50]}...")
            continue
        
        dataset.append({
            "id": i,
            "question": question,
            "gold_articles": gold_articles,
            "reference_answer": reference_answer  # Add reference answer
        })
    
    return dataset


def main():
    """Parse and save dataset."""
    
    # Paths
    txt_file = project_root / "OrinAi_1 (1).txt"
    output_file = project_root / "data" / "evaluation" / "test_set.json"
    
    print("="*60)
    print("Parsing Armenian Labor Law Test Dataset")
    print("="*60)
    
    print(f"\nReading from: {txt_file}")
    
    # Parse dataset
    dataset = parse_armenian_dataset(txt_file)
    
    print(f"\n✅ Parsed {len(dataset)} questions")
    
    # Show sample
    print("\nSample entries:")
    for entry in dataset[:3]:
        print(f"\nID {entry['id']}:")
        print(f"  Question: {entry['question'][:60]}...")
        print(f"  Gold Articles: {entry['gold_articles']}")
    
    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Dataset saved to: {output_file}")
    print(f"\nTotal questions: {len(dataset)}")
    print(f"Total gold articles: {sum(len(q['gold_articles']) for q in dataset)}")
    print(f"Avg articles per question: {sum(len(q['gold_articles']) for q in dataset) / len(dataset):.1f}")
    
    print("\n" + "="*60)
    print("✅ Dataset ready for evaluation!")
    print("="*60)


if __name__ == "__main__":
    main()
