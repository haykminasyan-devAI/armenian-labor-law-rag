#!/usr/bin/env python3
"""
Extract text from PDF documents and save to processed data folder.
Handles Armenian Unicode characters properly.
"""

import sys
import os
from pathlib import Path
import pdfplumber
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def extract_text_from_pdf(pdf_path: str, output_path: str, metadata_path: str = None):
    """
    Extract text from PDF and save to text file.
    
    Args:
        pdf_path: Path to input PDF file
        output_path: Path to output text file
        metadata_path: Optional path to save metadata JSON
    """
    print(f"📖 Reading PDF: {pdf_path}")
    
    all_text = []
    metadata = {
        'source_file': pdf_path,
        'total_pages': 0,
        'total_chars': 0,
        'total_words': 0,
        'pages_info': []
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            metadata['total_pages'] = len(pdf.pages)
            print(f"📄 Total pages: {len(pdf.pages)}")
            
            for i, page in enumerate(pdf.pages, 1):
                # Extract text from page
                text = page.extract_text()
                
                if text:
                    all_text.append(text)
                    page_info = {
                        'page_num': i,
                        'chars': len(text),
                        'words': len(text.split())
                    }
                    metadata['pages_info'].append(page_info)
                    
                    if i % 10 == 0:
                        print(f"  Processed {i}/{len(pdf.pages)} pages...")
                else:
                    print(f"  ⚠️  Warning: Page {i} has no extractable text")
        
        # Combine all pages
        full_text = '\n\n'.join(all_text)
        
        # Calculate statistics
        metadata['total_chars'] = len(full_text)
        metadata['total_words'] = len(full_text.split())
        
        # Save text file
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        print(f"\n✅ Text extracted successfully!")
        print(f"📊 Statistics:")
        print(f"   - Pages: {metadata['total_pages']}")
        print(f"   - Characters: {metadata['total_chars']:,}")
        print(f"   - Words: {metadata['total_words']:,}")
        print(f"   - Output: {output_path}")
        
        # Save metadata if requested
        if metadata_path:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"   - Metadata: {metadata_path}")
        
        return full_text, metadata
        
    except Exception as e:
        print(f"❌ Error extracting PDF: {e}")
        raise


def main():
    """Main execution function."""
    # Paths
    pdf_path = project_root / "data" / "raw" / "labor_law.pdf"
    output_path = project_root / "data" / "processed" / "labor_law.txt"
    metadata_path = project_root / "data" / "processed" / "labor_law_metadata.json"
    
    # Check if PDF exists
    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found at {pdf_path}")
        print(f"   Please ensure your PDF is in the data/raw/ folder")
        sys.exit(1)
    
    # Extract text
    extract_text_from_pdf(
        str(pdf_path),
        str(output_path),
        str(metadata_path)
    )
    
   


if __name__ == "__main__":
    main()
