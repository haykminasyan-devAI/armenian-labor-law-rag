import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DocumentCleaner:
    def __init__(self, remove_extra_whitespace: bool = True,
                     remove_page_numbers: bool = True,
                     normalize_unicode: bool = True):
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_page_numbers = remove_page_numbers
        self.normalize_unicode = normalize_unicode

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Normalize unicode if requested
        if self.normalize_unicode:
            import unicodedata
            text = unicodedata.normalize('NFKC', text)

        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Remove lines that are only page numbers
            if self.remove_page_numbers and line.strip().isdigit():
                continue
            
            # Remove extra whitespace
            if self.remove_extra_whitespace:
                line = re.sub(r'\s+', ' ', line)
                line = line.strip()
            
            # Keep non-empty lines
            if line:
                cleaned_lines.append(line)
        
        # Rejoin with single newlines
        text = '\n'.join(cleaned_lines)
        
        return text
    def clean_document(self, document: Dict) -> Dict:
        """
        Clean a document dictionary.
        
        Args:
            document: Dictionary containing document metadata and text
            
        Returns:
            Cleaned document dictionary
        """
        cleaned_doc = document.copy()
        
        if 'text' in cleaned_doc:
            cleaned_doc['text'] = self.clean_text(cleaned_doc['text'])
        
        if 'title' in cleaned_doc:
            cleaned_doc['title'] = self.clean_text(cleaned_doc['title'])
            
        return cleaned_doc
    
    def clean_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Clean a list of documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of cleaned documents
        """
        logger.info(f"Cleaning {len(documents)} documents...")
        cleaned = [self.clean_document(doc) for doc in documents]
        logger.info("Cleaning complete.")
        return cleaned

                     