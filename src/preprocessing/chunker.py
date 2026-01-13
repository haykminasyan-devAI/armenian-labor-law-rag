import re
from typing import List, Dict, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    ARTICLE_BASED = "article_based"
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence"
    PARAGRAPH_BASED = "paragraph"

class ArmenianLegalChunker:
    def __init__(self,
    strategy: ChunkingStrategy = ChunkingStrategy.ARTICLE_BASED,
    chunk_size: int = 512,
    overlap: int = 50):

        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap


         # Armenian legal document patterns
        self.article_pattern = re.compile(r'Հոդված\s+(\d+)\.?\s*(.*?)(?=Հոդված\s+\d+|$)', 
                                         re.IGNORECASE | re.DOTALL)

    def chunk_by_article(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        chunks = []

        matches = self.article_pattern.finditer(text) 

        for match in matches:
            article_num = match.group(1)
            article_text = match.group(2).strip()
            if not article_text:
                continue

            chunk = {
                'chunk_id': len(chunks),
                'text': f"Հոդված {article_num}. {article_text}",
                'article_number': int(article_num),
                'char_count': len(article_text),
                'word_count': len(article_text.split()),
                'chunk_type': 'article'
            }
            
            if metadata:
                chunk['metadata'] = metadata
                
            chunks.append(chunk)
        
        # Fallback: if no articles found, use fixed-size chunking
        if not chunks:
            logger.warning("No articles found! Falling back to fixed-size chunking.")
            return self.chunk_by_fixed_size(text, metadata)
            
        logger.info(f"Extracted {len(chunks)} articles")
        return chunks


    def chunk_by_fixed_size(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunk = {
                'chunk_id': len(chunks),
                'text': chunk_text,
                'start_idx': start,
                'end_idx': min(end, len(text)),
                'char_count': len(chunk_text),
                'word_count': len(chunk_text.split()),
                'chunk_type': 'fixed_size'
            }
            
            if metadata:
                chunk['metadata'] = metadata
                
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
            
        return chunks

    def chunk_by_paragraph(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
       
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            chunk = {
                'chunk_id': len(chunks),
                'text': para,
                'char_count': len(para),
                'word_count': len(para.split()),
                'chunk_type': 'paragraph'
            }
            
            if metadata:
                chunk['metadata'] = metadata
                
            chunks.append(chunk)
            
        return chunks

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        if self.strategy == ChunkingStrategy.ARTICLE_BASED:
            return self.chunk_by_article(text, metadata)
        elif self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self.chunk_by_fixed_size(text, metadata)
        elif self.strategy == ChunkingStrategy.PARAGRAPH_BASED:
            return self.chunk_by_paragraph(text, metadata)
        else:
             logger.warning(f"Strategy {self.strategy} not implemented, using article-based")
             return self.chunk_by_article(text, metadata)
    
    def chunk_document(self, document: Dict) -> List[Dict]:
       
        text = document.get('text', '')
        metadata = {k: v for k, v in document.items() if k != 'text'}
        
        return self.chunk_text(text, metadata)

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        
        logger.info(f"Chunking {len(documents)} documents with strategy: {self.strategy.value}")
        
        all_chunks = []
        for doc_idx, doc in enumerate(documents):
            chunks = self.chunk_document(doc)
            
            # Add document index to each chunk
            for chunk in chunks:
                chunk['doc_id'] = doc_idx
                chunk['global_chunk_id'] = len(all_chunks)
                all_chunks.append(chunk)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks



