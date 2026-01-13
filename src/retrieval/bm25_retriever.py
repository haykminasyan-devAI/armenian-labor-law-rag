from typing import List, Dict
import logging
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path

from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)

class BM25Retriever(BaseRetriever):
    def __init__(self, chunks: List[Dict]):
        super().__init__(chunks)
        self.bm25 = None
        self.tokenized_corpus = None

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def build_index(self):
        logger.info("Building BM25 index...")

        self.tokenized_corpus = [
            self._tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.is_indexed = True
        logger.info(f"BM25 index built with {len(self.chunks)} documents")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index() first.")

        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True)[:top_k]

        results = []
        for idx in top_indices:
            result = self.chunks[idx].copy()
            result['score'] = float(scores[idx])
            result['retrieval_method'] = 'bm25'
            results.append(result)
        
        logger.debug(f"BM25 search for '{query}': found {len(results)} results")
        return results

    def save_index(self, path: str):
        """Save BM25 index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'tokenized_corpus': self.tokenized_corpus
            }, f)
        
        logger.info(f"BM25 index saved to {path}")
    
    def load_index(self, path: str):
        """Load BM25 index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.bm25 = data['bm25']
        self.tokenized_corpus = data['tokenized_corpus']
        self.is_indexed = True
        
        logger.info(f"BM25 index loaded from {path}")




