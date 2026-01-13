from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class BaseRetriever(ABC):
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.indexed = False
        logger.info(f"Initialized {self.__class__.__name__} with {len(chunks)} chunks")

    @abstractmethod
    def build_index(self):
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        pass

        
    def save_index(self, path: str):
        """Save index to disk. Optional."""
        raise NotImplementedError("Subclass must implement save_index()")
    
    def load_index(self, path: str):
        """Load index from disk. Optional."""
        raise NotImplementedError("Subclass must implement load_index()")

