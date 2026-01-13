# RAG pipeline
"""
Complete RAG pipeline: Retrieval + Generation.
"""

import logging
from typing import List, Dict, Optional
from .prompt_templates import get_qa_prompt_template, format_context
from .generator import LLMGenerator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline."""
    
    def __init__(self, retriever, generator: Optional[LLMGenerator] = None):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Any retriever (BM25, Dense, or Hybrid)
            generator: LLM generator (creates one if None)
        """
        self.retriever = retriever
        self.generator = generator or LLMGenerator()
        
        logger.info(f"Initialized RAG pipeline with {retriever.__class__.__name__}")
    
    def answer_question(self, 
                       question: str, 
                       top_k: int = 3,
                       return_context: bool = False) -> Dict:
        """
        Answer a question using RAG.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            return_context: Whether to return retrieved context
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"RAG Query: '{question}'")
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retriever.search(question, top_k=top_k)
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'retrieved_chunks': [],
                'scores': []
            }
        
        # Step 2: Format context
        context = format_context(retrieved_chunks)
        
        # Step 3: Build prompt
        prompt_template = get_qa_prompt_template()
        prompt = prompt_template.format(context=context, question=question)
        
        # Step 4: Generate answer
        logger.info("Generating answer with LLM...")
        answer = self.generator.generate(prompt)
        
        # Step 5: Prepare response
        result = {
            'question': question,
            'answer': answer,
            'num_chunks_used': len(retrieved_chunks),
            'retrieval_method': retrieved_chunks[0].get('retrieval_method', 'unknown')
        }
        
        if return_context:
            result['retrieved_chunks'] = retrieved_chunks
            result['scores'] = [c['score'] for c in retrieved_chunks]
            result['article_numbers'] = [c.get('article_number', 'N/A') for c in retrieved_chunks]
        
        logger.info(f"Answer generated successfully")
        return result
    
    def batch_answer(self, questions: List[str], top_k: int = 3) -> List[Dict]:
        """
        Answer multiple questions.
        
        Args:
            questions: List of questions
            top_k: Number of chunks to retrieve per question
            
        Returns:
            List of answer dictionaries
        """
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            result = self.answer_question(question, top_k=top_k)
            results.append(result)
        return results