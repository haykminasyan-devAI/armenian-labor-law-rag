# Prompt templates
"""
Prompt templates for RAG-based QA.
"""

def get_qa_prompt_template() -> str:
    """
    Get the QA prompt template in Armenian.
    
    Returns:
        Formatted prompt template string
    """
    return """Դու իրավական օգնական ես։

ՀԱՄԱՏԵՔՍՏ՝
{context}

ՀԱՐՑ՝
{question}

ՊԱՐՏԱԴԻՐ՝ Պատասխանիր հայերեն լեզվով, օգտագործելով միայն վերևի համատեքստից տեղեկություն։ Նշիր հոդվածների համարները։

ՊԱՏԱՍԽԱՆ (հայերեն)՝"""


def get_qa_prompt_template_english() -> str:
    """
    Get the QA prompt template in English (alternative).
    
    Returns:
        Formatted prompt template string
    """
    return """You are an expert on Armenian Labor Law. Answer the question based ONLY on the provided context from the law.

Context from Armenian Labor Law:
{context}

Question: {question}

Instructions:
- Answer in Armenian if the question is in Armenian
- Be precise and cite article numbers when possible
- If the context doesn't contain the answer, say "I cannot find this information in the provided articles"
- Do not make up information

Answer:"""


def get_summarization_prompt_template() -> str:
    """Get the summarization prompt template."""
    return """Summarize the following article from Armenian Labor Law:

Article:
{context}

Provide a clear, concise summary in Armenian (or English if requested).

Summary:"""


def format_context(retrieved_chunks: list) -> str:
    """
    Format retrieved chunks into context string.
    
    Args:
        retrieved_chunks: List of chunk dictionaries with 'text' field
        
    Returns:
        Formatted context string
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        article_num = chunk.get('article_number', 'N/A')
        text = chunk.get('text', '')
        context_parts.append(f"[Article {article_num}]\n{text}")
    
    return "\n\n".join(context_parts)