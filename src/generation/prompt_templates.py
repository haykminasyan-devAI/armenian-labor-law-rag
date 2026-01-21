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
    return """Դու իրավական օգնական ես, մասնագիտացված Հայաստանի Աշխատանքային օրենսգրքում։

ՀԱՄԱՏԵՔՍՏ՝ Աշխատանքային օրենսգրքից
{context}

ՀԱՐՑ՝
{question}

ՀՐԱՀԱՆԳՆԵՐ՝
• Պատասխանը գրիր ՄԻԱՅՆ ՀԱՅԵՐԵՆ լեզվով
• Օգտագործիր ՄԻԱՅՆ վերևի համատեքստից տեղեկություն
• Լինիր ՄԱՆՐԱՄԱՍՆ և ՀԱՄԱՊԱՐՓԱԿ - բացատրիր ամբողջ համատեքստը

• Բաժանիր պատասխանը ԲԱԺԻՆՆԵՐՈՎ՝
  1. Ընդհանուր բացատրություն (2-3 պարբերություն)
  2. Կոնկրետ դետալներ յուրաքանչյուր հոդվածից
  3. Օրինակներ և հետևանքներ (եթե հնարավոր է)
• ՊԱՐՏԱԴԻՐ նշիր հոդվածի համարը յուրաքանչյուր տեղեկության համար (օրինակ՝ «Համաձայն Հոդված 145-ի...»)
• Եթե համատեքստում մի քանի հոդվածներ կան, բացատրիր ԲՈԼՈՐԸ
• Ավելացրու հղումներ հոդվածներին և դրանց մասերին
ՊԱՐՏԱԴԻՐ ավարտիր պատասխանը ամբողջական նախադասությամբ:
եղիր հնարավորինս մանրակրկիտ և տվիր ամբողջական պատասխան։

ՊԱՏԱՍԽԱՆ (հայերեն, մանրամասն)՝"""


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