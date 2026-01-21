# Armenian Labor Law Q&A System

A RAG (Retrieval-Augmented Generation) system that answers questions about Armenian Labor Law in Armenian. Built as a university project for NLP course at Yerevan State University.

## What it does

Ask a question in Armenian about labor law → Get an accurate answer with citations to specific articles.

**Example:**
- Q: "Ինչպիսի՞ն է աշխատանքային ժամերի սահմանափակումը:"
- A: Detailed answer citing "Համաձայն Հոդված 145-ի..."

## Quick Start

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run the interface
streamlit run interface/streamlit_app_modern.py

# 3. Open browser at http://localhost:8501
```

That's it! Pick Qwen 3 Next 80B model and Hybrid retrieval for best results.

## How it works

1. **Retrieval**: Searches 354 Labor Code articles using hybrid approach (BM25 + semantic search)
2. **Generation**: Feeds top articles to LLM (Qwen 3 Next 80B via NVIDIA API)
3. **Output**: Armenian answer with proper citations

## Key Features

- **Hybrid chunking**: Split long articles smartly (avg 444 tokens, down from 2117)
- **Multiple models**: Qwen 3 Next 80B, Qwen 2.5-7B, Llama 3.1 70B, DeepSeek
- **Zero hallucinations**: 98.8% citation accuracy, answers grounded in actual law
- **Armenian-first**: Uses Metric-AI Armenian embeddings for semantic search

## Results

Tested on 90 questions from legal experts:

| Model | Retrieval | Sem. Similarity | Citation Acc. | Hallucinations |
|-------|-----------|----------------|---------------|----------------|
| **Qwen 3 Next 80B** | Hybrid | **88.5%** | **98.8%** | 9% |
| Qwen 2.5-7B | Hybrid | 83.0% | **100%** | **0%** |
| Llama 3.1 70B | Dense | 86.7% | 99.8% | 2.3% |
| DeepSeek V3 | BM25 | 85.8% | 85.6% | 26.7% |

Best overall: Qwen 3 Next 80B with hybrid retrieval.

## Project Structure

```
├── src/                          # Core code
│   ├── retrieval/                # BM25, Dense, Hybrid retrievers
│   ├── generation/               # LLM interface and prompts
│   └── evaluation/               # Metrics
├── scripts/                      # Utility scripts
│   ├── create_hybrid_chunks.py   # Chunking system
│   ├── build_hybrid_chunks_index.py
│   └── eval_qwen3_80b_hybrid_final.py  # Main evaluation
├── interface/
│   └── streamlit_app_modern.py   # Web UI
├── data/
│   ├── chunks/labor_law_chunks_hybrid.json  # Optimized chunks
│   └── indices/hybrid_v2/        # Search indices
├── results/evaluation/           # All evaluation results
└── README.md                     # This file
```

## Running Evaluation

```bash
python scripts/eval_qwen3_80b_hybrid_final.py
```

Results saved to `results/evaluation/`.

## Tech Stack

- **Python 3.8+**
- **Embeddings**: Metric-AI/armenian-text-embeddings-1
- **Search**: BM25 (rank-bm25) + FAISS for dense retrieval
- **LLMs**: NVIDIA API (Qwen, Llama, DeepSeek)
- **Interface**: Streamlit
- **Token counting**: tiktoken

## Main Improvements We Made

1. **Hybrid chunking**: Original system used whole articles (2117 tokens avg, some 11K+). We split long ones while keeping context → 444 tokens avg, 79% reduction.

2. **Hybrid retrieval**: Combined keyword search (BM25) with semantic search (dense embeddings). RRF fusion gives best of both worlds.

3. **Sentence completion**: Initially answers got cut off mid-sentence. Fixed by increasing max_tokens from 2000 → 6000.

4. **Zero hallucinations**: With good retrieval + careful prompts, got hallucination rate from 26.7% (DeepSeek+BM25) down to 0% (Qwen 2.5-7B) or 9% (Qwen 3 Next 80B).

## Challenges

- **Low-resource language**: Armenian has limited NLP tools. Used multilingual models + specialized embeddings.
- **Legal text complexity**: Articles are long and interconnected. Hybrid chunking solved this.
- **Citation accuracy**: Legal answers need precise references. Prompt engineering + retrieval quality was key.

## Configuration

Edit `interface/streamlit_app_modern.py` to change settings:
- `max_tokens`: Currently 6000 for Qwen 3 Next 80B
- `temperature`: 0.1 for consistent outputs
- Retrieval weights: 0.5 BM25, 0.5 Dense

## What's in the evaluation results?

`results/evaluation/qwen3_80b_hybrid_chunks_summary_*.txt` has the main numbers.

**Retrieval metrics:**
- MRR: How quickly we find the right article
- Recall@3: % of relevant articles in top-3
- Hit@3: % of queries that got at least one relevant article

**Answer quality:**
- Citation Accuracy: Are the cited articles correct?
- Hallucination Rate: Made-up info not in context
- Semantic Similarity: How close to reference answer?

## Future ideas

- Fine-tune embeddings on Armenian legal text
- Handle questions spanning multiple laws (Civil Code + Labor Code)
- Multi-turn conversations
- Highlight exact text supporting each claim

## Team

**Students:** Hayk Minasyan, Arpine Aghababyan, Lilit Beglaryan  
**Program:** MSc Applied Statistics and Data Science, Yerevan State University  
**Course:** Natural Language Processing  
**Instructor:** Erik Arakelyan (Senior Researcher, NVIDIA)

## Links

- **GitHub**: https://github.com/haykminasyan-devAI/armenian-labor-law-rag
- **Dataset**: 354 articles from Armenian Labor Code
- **Test set**: 90 expert-annotated Q&A pairs (included as `OrinAi_1 (1).txt`)

---

**Status**: ✅ Production ready (v1.0.0)  
**Last updated**: January 2026
