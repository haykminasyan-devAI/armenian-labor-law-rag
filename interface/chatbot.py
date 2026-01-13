#!/usr/bin/env python3
"""
Simple CLI Chatbot for Armenian Labor Law Q&A
Uses the RAG pipeline with BM25 or Dense retrieval.
"""

import sys
import json
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.rag_pipeline import RAGPipeline
from src.generation.generator import LLMGenerator

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings/errors
    format='%(levelname)s: %(message)s'
)


class LaborLawChatbot:
    """Interactive chatbot for Armenian Labor Law Q&A."""
    
    def __init__(self, retrieval_method='bm25'):
        """
        Initialize chatbot.
        
        Args:
            retrieval_method: 'bm25', 'dense', or 'hybrid'
        """
        print("🤖 Initializing Armenian Labor Law Chatbot...")
        print(f"📇 Loading {retrieval_method.upper()} retriever...")
        
        # Load chunks
        chunks_file = project_root / "data" / "chunks" / "labor_law_chunks.json"
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Initialize retriever
        if retrieval_method == 'bm25':
            self.retriever = BM25Retriever(chunks)
            index_path = project_root / "indices" / "bm25" / "bm25_index.pkl"
            self.retriever.load_index(str(index_path))
        elif retrieval_method == 'dense':
            self.retriever = DenseRetriever(chunks)
            index_path = project_root / "indices" / "dense"
            self.retriever.load_index(str(index_path))
        else:
            raise ValueError(f"Unknown retrieval method: {retrieval_method}")
        
        print("✅ Retriever loaded")
        
        # Initialize generator
        print("🤖 Connecting to NVIDIA Llama 3.1-70B...")
        api_key = "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"
        self.generator = LLMGenerator(
            model_name="meta/llama-3.1-70b-instruct",
            provider="nvidia",
            api_key=api_key,
            max_tokens=1000,
            temperature=0.1
        )
        print("✅ Generator ready")
        
        # Initialize RAG pipeline
        self.rag = RAGPipeline(retriever=self.retriever, generator=self.generator)
        self.retrieval_method = retrieval_method
        
        print("\n" + "=" * 80)
        print("✅ CHATBOT READY!")
        print("=" * 80)
    
    def chat(self):
        """Start interactive chat loop."""
        print("\n🇦🇲 Բարև ձեզ! Ես Հայաստանի Աշխատանքային օրենսգրքի վիրտուալ օգնականն եմ։")
        print("📚 Հարցրեք ինձ աշխատանքային իրավունքի մասին։")
        print("\nՀրահանգներ՝")
        print("  • Գրեք ձեր հարցը հայերեն և սեղմեք Enter")
        print("  • Գրեք 'exit' կամ 'quit' ելքի համար")
        print("  • Գրեք 'help' օգնության համար")
        print("=" * 80)
        
        while True:
            try:
                # Get user input
                print("\n💬 Ձեր հարցը: ", end='')
                question = input().strip()
                
                # Handle commands
                if question.lower() in ['exit', 'quit', 'ելք', 'դուրս']:
                    print("\n👋 Ցտեսություն!")
                    break
                
                if question.lower() in ['help', 'օգնություն']:
                    self.show_help()
                    continue
                
                if not question:
                    continue
                
                # Process question
                print("\n🔍 Փնտրում եմ համապատասխան հոդվածներ...")
                result = self.rag.answer_question(question, top_k=3, return_context=True)
                
                # Display results
                print(f"\n📊 Գտնված հոդվածներ: {result['article_numbers']}")
                print(f"📊 Վստահության միավորներ: {[f'{s:.2f}' for s in result['scores']]}")
                print(f"📊 Որոնման մեթոդ: {result['retrieval_method']}")
                
                print("\n" + "=" * 80)
                print("💡 ՊԱՏԱՍԽԱՆ:")
                print("=" * 80)
                print(result['answer'])
                print("=" * 80)
                
            except KeyboardInterrupt:
                print("\n\n👋 Ցտեսություն!")
                break
            except Exception as e:
                print(f"\n❌ Սխալ: {e}")
                print("Փորձեք նորից:")
    
    def show_help(self):
        """Show help message."""
        print("\n" + "=" * 80)
        print("📖 ՕԳՆՈՒԹՅՈՒՆ")
        print("=" * 80)
        print("\nՕրինակ հարցեր՝")
        print("  • Որո՞նք են նվազագույն աշխատավարձի կանոնները։")
        print("  • Քանի՞ արձակուրդային օր կա։")
        print("  • Ինչպե՞ս է սահմանվում գործուղման օրապահիկը։")
        print("  • Ի՞նչ իրավունքներ ունի աշխատողը։")
        print("\nՀրահանգներ՝")
        print(f"  • Ընթացիկ մեթոդ: {self.retrieval_method.upper()}")
        print("  • Պատասխանները հիմնված են Աշխատանքային օրենսգրքի վրա")
        print("  • Յուրաքանչյուր պատասխան ներառում է հոդվածի հղումներ")
        print("=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Armenian Labor Law Q&A Chatbot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interface/chatbot.py                    # Use BM25 (default)
  python interface/chatbot.py --method dense     # Use Dense retrieval
  python interface/chatbot.py --method hybrid    # Use Hybrid retrieval
        """
    )
    parser.add_argument(
        '--method',
        choices=['bm25', 'dense', 'hybrid'],
        default='bm25',
        help='Retrieval method to use (default: bm25)'
    )
    
    args = parser.parse_args()
    
    # Create and start chatbot
    try:
        chatbot = LaborLawChatbot(retrieval_method=args.method)
        chatbot.chat()
    except KeyboardInterrupt:
        print("\n\n👋 Ցտեսություն!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
