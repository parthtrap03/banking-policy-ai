"""
Banking Policy Intelligence Platform - Interactive CLI Query Tool
Run queries against your banking policy documents with conversational memory.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class InteractiveQueryTool:
    """Interactive CLI interface for querying banking policy RAG"""

    def __init__(self, persist_dir: str = './chroma_legal_db'):
        self.persist_dir = persist_dir
        self.rag = None
        self._initialize()

    def _initialize(self):
        """Load system and initialize RAG"""
        try:
            from legal_rag_system import BankingPolicyRAG, HybridStoreManager, BankingDocumentProcessor
        except ImportError:
            print("❌ Error: legal_rag_system.py not found")
            sys.exit(1)

        if not Path(self.persist_dir).exists():
            print("✗ Vector database not found at:", self.persist_dir)
            print("\nPlease run setup first:")
            print("  python legal_rag_system.py")
            sys.exit(1)

        print("⟳ Loading Banking Policy Intelligence Platform...")

        # Load documents for hybrid retrieval
        processor = BankingDocumentProcessor()
        documents = processor.load_documents()
        documents = processor.enrich_metadata(documents)
        chunks = processor.smart_chunk(documents)

        # Load vector store
        store_manager = HybridStoreManager(persist_dir=self.persist_dir)
        vectordb = store_manager.load_vector_store()

        # Initialize RAG
        self.rag = BankingPolicyRAG(vectordb=vectordb, chunks=chunks)
        print("✓ System ready\n")

    def print_banner(self):
        """Print welcome banner"""
        print("\n" + "=" * 75)
        print("BANKING POLICY INTELLIGENCE - INTERACTIVE MODE".center(75))
        print("=" * 75)
        print("\nAsk questions about banking regulations, RBI guidelines,")
        print("UPI policies, KYC norms, data protection, and more.")
        print("\nType 'help' for commands, 'quit' to exit.\n")

    def print_help(self):
        """Print help message"""
        help_text = """
COMMANDS:
  help              Show this help message
  history           View conversation history
  clear             Clear conversation memory
  settings          Show system settings
  suggest           Show suggested questions
  export [file]     Export conversation to JSON file
  quit              Exit the program

TIPS:
  - The system remembers your previous questions for context
  - Ask follow-up questions like "What about the penalties for that?"
  - You'll see the query intent, confidence level, and sources for each answer
  
  Examples:
    * "What is the UPI transaction limit?"
    * "How do I file a complaint about a failed UPI transaction?"
    * "What's the difference between minimum KYC and full KYC?"
    * "What are the penalties under DPDP Act?"
        """
        print(help_text)

    def print_settings(self):
        """Show current settings"""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        stats = self.rag.retriever.get_retrieval_stats() if self.rag.retriever else {}

        print("\nSYSTEM SETTINGS:")
        print(f"  LLM Model: Groq ({os.getenv('LLM_MODEL', 'llama-3.1-8b-instant')})")
        print(f"  Embedding: {os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5')}")
        print(f"  Retrieval: {stats.get('retrieval_method', 'Unknown')}")
        print(f"  Total Chunks: {stats.get('total_chunks', 'Unknown')}")
        print(f"  BM25 Enabled: {stats.get('bm25_enabled', False)}")
        print(f"  Reranker Enabled: {stats.get('reranker_enabled', False)}")
        print(f"  Temperature: {os.getenv('LLM_TEMPERATURE', '0.2')}")
        print(f"  Retrieval K: {os.getenv('RETRIEVAL_K', '5')}")

    def print_suggestions(self):
        """Show suggested questions"""
        suggested = self.rag.get_suggested_questions()
        print("\n💡 SUGGESTED QUESTIONS:\n")
        for category, questions in suggested.items():
            print(f"  📌 {category}")
            for q in questions[:2]:
                print(f"     → {q}")
            print()

    def process_command(self, user_input: str):
        """
        Process special commands.
        Returns: True if should continue, False if should exit, None if not a command
        """
        command = user_input.lower().strip()

        if command == 'help':
            self.print_help()
            return True
        elif command in ('quit', 'exit'):
            print("\nGoodbye! 👋")
            return False
        elif command == 'history':
            self._print_history()
            return True
        elif command == 'clear':
            self.rag.clear_history()
            print("✓ Conversation memory cleared")
            return True
        elif command == 'settings':
            self.print_settings()
            return True
        elif command == 'suggest':
            self.print_suggestions()
            return True
        elif command.startswith('export'):
            parts = command.split()
            filename = 'banking_queries.json' if len(parts) < 2 else parts[1]
            self._export_history(filename)
            return True
        elif command.strip() == '':
            return True
        else:
            return None

    def _print_history(self):
        """Print conversation history"""
        history = self.rag.chat_history
        if not history:
            print("\nNo conversation history")
            return

        print(f"\n{'=' * 75}")
        print(f"CONVERSATION HISTORY ({len(history)} exchanges)")
        print(f"{'=' * 75}\n")

        for i, entry in enumerate(history, 1):
            print(f"{i}. [{entry.get('intent', 'GENERAL')}] {entry['question'][:60]}")
            print(f"   Answer: {entry['answer'][:100]}...")
            print()

    def _export_history(self, filename: str):
        """Export conversation to JSON"""
        history = self.rag.chat_history
        if not history:
            print("No conversation to export")
            return

        data = {
            'export_date': datetime.now().isoformat(),
            'total_exchanges': len(history),
            'conversation': history,
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Exported {len(history)} exchanges to {filename}")

    def run_interactive_mode(self):
        """Main interactive loop"""
        self.print_banner()

        try:
            while True:
                try:
                    user_input = input("\n🏦 Your question: ").strip()

                    if not user_input:
                        continue

                    # Check if it's a command
                    should_continue = self.process_command(user_input)
                    if should_continue is False:
                        break
                    elif should_continue is True:
                        continue

                    # It's a query
                    print("\n🔍 Searching policy documents...\n")
                    result = self.rag.query(user_input, k=5)

                    # Print answer
                    print("-" * 75)
                    print(f"📋 Intent: {result['intent']} | "
                          f"🎯 Confidence: {result['confidence']} | "
                          f"⚡ {result['response_time_ms']}ms")
                    print("-" * 75)
                    print(f"\n{result['answer']}\n")
                    print("-" * 75)

                    # Print sources
                    print(f"\n📄 SOURCES ({result['num_retrieved']} documents):")
                    for i, src in enumerate(result['sources'][:3], 1):
                        print(f"\n  [{i}] {src['document_type']}")
                        print(f"      Section: {src['section']}")
                        print(f"      Authority: {src['issuing_authority']}")
                        print(f"      Preview: {src['content'][:150]}...")

                except KeyboardInterrupt:
                    print("\n\nInterrupted. Goodbye! 👋")
                    break

        except Exception as e:
            print(f"\nError: {e}")
            raise


def main():
    """Entry point"""
    try:
        tool = InteractiveQueryTool()
        tool.run_interactive_mode()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
