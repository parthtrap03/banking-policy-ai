"""Measure TRUE cold start — loading pre-built DB only (no rebuild)."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')

t0 = time.time()

from legal_rag_system import BankingPolicyRAG, HybridStoreManager, BankingDocumentProcessor

# Step 1: Load documents + chunk (for BM25)
t1 = time.time()
processor = BankingDocumentProcessor()
documents = processor.load_documents()
documents = processor.enrich_metadata(documents)
chunks = processor.smart_chunk(documents)
print(f"  Docs + chunks: {time.time()-t1:.1f}s")

# Step 2: Load pre-built vector store
t2 = time.time()
