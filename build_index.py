"""
build_index.py — Offline index builder for fast cold start.

Run this ONCE (or after adding new documents) to pre-compute:
  1. FAISS vector index (from fastembed BGE-small embeddings)
  2. BM25 keyword index
  3. Chunk metadata

Usage:
    python build_index.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Only import heavy libraries during BUILD, not at runtime
from fastembed import TextEmbedding
import faiss

# Reuse existing document processor (no heavy deps)
from legal_rag_system import BankingDocumentProcessor

INDEX_DIR = Path("./search_index")

def build():
    t0 = time.time()
    print("=" * 60)
    print("BUILDING SEARCH INDEX (offline, run once)")
    print("=" * 60)

    # Step 1: Load and process documents
    print("\n[1/4] Loading documents...")
    processor = BankingDocumentProcessor()
    documents = processor.load_documents()
    documents = processor.enrich_metadata(documents)
    chunks = processor.smart_chunk(documents)
    print(f"  ✓ {len(chunks)} chunks from {len(documents)} documents")

    # Step 2: Compute embeddings with fastembed (ONNX, fast)
    print("\n[2/4] Computing embeddings (BAAI/bge-small-en-v1.5)...")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = list(model.embed(texts))
    embeddings_np = np.array(embeddings, dtype=np.float32)
    print(f"  ✓ {embeddings_np.shape[0]} embeddings, {embeddings_np.shape[1]} dimensions")

    # Step 3: Build FAISS index (Inner Product for cosine similarity with normalized vectors)
    print("\n[3/4] Building FAISS index...")
    dim = embeddings_np.shape[1]
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_np)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_np)
    print(f"  ✓ FAISS index: {index.ntotal} vectors, {dim} dimensions")

    # Step 4: Build BM25 index
    print("\n[4/4] Building BM25 index...")
    from rank_bm25 import BM25Okapi
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    print(f"  ✓ BM25 index: {len(tokenized)} documents")

    # Save everything
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_DIR / "faiss_index.bin"))

    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    with open(INDEX_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(INDEX_DIR / "texts.pkl", "wb") as f:
        pickle.dump(texts, f)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"✓ Index built in {elapsed:.1f}s")
    print(f"  Saved to: {INDEX_DIR.resolve()}")
    print(f"  Files: faiss_index.bin, chunks.pkl, bm25_index.pkl, texts.pkl")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    build()
