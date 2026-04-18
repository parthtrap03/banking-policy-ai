"""
Banking Policy Intelligence Platform - Hybrid Retrieval Engine

Combines three retrieval strategies for maximum accuracy:
1. Dense Retrieval (ChromaDB vector search) — semantic understanding
2. Sparse Retrieval (BM25 keyword search) — exact term matching
3. Cross-Encoder Reranking — precision filtering

Uses Reciprocal Rank Fusion (RRF) to merge dense + sparse results.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    print("⚠ rank-bm25 not installed. BM25 search disabled. Run: pip install rank-bm25")

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None
    print("⚠ sentence-transformers not installed. Reranking disabled.")


# =============================================================================
# TEXT PREPROCESSING FOR BM25
# =============================================================================

def preprocess_text(text: str) -> List[str]:
    """Tokenize and clean text for BM25 indexing"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s\.\-]', ' ', text)
    tokens = text.split()
    # Remove very short tokens but keep important ones like "rs" (rupees)
    tokens = [t for t in tokens if len(t) > 1 or t in ('rs', 'rbi', 'upi', 'kyc')]
    return tokens


# =============================================================================
# RECIPROCAL RANK FUSION
# =============================================================================

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Merge multiple ranked result lists using RRF.
    
    Each ranked_list is a list of (doc_index, score) tuples.
    RRF score = sum(1 / (k + rank)) across all lists.
    
    Args:
        ranked_lists: List of ranked result lists
        k: RRF constant (default 60, standard in literature)
    
    Returns:
        Fused ranking as list of (doc_index, rrf_score) sorted by score desc
    """
    rrf_scores: Dict[int, float] = {}

    for ranked_list in ranked_lists:
        for rank, (doc_idx, _score) in enumerate(ranked_list, start=1):
            if doc_idx not in rrf_scores:
                rrf_scores[doc_idx] = 0.0
            rrf_scores[doc_idx] += 1.0 / (k + rank)

    # Sort by RRF score descending
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


# =============================================================================
# HYBRID RETRIEVER
# =============================================================================

class HybridRetriever:
    """
    Combines vector search + BM25 keyword search with cross-encoder reranking.
    
    Pipeline:
    1. Vector search (ChromaDB) → top initial_k results
    2. BM25 keyword search → top initial_k results
    3. Reciprocal Rank Fusion merges both lists
    4. Cross-encoder reranks top candidates → final top k results
    """

    def __init__(
        self,
        vectordb,
        chunks: List[Document],
        reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        enable_bm25: bool = True,
        enable_reranker: bool = True,
    ):
        """
        Args:
            vectordb: ChromaDB vector store instance
            chunks: List of Document objects (used to build BM25 index)
            reranker_model: HuggingFace cross-encoder model name
            enable_bm25: Whether to enable BM25 keyword search
            enable_reranker: Whether to enable cross-encoder reranking
        """
        self.vectordb = vectordb
        self.chunks = chunks
        self.enable_bm25 = enable_bm25 and BM25Okapi is not None
        self.enable_reranker = enable_reranker and CrossEncoder is not None

        # Build BM25 index
        if self.enable_bm25:
            print("⟳ Building BM25 keyword index...")
            tokenized_corpus = [preprocess_text(doc.page_content) for doc in chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"✓ BM25 index built ({len(chunks)} chunks)")
        else:
            self.bm25 = None

        # Load cross-encoder reranker
        if self.enable_reranker:
            print("⟳ Loading cross-encoder reranker...")
            self.reranker = CrossEncoder(reranker_model)
            print(f"✓ Reranker loaded ({reranker_model})")
        else:
            self.reranker = None

    def _vector_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Dense retrieval via ChromaDB"""
        results = self.vectordb.similarity_search_with_relevance_scores(query, k=k)

        ranked = []
        for doc, score in results:
            # Find matching chunk index
            chunk_idx = doc.metadata.get('chunk_index', -1)
            if chunk_idx == -1:
                # Fallback: match by content
                for i, chunk in enumerate(self.chunks):
                    if chunk.page_content == doc.page_content:
                        chunk_idx = i
                        break
            if chunk_idx >= 0:
                ranked.append((chunk_idx, float(score)))

        return ranked

    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Sparse retrieval via BM25"""
        if not self.bm25:
            return []

        tokenized_query = preprocess_text(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        ranked = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

        return ranked

    def _rerank(self, query: str, candidates: List[Document], top_k: int) -> List[Tuple[Document, float]]:
        """Cross-encoder reranking for precision"""
        if not self.reranker or not candidates:
            return [(doc, 0.5) for doc in candidates[:top_k]]

        # Create query-document pairs for cross-encoder
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # Sort by reranker score
        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]

    def retrieve(
        self,
        query: str,
        k: int = 5,
        initial_k: int = 20,
    ) -> List[Dict]:
        """
        Full hybrid retrieval pipeline.
        
        Args:
            query: User's question
            k: Number of final results to return
            initial_k: Number of candidates to fetch from each retriever
        
        Returns:
            List of dicts with keys: document, content, metadata, relevance_score
        """
        # Step 1: Get candidates from both retrievers
        vector_results = self._vector_search(query, initial_k)

        if self.enable_bm25:
            bm25_results = self._bm25_search(query, initial_k)
            # Step 2: Fuse with RRF
            fused_ranking = reciprocal_rank_fusion([vector_results, bm25_results])
        else:
            # Vector-only fallback
            fused_ranking = vector_results

        # Step 3: Get candidate documents
        candidate_indices = [idx for idx, _ in fused_ranking[:initial_k]]
        candidates = [self.chunks[i] for i in candidate_indices if i < len(self.chunks)]

        # Step 4: Rerank for precision
        reranked = self._rerank(query, candidates, k)

        # Step 5: Format results
        results = []
        for doc, score in reranked:
            results.append({
                'document': doc,
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': float(score),
                'document_type': doc.metadata.get('document_type', 'Unknown'),
                'section': doc.metadata.get('section', 'General'),
                'source_file': doc.metadata.get('source', 'Unknown'),
                'issuing_authority': doc.metadata.get('issuing_authority', 'Unknown'),
            })

        return results

    def get_retrieval_stats(self) -> Dict:
        """Return retrieval configuration for display"""
        return {
            'bm25_enabled': self.enable_bm25,
            'reranker_enabled': self.enable_reranker,
            'total_chunks': len(self.chunks),
            'retrieval_method': 'Hybrid (Vector + BM25 + Reranker)' if self.enable_bm25 and self.enable_reranker
                               else 'Hybrid (Vector + BM25)' if self.enable_bm25
                               else 'Vector Only',
        }
