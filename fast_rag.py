"""
fast_rag.py — Lightweight RAG engine for instant cold start.

Loads pre-built FAISS + BM25 indexes (from build_index.py).
Uses fastembed (ONNX) for query encoding — no transformers import needed.
Keeps cross-encoder reranking for full accuracy.

Cold start: ~3-5s (vs ~50s with the full system)
"""

import os
import re
import time
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import faiss
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq

from banking_prompts import (
    PROMPT_REGISTRY,
    GENERAL_PROMPT,
    SUGGESTED_QUESTIONS,
    SIMPLIFY_INSTRUCTIONS,
)

INDEX_DIR = Path("./search_index")


class FastRAG:
    """
    Lightweight RAG engine:
    - FAISS vector search (pre-built index, loads in 0.1s)
    - BM25 keyword search (pre-built, loads in 0.1s)
    - fastembed query encoder (ONNX, loads in ~2s)
    - Cross-encoder reranker (optional, loads in ~3s)
    - Groq LLM for answer generation
    """

    def __init__(self, enable_reranker: bool = True):
        t0 = time.time()

        # Load pre-built indexes
        self.index = faiss.read_index(str(INDEX_DIR / "faiss_index.bin"))
        with open(INDEX_DIR / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        with open(INDEX_DIR / "bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(INDEX_DIR / "texts.pkl", "rb") as f:
            self.texts = pickle.load(f)

        print(f"✓ Loaded {self.index.ntotal} vectors + BM25 index ({time.time()-t0:.1f}s)")

        # Query encoder (fastembed, ONNX — lightweight)
        t1 = time.time()
        self.encoder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        print(f"✓ Query encoder ready ({time.time()-t1:.1f}s)")

        # Cross-encoder reranker (for accuracy)
        self.reranker = None
        if enable_reranker:
            t2 = time.time()
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                self.reranker = TextCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
                print(f"✓ Cross-encoder reranker ready ({time.time()-t2:.1f}s)")
            except Exception as e:
                print(f"⚠ Reranker unavailable: {e}")

        # LLM
        t3 = time.time()
        self.llm = ChatGroq(
            model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "600")),
        )
        print(f"✓ LLM ready ({time.time()-t3:.1f}s)")

        # Conversation memory
        self.chat_history: List[Dict] = []
        self.max_history = 5

        # Analytics (lazy import)
        self.analytics = None
        try:
            from analytics import AnalyticsEngine
            self.analytics = AnalyticsEngine()
        except Exception:
            pass

        print(f"\n>>> TOTAL LOAD TIME: {time.time()-t0:.1f}s <<<\n")

    # ----- Retrieval -----

    def _vector_search(self, query: str, k: int = 12) -> List[Dict]:
        """FAISS cosine similarity search."""
        query_emb = list(self.encoder.embed([query]))
        query_np = np.array(query_emb, dtype=np.float32)
        faiss.normalize_L2(query_np)
        scores, indices = self.index.search(query_np, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append({
                "index": int(idx),
                "score": float(score),
                "content": self.texts[idx],
                "chunk": self.chunks[idx],
            })
        return results

    def _bm25_search(self, query: str, k: int = 12) -> List[Dict]:
        """BM25 keyword search."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append({
                "index": int(idx),
                "score": float(scores[idx]),
                "content": self.texts[idx],
                "chunk": self.chunks[idx],
            })
        return results

    def _rerank(self, query: str, results: List[Dict], k: int = 5) -> List[Dict]:
        """Cross-encoder reranking for accuracy."""
        if not self.reranker or not results:
            return results[:k]

        passages = [r["content"] for r in results]
        scores = list(self.reranker.rerank(query, passages))

        for i, score_obj in enumerate(scores):
            if i < len(results):
                results[i]["rerank_score"] = float(score_obj.score) if hasattr(score_obj, 'score') else float(score_obj)

        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return results[:k]

    def _hybrid_retrieve(self, query: str, k: int = 5, initial_k: int = 12) -> List[Dict]:
        """Hybrid retrieval: Vector + BM25 + Rerank."""
        # Get candidates from both
        vector_results = self._vector_search(query, k=initial_k)
        bm25_results = self._bm25_search(query, k=initial_k)

        # Reciprocal Rank Fusion
        rrf_scores = {}
        rrf_k = 60
        for rank, r in enumerate(vector_results):
            idx = r["index"]
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)
        for rank, r in enumerate(bm25_results):
            idx = r["index"]
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

        # Merge and sort
        all_results = {}
        for r in vector_results + bm25_results:
            idx = r["index"]
            if idx not in all_results:
                all_results[idx] = r
            all_results[idx]["rrf_score"] = rrf_scores.get(idx, 0)

        merged = sorted(all_results.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Rerank top candidates
        return self._rerank(query, merged[:initial_k], k=k)

    # ----- Intent Classification -----

    def _classify_intent(self, question: str) -> str:
        """Fast regex-based intent classification."""
        q = question.lower().strip()
        if re.search(r'(differ|compar|vs\.?|versus|between .+ and)', q):
            return 'COMPARISON'
        if re.search(r'(how (do|can|does|to|is)|step|process|procedure|file a|apply for|register)', q):
            return 'PROCEDURAL'
        if re.search(r'(must|required|obligat|comply|complian|penalty|penalt|mandatory|prohibit|allowed|permitted|legal)', q):
            return 'COMPLIANCE'
        if re.search(r'(what (is|are|was|were)|how (much|many|often|long)|limit|fee|charge|rate|amount|list|define|meaning)', q):
            return 'FACTUAL'
        return 'GENERAL'

    # ----- Query -----

    def query(self, question: str, k: int = 5, example_style: str = "None") -> Dict:
        """Full query pipeline with hybrid retrieval + LLM answer."""
        start_time = time.time()
        initial_k = int(os.getenv("RERANK_INITIAL_K", "12"))

        # 1. Classify intent
        intent = self._classify_intent(question)

        # 2. Retrieve
        results = self._hybrid_retrieve(question, k=k, initial_k=initial_k)

        # 3. Build context
        context = "\n\n---\n\n".join(r["content"] for r in results)

        # 4. Format prompt
        prompt_template = PROMPT_REGISTRY.get(intent, GENERAL_PROMPT)
        chat_history = self._get_chat_history_text()
        formatted_prompt = prompt_template.format(
            context=context, question=question, chat_history=chat_history
        )

        # Append simplification instructions
        simplify = SIMPLIFY_INSTRUCTIONS.get(example_style, "")
        if simplify:
            formatted_prompt += simplify

        # 5. Generate answer
        response = self.llm.invoke(formatted_prompt)
        answer = response.content

        # 6. Build sources
        sources = []
        for r in results:
            chunk = r["chunk"]
            meta = chunk.metadata
            content_preview = r["content"][:300]
            sources.append({
                "content": content_preview,
                "metadata": meta,
                "document_type": meta.get("document_type", "Unknown"),
                "section": meta.get("section", "General").replace("_", " ").title(),
                "issuing_authority": meta.get("issuing_authority", "Unknown"),
                "source_file": meta.get("source", "Unknown"),
                "relevance_score": r.get("rerank_score", r.get("rrf_score", 0)),
            })

        # 7. Extract confidence
        confidence = self._extract_confidence(answer)

        # 8. Update memory
        self.chat_history.append({"question": question, "answer": answer, "intent": intent})

        response_time_ms = int((time.time() - start_time) * 1000)

        # 9. Log analytics
        query_id = None
        if self.analytics:
            query_id = self.analytics.log_query(
                question=question, answer=answer, sources=sources,
                intent_category=intent, response_time_ms=response_time_ms,
                confidence=confidence, retrieval_method="FAISS+BM25+Rerank",
            )

        return {
            "answer": answer,
            "sources": sources,
            "num_retrieved": len(sources),
            "intent": intent,
            "confidence": confidence,
            "response_time_ms": response_time_ms,
            "retrieval_method": "FAISS+BM25+Rerank",
            "query_id": query_id,
        }

    def _get_chat_history_text(self) -> str:
        if not self.chat_history:
            return "No previous conversation."
        lines = []
        for entry in self.chat_history[-self.max_history:]:
            lines.append(f"User: {entry['question']}")
            lines.append(f"Assistant: {entry['answer'][:200]}...")
            lines.append("")
        return "\n".join(lines)

    def _extract_confidence(self, answer: str) -> str:
        a = answer.lower()
        if "confidence: high" in a or "**high**" in a:
            return "High"
        if "confidence: low" in a or "**low**" in a or "don't have enough information" in a:
            return "Low"
        return "Medium"

    def submit_feedback(self, query_id: int, rating: int, comment: str = "") -> None:
        if self.analytics and query_id:
            self.analytics.log_feedback(query_id, rating, comment)

    def clear_history(self):
        self.chat_history = []

    def get_suggested_questions(self) -> Dict:
        return SUGGESTED_QUESTIONS
