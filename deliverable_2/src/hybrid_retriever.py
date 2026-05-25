"""
Hybrid Retriever – Dense + Sparse Search with Reciprocal Rank Fusion
=====================================================================

This module implements a **hybrid retrieval strategy** that combines:
  1. **Dense retrieval** – semantic search via embeddings (e.g., ChromaDB)
  2. **Sparse retrieval** – keyword-based search via BM25

Combining both approaches is well-established in information retrieval
research.  Dense retrieval excels at capturing *meaning* while BM25
captures exact *keyword matches*.  Reciprocal Rank Fusion (RRF) merges
the two ranked lists without needing score normalisation.

This is a **BONUS** feature for the UAE Legal RAG project.

Usage example::

    from deliverable_2.src.hybrid_retriever import BM25Retriever, HybridRetriever

    bm25 = BM25Retriever(all_chunks)
    hybrid = HybridRetriever(dense_retriever, bm25, dense_weight=0.6)
    results = hybrid.retrieve("What is the penalty for theft?", top_k=5)
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

# ---------------------------------------------------------------------------
# Project imports – allow running from any directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rank_bm25 import BM25Okapi
from shared.utils import Chunk, RetrievalResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# BM25 (Sparse) Retriever
# ═══════════════════════════════════════════════════════════════════════════

class BM25Retriever:
    """Sparse retriever using the Okapi BM25 ranking function.

    BM25 is a *bag-of-words* retrieval model that scores documents based on
    term frequency, inverse document frequency, and document length
    normalisation.  It is particularly effective at matching specific legal
    terminology and article numbers that dense models may gloss over.

    Attributes:
        chunks: The original chunk objects (preserved for metadata access).
        tokenised_corpus: List of tokenised chunk texts.
        bm25: The underlying ``BM25Okapi`` index.
    """

    def __init__(self, chunks: List[Chunk]) -> None:
        """Build a BM25 index from a list of Chunk objects.

        Args:
            chunks: Pre-processed text chunks with metadata.  Each chunk's
                ``text`` field is tokenised by simple whitespace splitting
                and lowercased for case-insensitive matching.
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index from an empty chunk list.")

        self.chunks: List[Chunk] = chunks

        # Tokenise each chunk (simple whitespace split + lowercase)
        # More sophisticated tokenisation (stemming, stop-word removal) could
        # improve results but adds complexity beyond the scope of this project.
        self.tokenised_corpus: List[List[str]] = [
            self._tokenise(chunk.text) for chunk in chunks
        ]

        # Build the BM25 index
        self.bm25 = BM25Okapi(self.tokenised_corpus)
        logger.info("BM25 index built with %d chunks.", len(chunks))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve the top-k most relevant chunks for *query*.

        Args:
            query: The user's natural-language question.
            top_k: Number of results to return (default 5).

        Returns:
            A list of ``RetrievalResult`` objects sorted by descending
            BM25 score.  Scores are *not* normalised to [0, 1].
        """
        tokenised_query = self._tokenise(query)
        scores = self.bm25.get_scores(tokenised_query)

        # Get indices of top-k scores (argsort returns ascending, so negate)
        # We use Python's sorted() for clarity over numpy to avoid the dep.
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results: List[RetrievalResult] = []
        for idx, score in scored_indices:
            if score > 0:  # Only include chunks with non-zero BM25 score
                results.append(RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                ))

        logger.debug("BM25 retrieved %d results for query: '%s'", len(results), query[:80])
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Simple whitespace tokeniser with lowercasing.

        Args:
            text: Raw text string.

        Returns:
            List of lowercase tokens.
        """
        return text.lower().split()


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid Retriever  (Dense + Sparse via RRF)
# ═══════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """Combines dense and sparse retrievers using Reciprocal Rank Fusion.

    **Why hybrid?**  Dense (embedding-based) retrieval captures *semantic
    similarity* — useful for paraphrased questions.  Sparse (BM25) retrieval
    captures *exact keyword matches* — useful for article numbers, specific
    legal terms, and proper nouns.  By fusing both ranked lists we get the
    best of both worlds.

    **Reciprocal Rank Fusion (RRF)** is a simple, parameter-light method
    introduced by Cormack et al. (2009) that avoids the need to normalise
    scores across different scoring functions.

    Attributes:
        dense_retriever: Any retriever with a ``.retrieve(query, top_k)``
            method returning ``list[RetrievalResult]``.
        bm25_retriever: A ``BM25Retriever`` instance.
        dense_weight: Weight for dense results in final scoring (0–1).
    """

    def __init__(
        self,
        dense_retriever,
        bm25_retriever: BM25Retriever,
        dense_weight: float = 0.6,
    ) -> None:
        """Initialise the hybrid retriever.

        Args:
            dense_retriever: An object with a ``retrieve(query, top_k)``
                method (e.g., a ChromaDB-backed retriever).
            bm25_retriever: A ``BM25Retriever`` instance.
            dense_weight: Weight ∈ [0, 1] controlling how much dense results
                contribute to the fused score.  ``1 - dense_weight`` is used
                for sparse results.  Default ``0.6`` slightly favours
                semantic search.
        """
        if not (0.0 <= dense_weight <= 1.0):
            raise ValueError(f"dense_weight must be in [0, 1], got {dense_weight}")

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        logger.info(
            "HybridRetriever initialised (dense_weight=%.2f, sparse_weight=%.2f)",
            dense_weight, 1 - dense_weight,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve top-k results by fusing dense and sparse rankings.

        Steps:
            1. Run *both* retrievers with a larger candidate pool (2×top_k).
            2. Merge results via Reciprocal Rank Fusion.
            3. Return the top-k fused results.

        Args:
            query: User's natural-language question.
            top_k: Number of final results to return.

        Returns:
            Fused list of ``RetrievalResult`` objects sorted by RRF score.
        """
        # Fetch more candidates than needed so RRF has enough to work with
        candidate_k = top_k * 2

        logger.debug("Running dense retrieval for hybrid search…")
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)

        logger.debug("Running BM25 retrieval for hybrid search…")
        sparse_results = self.bm25_retriever.retrieve(query, top_k=candidate_k)

        # Fuse the two ranked lists
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        logger.info(
            "Hybrid retrieval returned %d fused results (top_k=%d).",
            min(len(fused), top_k), top_k,
        )
        return fused[:top_k]

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        k: int = 60,
    ) -> List[RetrievalResult]:
        """Merge two ranked lists using Reciprocal Rank Fusion (RRF).

        The RRF score for a document *d* is:

            RRF(d) = Σ  weight_i / (k + rank_i(d))

        where *k* is a smoothing constant (default 60 per the original
        paper) and *rank_i(d)* is the 1-based rank of *d* in list *i*.

        Args:
            dense_results: Ranked results from the dense retriever.
            sparse_results: Ranked results from the BM25 retriever.
            k: Smoothing constant.  Higher values reduce the impact of
                high-ranking documents.

        Returns:
            A single merged list of ``RetrievalResult`` sorted by RRF score
            (descending).
        """
        # Map chunk_id → (rrf_score, best_chunk_object)
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, Chunk] = {}

        sparse_weight = 1.0 - self.dense_weight

        # Score dense results
        for rank, result in enumerate(dense_results, start=1):
            cid = result.chunk.chunk_id or result.chunk.text[:60]
            rrf_scores[cid] += self.dense_weight / (k + rank)
            chunk_map[cid] = result.chunk  # keep reference

        # Score sparse results
        for rank, result in enumerate(sparse_results, start=1):
            cid = result.chunk.chunk_id or result.chunk.text[:60]
            rrf_scores[cid] += sparse_weight / (k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = result.chunk

        # Build fused result list sorted by RRF score
        fused: List[RetrievalResult] = [
            RetrievalResult(chunk=chunk_map[cid], score=score)
            for cid, score in sorted(
                rrf_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]

        return fused
