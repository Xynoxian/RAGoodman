"""
Retriever Module - Semantic Search Over Legal Chunks
=====================================================

Takes a user query, encodes it with the Embedder, searches the VectorStore,
and returns ranked RetrievalResult objects filtered by a similarity threshold.
"""

import logging
import time
from typing import List

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils import Chunk, RetrievalResult
from shared.config import TOP_K, SIMILARITY_THRESHOLD
from deliverable_1.src.embedder import Embedder
from deliverable_1.src.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves the most relevant legal chunks for a given query.

    Combines the Embedder (query encoding) and VectorStore (ANN search)
    to produce a ranked list of RetrievalResult objects.

    Args:
        embedder: An initialised Embedder instance.
        vector_store: An initialised VectorStore instance.
        top_k: Maximum number of results to return.
        similarity_threshold: Minimum cosine similarity to accept.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        logger.info(
            "Retriever initialised – top_k=%d, threshold=%.2f",
            top_k,
            similarity_threshold,
        )

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve relevant chunks for *query*.

        Steps:
            1. Encode the query with the Embedder.
            2. Search ChromaDB for top-k nearest neighbours.
            3. Convert distances → cosine similarity scores.
            4. Filter by similarity threshold.
            5. Return sorted RetrievalResult list.

        Returns:
            List[RetrievalResult] sorted by descending score.
        """
        start = time.time()

        # Step 1 – encode query
        query_vec = self.embedder.embed_text(query)
        # Step 2 – ANN search
        raw = self.vector_store.search(query_vec, top_k=self.top_k)

        results: List[RetrievalResult] = []

        if not raw["ids"] or not raw["ids"][0]:
            logger.warning("No results returned for query: %s", query)
            return results

        # Step 3 & 4 – build RetrievalResult objects, filter
        for idx, doc_id in enumerate(raw["ids"][0]):
            # ChromaDB cosine distance ∈ [0, 2]; similarity = 1 - distance
            distance = raw["distances"][0][idx]
            score = 1.0 - distance

            if score < self.similarity_threshold:
                continue

            chunk = Chunk(
                text=raw["documents"][0][idx],
                metadata=raw["metadatas"][0][idx] if raw["metadatas"] else {},
                chunk_id=doc_id,
            )
            results.append(RetrievalResult(chunk=chunk, score=round(score, 4)))

        # Step 5 – sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        elapsed = time.time() - start
        logger.info(
            "Retrieved %d results in %.3fs for query: %s",
            len(results),
            elapsed,
            query[:80],
        )
        return results
