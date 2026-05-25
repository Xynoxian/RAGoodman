"""
Cross-Encoder Reranker — Second-Stage Relevance Scoring
========================================================

In a typical RAG pipeline, the first-stage retriever (dense or BM25) is
optimised for *recall* — casting a wide net to find potentially relevant
chunks.  However, first-stage scores are often noisy.

A **cross-encoder reranker** acts as a second stage that scores each
(query, chunk) pair *jointly* through a transformer model, producing
much more accurate relevance scores.  This is computationally expensive
(O(n) forward passes vs one for bi-encoder), which is why we only
rerank the top-k candidates from stage 1.

**Architecture difference:**
- Bi-encoder (stage 1): Embeds query and document separately, compares
  with cosine similarity.  Fast but less accurate.
- Cross-encoder (stage 2): Concatenates [query, SEP, document] and
  produces a single relevance score.  Slow but very accurate.

This module provides a ``CrossEncoderReranker`` class that wraps the
``sentence_transformers.CrossEncoder`` model.

Usage::

    from deliverable_2.src.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("What is the penalty?", retrieval_results, top_k=3)
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils import RetrievalResult, Chunk, setup_logging

logger = setup_logging(__name__)


class CrossEncoderReranker:
    """Reranks retrieval results using a cross-encoder model.

    Cross-encoders concatenate the query and document into a single input
    and produce a relevance score via a classification head.  This allows
    the model to attend to fine-grained interactions between query and
    document tokens, yielding much better relevance judgements than
    cosine similarity over independently encoded vectors.

    The default model ``cross-encoder/ms-marco-MiniLM-L-6-v2`` is trained
    on the MS MARCO passage ranking dataset and generalises well to
    legal text despite not being domain-specific.

    Attributes:
        model_name: HuggingFace model identifier for the cross-encoder.
        _model: Lazy-loaded CrossEncoder instance.
    """

    # Default model: good trade-off between speed and accuracy
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialise the reranker.

        The actual model loading is deferred until the first ``rerank()``
        call to keep startup fast.

        Args:
            model_name: HuggingFace model ID for a cross-encoder model.
                Defaults to ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
        """
        self.model_name: str = model_name or self.DEFAULT_MODEL
        self._model = None  # Lazy-loaded

        logger.info(
            "CrossEncoderReranker initialised — model=%s (lazy-loaded)",
            self.model_name,
        )

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load the cross-encoder model on first use.

        Raises:
            ImportError: If sentence-transformers is not installed.
            Exception: If the model fails to download/load.
        """
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder model '%s'…", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully.")

        except ImportError:
            logger.error(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as exc:
            logger.error("Failed to load cross-encoder model: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Rerank retrieval results using the cross-encoder.

        Each (query, chunk.text) pair is scored by the cross-encoder.
        Results are re-sorted by the new scores and the top-k are returned.

        The original retrieval scores are preserved in the metadata under
        the key ``original_score`` for analysis, while the ``score`` field
        is replaced with the cross-encoder score (sigmoid-normalised to
        [0, 1]).

        Args:
            query: The user's natural-language question.
            results: List of RetrievalResult objects from the first-stage
                retriever.  These will be reranked.
            top_k: Number of top results to return after reranking.

        Returns:
            A new list of RetrievalResult objects sorted by cross-encoder
            score (descending), limited to ``top_k`` items.

        Note:
            If ``results`` is empty or the model fails to load, the original
            results are returned unchanged (graceful degradation).
        """
        if not results:
            logger.warning("rerank() called with empty results — returning [].")
            return []

        if len(results) <= 1:
            logger.debug("Only one result — no reranking needed.")
            return results[:top_k]

        # Load model (no-op if already loaded)
        try:
            self._load_model()
        except Exception as exc:
            logger.warning(
                "Cross-encoder model unavailable (%s) — returning original ranking.",
                exc,
            )
            return results[:top_k]

        # Build (query, document) pairs for the cross-encoder
        pairs = [(query, result.chunk.text) for result in results]

        logger.debug("Reranking %d results with cross-encoder…", len(pairs))

        try:
            # CrossEncoder.predict returns raw logits; for ms-marco models
            # these are relevance scores (higher = more relevant)
            scores = self._model.predict(pairs)

            # Normalise scores to [0, 1] using sigmoid
            import numpy as np
            normalised_scores = 1 / (1 + np.exp(-np.array(scores)))

        except Exception as exc:
            logger.error("Cross-encoder prediction failed: %s", exc)
            return results[:top_k]

        # Create new RetrievalResult objects with updated scores
        reranked: List[RetrievalResult] = []
        for i, result in enumerate(results):
            new_score = float(normalised_scores[i])

            # Preserve original score in metadata for analysis
            updated_metadata = {
                **result.chunk.metadata,
                "original_score": result.score,
                "reranker_score": new_score,
                "reranked": True,
            }

            reranked_chunk = Chunk(
                text=result.chunk.text,
                metadata=updated_metadata,
                chunk_id=result.chunk.chunk_id,
            )

            reranked.append(RetrievalResult(
                chunk=reranked_chunk,
                score=new_score,
            ))

        # Sort by cross-encoder score (descending)
        reranked.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            "Reranking complete: top score %.4f → %.4f, returned %d/%d results.",
            reranked[0].score if reranked else 0,
            reranked[-1].score if reranked else 0,
            min(top_k, len(reranked)),
            len(results),
        )

        return reranked[:top_k]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_model_info(self) -> dict:
        """Return metadata about the reranker model.

        Returns:
            Dict with keys: model_name, loaded.
        """
        return {
            "model_name": self.model_name,
            "loaded": self._model is not None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CrossEncoderReranker — Smoke Test")
    print("=" * 60)

    # Create some mock retrieval results
    mock_results = [
        RetrievalResult(
            chunk=Chunk(
                text="Article 461: The penalty for theft is imprisonment for a period not exceeding 3 years.",
                metadata={"law_name": "UAE Penal Code", "article_number": "461"},
                chunk_id="c1",
            ),
            score=0.75,
        ),
        RetrievalResult(
            chunk=Chunk(
                text="Article 30: Annual leave entitlement is 30 days for employees who have completed one year.",
                metadata={"law_name": "UAE Labour Law", "article_number": "30"},
                chunk_id="c2",
            ),
            score=0.82,
        ),
        RetrievalResult(
            chunk=Chunk(
                text="Article 462: Aggravated theft carries a higher penalty of up to 7 years imprisonment.",
                metadata={"law_name": "UAE Penal Code", "article_number": "462"},
                chunk_id="c3",
            ),
            score=0.65,
        ),
    ]

    reranker = CrossEncoderReranker()
    query = "What is the punishment for theft?"

    print(f"\nQuery: {query}")
    print(f"\nBefore reranking:")
    for r in mock_results:
        print(f"  Score={r.score:.4f} | {r.chunk.text[:80]}…")

    reranked = reranker.rerank(query, mock_results, top_k=3)

    print(f"\nAfter reranking:")
    for r in reranked:
        print(f"  Score={r.score:.4f} (orig={r.chunk.metadata.get('original_score', '?')}) | {r.chunk.text[:80]}…")

    print("\n✅ Smoke test completed!")
