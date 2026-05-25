"""
Embedder Module — Sentence-Transformer Embedding Wrapper
=========================================================

Provides a clean, production-ready interface for generating dense vector
embeddings from text using HuggingFace sentence-transformers.

**Why sentence-transformers?**
Unlike raw BERT, sentence-transformer models are fine-tuned so that
semantically similar sentences produce *close* vectors (measured by cosine
similarity).  This is essential for a retrieval step in RAG.

Usage::

    from deliverable_1.src.embedder import EmbeddingModel
    model = EmbeddingModel()
    vec = model.embed_text("What is the penalty for theft?")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Path fix — allow imports from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION  # noqa: E402
from shared.utils import Chunk, setup_logging                    # noqa: E402

logger = setup_logging(__name__)


# ═══════════════════════════════════════════════════════════════════════════
class EmbeddingModel:
    """Wraps a sentence-transformer model for embedding text.

    The model is **lazy-loaded**: the heavy ``SentenceTransformer`` import
    and weight download only happen the first time you actually embed
    something.  This keeps startup fast when only configuration is needed.

    Args:
        model_name: HuggingFace model identifier.  Defaults to the value
            in ``shared.config.EMBEDDING_MODEL``
            (``sentence-transformers/all-MiniLM-L6-v2``).
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name: str = model_name or EMBEDDING_MODEL
        self.dimension: int = EMBEDDING_DIMENSION
        self._model = None  # lazy-loaded on first embed call

        logger.info(
            "EmbeddingModel initialised  ·  model=%s  ·  dim=%d",
            self.model_name,
            self.dimension,
        )

    # ------------------------------------------------------------------
    # Lazy model loading — avoids the multi-second import until needed
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load the sentence-transformer model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformer model '%s' …", self.model_name)
            self._model = SentenceTransformer(self.model_name)

            # Verify dimension matches config expectation
            test_dim = self._model.get_sentence_embedding_dimension()
            if test_dim != self.dimension:
                logger.warning(
                    "Model dimension (%d) differs from config EMBEDDING_DIMENSION (%d). "
                    "Updating to match model.",
                    test_dim,
                    self.dimension,
                )
                self.dimension = test_dim

            logger.info(
                "Model loaded successfully  ·  dimension=%d  ·  max_seq_length=%s",
                self.dimension,
                getattr(self._model, "max_seq_length", "unknown"),
            )

        except ImportError:
            logger.error(
                "sentence-transformers is not installed. "
                "Run:  pip install sentence-transformers"
            )
            raise
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Model info (useful for experiment logging)
    # ------------------------------------------------------------------
    def get_model_info(self) -> dict:
        """Return a dict of model metadata for logging / experiment tracking.

        Returns:
            dict with keys *model_name*, *dimension*, *max_seq_length*, *loaded*.
        """
        info = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "loaded": self._model is not None,
        }
        if self._model is not None:
            info["max_seq_length"] = getattr(self._model, "max_seq_length", None)
        return info

    # ==================================================================
    # Public API
    # ==================================================================

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string and return its vector.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats of length ``self.dimension``.
        """
        self._load_model()

        if not text or not text.strip():
            logger.warning("embed_text called with empty/blank text — returning zero vector.")
            return [0.0] * self.dimension

        # encode returns np.ndarray; .tolist() converts to plain Python floats
        embedding = self._model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch-embed multiple texts with a progress bar.

        Sentence-transformers handles batching internally, so this is
        more efficient than calling ``embed_text`` in a loop.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        self._load_model()

        if not texts:
            logger.warning("embed_texts called with empty list — returning [].")
            return []

        logger.info("Embedding %d texts …", len(texts))

        # show_progress_bar=True gives a nice tqdm bar for large batches
        embeddings = self._model.encode(
            texts,
            show_progress_bar=len(texts) > 10,  # only show bar for non-trivial batches
            batch_size=64,
        )

        logger.info("Embedding complete  ·  shape=%s", embeddings.shape)
        return embeddings.tolist()

    def embed_chunks(
        self, chunks: List[Chunk]
    ) -> List[Tuple[Chunk, List[float]]]:
        """Embed a list of Chunk objects and return (chunk, vector) pairs.

        This is the primary entry-point used by the indexing pipeline.

        Args:
            chunks: List of ``Chunk`` dataclass instances.

        Returns:
            List of ``(Chunk, embedding)`` tuples.
        """
        if not chunks:
            logger.warning("embed_chunks called with empty list — returning [].")
            return []

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        # Pair each chunk back with its embedding
        results: List[Tuple[Chunk, List[float]]] = list(zip(chunks, embeddings))
        logger.info("Embedded %d chunks successfully.", len(results))
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Quick smoke-test when run directly
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("EmbeddingModel — Smoke Test")
    print("=" * 60)

    model = EmbeddingModel()
    print(f"\nModel info: {model.get_model_info()}")

    # Single text
    vec = model.embed_text("What is the penalty for theft in the UAE?")
    print(f"\nSingle embed  → dim={len(vec)}, first 5 values={vec[:5]}")

    # Batch
    batch = [
        "UAE labour law annual leave entitlement",
        "Child custody rules in UAE personal status law",
    ]
    vecs = model.embed_texts(batch)
    print(f"Batch embed   → {len(vecs)} vectors, dim={len(vecs[0])}")

    # Chunk objects
    test_chunks = [
        Chunk(text="Article 1: test", metadata={"source": "test"}),
        Chunk(text="Article 2: test", metadata={"source": "test"}),
    ]
    pairs = model.embed_chunks(test_chunks)
    print(f"Chunk embed   → {len(pairs)} (chunk, vec) pairs")

    print(f"\nModel info (after load): {model.get_model_info()}")
    print("\n✅ All smoke tests passed!")
