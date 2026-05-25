"""
RAG Pipeline Module - End-to-End Retrieval-Augmented Generation
================================================================

Orchestrates the full RAG flow:
  Query → Embed → Retrieve → Generate → (Hallucination Check) → RAGResponse

This is the single entry point that the web app and CLI use.
"""

import logging
import time
from typing import Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import (
    EMBEDDING_MODEL,
    VECTOR_DB_DIR,
    CHROMA_COLLECTION_NAME,
    TOP_K,
    SIMILARITY_THRESHOLD,
    LLM_PROVIDER,
)
from shared.utils import RAGResponse
from deliverable_1.src.embedder import Embedder
from deliverable_1.src.vector_store import VectorStore
from deliverable_1.src.retriever import Retriever
from deliverable_1.src.generator import Generator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for UAE legal question answering.

    Wires together Embedder → VectorStore → Retriever → Generator
    and exposes a single ``query()`` method.

    Args:
        embedding_model: HuggingFace model name for embeddings.
        vector_db_dir: Path to the ChromaDB persistence directory.
        collection_name: ChromaDB collection name.
        top_k: Number of chunks to retrieve.
        similarity_threshold: Minimum cosine similarity for retrieved chunks.
        llm_provider: LLM backend (``"gemini"``, ``"openai"``, ``"ollama"``).
        enable_hallucination_check: Whether to run the hallucination checker.
    """

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        vector_db_dir: Optional[str] = None,
        collection_name: str = CHROMA_COLLECTION_NAME,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        llm_provider: str = LLM_PROVIDER,
        enable_hallucination_check: bool = True,
    ):
        self.enable_hallucination_check = enable_hallucination_check

        # Resolve the vector DB directory
        db_dir = vector_db_dir or str(VECTOR_DB_DIR)

        logger.info("═" * 60)
        logger.info("  Initialising RAG Pipeline – Better Call Saul AI")
        logger.info("═" * 60)

        # Assemble pipeline components
        self.embedder = Embedder(model_name=embedding_model)
        self.vector_store = VectorStore(
            persist_directory=db_dir,
            collection_name=collection_name,
        )
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        self.generator = Generator(provider=llm_provider)

        logger.info("RAG Pipeline ready ✓")

    # ------------------------------------------------------------------
    # Main query method
    # ------------------------------------------------------------------
    def query(self, question: str) -> RAGResponse:
        """Run the full RAG pipeline for a user question.

        Steps:
            1. Retrieve relevant chunks via semantic search.
            2. Generate an answer using the LLM with retrieved context.
            3. Optionally check for hallucinations.
            4. Compute a confidence score from retrieval similarities.

        Args:
            question: The user's natural-language question.

        Returns:
            A fully populated RAGResponse object.
        """
        logger.info("Processing query: %s", question[:100])

        # ── Step 1: Retrieve ─────────────────────────────────────────
        t0 = time.time()
        retrieval_results = self.retriever.retrieve(question)
        retrieval_time = time.time() - t0

        if not retrieval_results:
            return RAGResponse(
                answer=(
                    "I couldn't find relevant legal provisions for your question "
                    "in our UAE law database. Could you try rephrasing or asking "
                    "about a specific UAE law topic?"
                ),
                sources=[],
                confidence=0.0,
                query=question,
                retrieval_time=retrieval_time,
                generation_time=0.0,
            )

        # ── Step 2: Generate ─────────────────────────────────────────
        answer, generation_time = self.generator.generate(question, retrieval_results)

        # ── Step 3: Hallucination check (optional) ───────────────────
        hallucination_flags = []
        if self.enable_hallucination_check:
            try:
                hallucination_flags = self.generator.check_hallucination(
                    answer, retrieval_results
                )
            except Exception as exc:
                logger.warning("Hallucination check skipped: %s", exc)

        # ── Step 4: Confidence score ─────────────────────────────────
        # Average of top retrieval scores, clamped to [0, 1]
        scores = [r.score for r in retrieval_results]
        confidence = min(1.0, max(0.0, sum(scores) / len(scores))) if scores else 0.0

        # Penalise confidence if hallucination flags exist
        if hallucination_flags:
            confidence *= 0.7

        return RAGResponse(
            answer=answer,
            sources=retrieval_results,
            confidence=round(confidence, 4),
            hallucination_flags=hallucination_flags,
            query=question,
            retrieval_time=round(retrieval_time, 4),
            generation_time=round(generation_time, 4),
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Return vector store statistics."""
        return self.vector_store.get_stats()
