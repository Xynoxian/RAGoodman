"""
Utility Classes & Helpers - UAE Legal RAG System
=================================================

This module contains:
1. Core data classes used throughout the pipeline (Document, Chunk, etc.)
2. Logging setup with colored output for better developer experience
3. Timing utilities for performance measurement
4. Directory creation helpers

These data classes define the "language" of our RAG pipeline — every module
speaks in terms of Documents, Chunks, RetrievalResults, and RAGResponses.

Usage:
    from shared.utils import Document, Chunk, RetrievalResult, RAGResponse
    from shared.utils import setup_logging, ensure_dirs
"""

import logging
import time
import functools
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# =============================================================================
# Core Data Classes
# =============================================================================
# These dataclasses flow through the pipeline like this:
#   PDF -> Document -> Chunk -> (embedded) -> RetrievalResult -> RAGResponse
#
# Think of them as the "nouns" of our RAG system.
# =============================================================================


@dataclass
class Document:
    """
    Represents a single parsed legal document (e.g., one UAE law PDF).

    In a RAG pipeline, this is the raw material — the full text of a law
    before it gets chunked into smaller pieces for embedding.

    Attributes:
        text: The full text content of the document.
        metadata: Structured info about the document:
            - source: filename or URL where this came from
            - law_name: official name of the law (e.g., "UAE Labour Law")
            - article_number: specific article if applicable
            - chapter: chapter heading
            - section: section heading
        doc_id: Unique identifier for this document (auto-generated if empty).
    """
    text: str
    metadata: dict  # keys: source, law_name, article_number, chapter, section
    doc_id: str = ""

    def __post_init__(self):
        """Ensure metadata has all expected keys with defaults."""
        default_keys = ["source", "law_name", "article_number", "chapter", "section"]
        for key in default_keys:
            if key not in self.metadata:
                self.metadata[key] = ""


@dataclass
class Chunk:
    """
    A smaller piece of a Document, sized for embedding and retrieval.

    Why chunk? Because embedding models have token limits, and smaller chunks
    give more precise retrieval. A 512-char chunk typically covers 1-2 articles,
    which is the right granularity for legal Q&A.

    Attributes:
        text: The chunk's text content.
        metadata: Inherits all Document metadata PLUS:
            - chunk_index: position of this chunk within its parent document
            - chunk_size: number of characters in this chunk
        chunk_id: Unique identifier (typically: doc_id + chunk_index).
    """
    text: str
    metadata: dict  # inherits Document metadata + chunk_index, chunk_size
    chunk_id: str = ""


@dataclass
class RetrievalResult:
    """
    A single search result: a Chunk paired with its relevance score.

    The score comes from the vector similarity search (cosine similarity)
    or BM25 ranking. Higher score = more relevant to the query.

    Attributes:
        chunk: The retrieved Chunk object.
        score: Relevance score (0.0 to 1.0 for cosine similarity).
    """
    chunk: Chunk
    score: float


@dataclass
class RAGResponse:
    """
    The final output of the RAG pipeline — an answer with full provenance.

    This is what the user sees (via the web UI or API). It includes:
    - The generated answer text (in Saul Goodman's voice)
    - Which sources were used (for transparency and verification)
    - Confidence score (how sure the system is)
    - Hallucination flags (any detected issues with grounding)
    - Timing info (for performance monitoring)

    Attributes:
        answer: The generated response text.
        sources: List of RetrievalResults that informed the answer.
        confidence: Overall confidence score (0.0 to 1.0).
        hallucination_flags: List of detected hallucination issues (empty = clean).
        query: The original user query.
        retrieval_time: Seconds spent on retrieval.
        generation_time: Seconds spent on LLM generation.
    """
    answer: str
    sources: List[RetrievalResult]
    confidence: float
    hallucination_flags: List[str] = field(default_factory=list)
    query: str = ""
    retrieval_time: float = 0.0
    generation_time: float = 0.0

    @property
    def total_time(self) -> float:
        """Total pipeline time = retrieval + generation."""
        return self.retrieval_time + self.generation_time

    @property
    def is_hallucinated(self) -> bool:
        """Quick check: did the hallucination detector flag anything?"""
        return len(self.hallucination_flags) > 0

    @property
    def source_count(self) -> int:
        """Number of sources used to generate this answer."""
        return len(self.sources)


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with colored output for better readability.

    Each module should call this once at the top:
        logger = setup_logging(__name__)
        logger.info("Starting document processing...")

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance with colored output.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Try to use coloredlogs for pretty terminal output
    try:
        import coloredlogs
        coloredlogs.install(
            level=level,
            logger=logger,
            fmt="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
            level_styles={
                "debug": {"color": "cyan"},
                "info": {"color": "green"},
                "warning": {"color": "yellow", "bold": True},
                "error": {"color": "red", "bold": True},
                "critical": {"color": "red", "bold": True, "background": "white"},
            },
            field_styles={
                "asctime": {"color": "blue"},
                "name": {"color": "magenta"},
                "levelname": {"color": "white", "bold": True},
            },
        )
    except ImportError:
        # Fallback to standard logging if coloredlogs is not installed
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# =============================================================================
# Timing Decorator
# =============================================================================


def timer_decorator(func):
    """
    Decorator that logs how long a function takes to execute.

    Useful for monitoring pipeline performance — we want to know if retrieval
    is slow, if embedding is the bottleneck, etc.

    Usage:
        @timer_decorator
        def embed_documents(docs):
            ...

    This will log: "embed_documents completed in 2.34s"
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()

        logger.info(f"⏱️  Starting {func.__name__}...")
        result = func(*args, **kwargs)

        elapsed = time.time() - start_time
        logger.info(f"✅ {func.__name__} completed in {elapsed:.2f}s")

        return result

    return wrapper


# =============================================================================
# Directory Management
# =============================================================================


def ensure_dirs() -> None:
    """
    Create all project directories if they don't already exist.

    This is called at startup to ensure the data pipeline directories
    are ready before any processing begins. Uses mkdir(parents=True)
    so intermediate directories are created automatically.

    Created directories:
        - deliverable_1/data/raw/        (for PDF inputs)
        - deliverable_1/data/processed/  (for cleaned text)
        - deliverable_1/data/chunks/     (for chunked documents)
        - deliverable_1/data/vectordb/   (for ChromaDB persistence)
        - deliverable_2/experiments/     (for experiment results)
    """
    # Import here to avoid circular imports (config imports from this module's sibling)
    from shared.config import (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_CHUNKS_DIR,
        VECTOR_DB_DIR,
        EXPERIMENTS_DIR,
    )

    directories = [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_CHUNKS_DIR,
        VECTOR_DB_DIR,
        EXPERIMENTS_DIR,
    ]

    logger = setup_logging(__name__)

    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"📁 Ensured directory exists: {dir_path}")

    logger.info(f"📁 All {len(directories)} project directories verified.")
