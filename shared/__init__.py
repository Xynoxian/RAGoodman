"""
Shared Module - UAE Legal RAG System ("Better Call Saul AI")
===========================================================

This package provides the foundational components shared across all deliverables:
- config.py   : Centralized configuration (paths, model names, hyperparameters)
- utils.py    : Data classes (Document, Chunk, RetrievalResult, RAGResponse) and helpers
- prompts.py  : All prompt templates including Saul Goodman persona variants
- constants.py: Project-wide constants (law categories, regex patterns, etc.)

Usage:
    from shared.config import *
    from shared.utils import Document, Chunk, RetrievalResult, RAGResponse
    from shared.prompts import SAUL_SYSTEM_PROMPT
    from shared.constants import UAE_LAW_CATEGORIES
"""

from shared.utils import Document, Chunk, RetrievalResult, RAGResponse
from shared.constants import UAE_LAW_CATEGORIES, DEFAULT_DISCLAIMER

__all__ = [
    "Document",
    "Chunk",
    "RetrievalResult",
    "RAGResponse",
    "UAE_LAW_CATEGORIES",
    "DEFAULT_DISCLAIMER",
]
