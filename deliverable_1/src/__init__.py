"""
Deliverable 1: Data Pipeline for UAE Legal RAG System
=====================================================

This package contains the data ingestion, preprocessing, and chunking
modules for the "Better Call Saul AI" UAE Legal RAG system.

Modules:
    - data_loader: Load legal documents from various sources (text, PDF, HuggingFace)
    - preprocessor: Clean, parse, and normalize legal text
    - chunker: Split documents into retrieval-optimized chunks
"""

from deliverable_1.src.data_loader import load_all_documents, load_text_file
from deliverable_1.src.preprocessor import preprocess_documents
from deliverable_1.src.chunker import chunk_documents

__all__ = [
    "load_all_documents",
    "load_text_file",
    "preprocess_documents",
    "chunk_documents",
]
