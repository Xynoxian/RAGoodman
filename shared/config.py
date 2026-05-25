"""
Centralized Configuration - UAE Legal RAG System
=================================================

All project-wide settings live here. Values are loaded from environment variables
(via .env file) with sensible defaults so the project works out-of-the-box.

Design Decisions:
- We use python-dotenv so developers only need to copy .env.example -> .env
- Path constants use pathlib for cross-platform compatibility
- Every setting can be overridden via environment variables for CI/CD flexibility

Usage:
    from shared.config import *
    # or
    from shared.config import PROJECT_ROOT, EMBEDDING_MODEL, TOP_K
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file at project root
# override=False means existing env vars take precedence over .env file
load_dotenv(override=False)

# =============================================================================
# Path Configuration
# =============================================================================
# PROJECT_ROOT points to the top-level project directory (parent of shared/)
PROJECT_ROOT = Path(__file__).parent.parent

# Deliverable 1: Data pipeline paths
DATA_RAW_DIR = PROJECT_ROOT / "deliverable_1" / "data" / "raw"          # Original PDFs
DATA_PROCESSED_DIR = PROJECT_ROOT / "deliverable_1" / "data" / "processed"  # Cleaned text
DATA_CHUNKS_DIR = PROJECT_ROOT / "deliverable_1" / "data" / "chunks"    # Chunked documents
VECTOR_DB_DIR = PROJECT_ROOT / "deliverable_1" / "data" / "vectordb"    # ChromaDB storage

# Deliverable 2: Experiment results and logs
EXPERIMENTS_DIR = PROJECT_ROOT / "deliverable_2" / "experiments"

# =============================================================================
# Embedding Configuration
# =============================================================================
# MiniLM-L6-v2 is a good balance of speed and quality for semantic search
# It produces 384-dimensional embeddings and runs fast on CPU
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # Fixed dimension for MiniLM-L6-v2

# =============================================================================
# Chunking Configuration
# =============================================================================
# CHUNK_SIZE: number of characters per chunk (512 is ~100 words, fits well in context)
# CHUNK_OVERLAP: overlap between consecutive chunks to preserve context at boundaries
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# =============================================================================
# Retrieval Configuration
# =============================================================================
# TOP_K: number of chunks to retrieve for each query
# SIMILARITY_THRESHOLD: minimum cosine similarity to include a result (filters noise)
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# =============================================================================
# LLM Configuration
# =============================================================================
# We support three providers so students can choose based on budget/preference:
#   - gemini: Google's API (free tier available)
#   - openai: OpenAI's API (paid, high quality)
#   - ollama: Local models (free, requires local setup)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# API Keys (set these in your .env file, never commit real keys!)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model identifiers for each provider
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# =============================================================================
# ChromaDB Configuration
# =============================================================================
# Collection name in ChromaDB — change if you want separate collections
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "uae_legal_docs")

# =============================================================================
# Web Application Configuration
# =============================================================================
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
