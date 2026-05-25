#!/usr/bin/env python3
"""
Document Ingestion Pipeline — Better Call Saul AI
===================================================

This script orchestrates the full ingestion pipeline:
    1. Load raw legal documents from data/raw/
    2. Preprocess and clean the text
    3. Chunk documents using article-based splitting
    4. Generate embeddings using sentence-transformers
    5. Store everything in ChromaDB for retrieval

Run this ONCE before using the RAG system, or whenever you add new documents.

Usage:
    python deliverable_1/scripts/ingest_documents.py

Design Decision:
    We process all documents in memory before batch-inserting into ChromaDB.
    For our dataset size (~200 articles), this is efficient. For larger corpora,
    you'd want streaming/batched insertion.
"""

import sys
import json
import time
from pathlib import Path

# Add project root to path so we can import shared modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils import setup_logging, ensure_dirs
from shared.config import DATA_RAW_DIR, DATA_CHUNKS_DIR
from deliverable_1.src.data_loader import load_all_documents
from deliverable_1.src.preprocessor import preprocess_documents
from deliverable_1.src.chunker import chunk_documents
from deliverable_1.src.embedder import EmbeddingModel
from deliverable_1.src.vector_store import VectorStore

logger = setup_logging("ingest_pipeline")


def main():
    """
    Main ingestion pipeline.

    Steps:
        1. Ensure all directories exist
        2. Load documents from data/raw/
        3. Preprocess (clean, normalize)
        4. Chunk (article-based splitting)
        5. Embed (sentence-transformers)
        6. Store in ChromaDB
        7. Print summary statistics
    """
    logger.info("=" * 70)
    logger.info("⚖️  BETTER CALL SAUL AI — Document Ingestion Pipeline")
    logger.info("=" * 70)

    start_time = time.time()

    # Step 0: Ensure directories exist
    logger.info("\n📁 Step 0: Creating project directories...")
    ensure_dirs()

    # Step 1: Load documents
    logger.info("\n📄 Step 1: Loading documents from %s", DATA_RAW_DIR)
    documents = load_all_documents(DATA_RAW_DIR)

    if not documents:
        logger.error("❌ No documents found in %s", DATA_RAW_DIR)
        logger.error("   Make sure you have .txt or .pdf files in the data/raw/ directory.")
        sys.exit(1)

    logger.info("   Loaded %d document(s)", len(documents))
    for doc in documents[:5]:  # Show first 5
        source = doc.metadata.get("source", "unknown")
        logger.info("   → %s (%d chars)", source, len(doc.text))

    # Step 2: Preprocess documents
    logger.info("\n🔧 Step 2: Preprocessing documents...")
    processed_docs = preprocess_documents(documents)
    logger.info("   Preprocessed %d document(s)", len(processed_docs))

    # Step 3: Chunk documents
    logger.info("\n✂️  Step 3: Chunking documents (article-based)...")
    chunks = chunk_documents(processed_docs, strategy="article")
    logger.info("   Created %d chunk(s)", len(chunks))

    if chunks:
        # Calculate chunk statistics
        chunk_sizes = [len(c.text) for c in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        logger.info("   Chunk size stats: avg=%.0f, min=%d, max=%d chars", avg_size, min_size, max_size)

    # Save chunks to JSON for inspection
    chunks_file = DATA_CHUNKS_DIR / "chunks.json"
    chunks_data = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
            "metadata": c.metadata,
            "full_length": len(c.text),
        }
        for c in chunks
    ]
    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    logger.info("   Saved chunk metadata to %s", chunks_file)

    # Step 4: Generate embeddings
    logger.info("\n🧠 Step 4: Generating embeddings...")
    embedder = EmbeddingModel()
    embeddings = embedder.embed_texts([c.text for c in chunks])
    logger.info("   Generated %d embedding(s) of dimension %d", len(embeddings), len(embeddings[0]) if embeddings else 0)

    # Step 5: Store in ChromaDB
    logger.info("\n💾 Step 5: Storing in ChromaDB vector store...")
    vector_store = VectorStore()

    # Delete existing collection and recreate (fresh start)
    vector_store.delete_collection()
    vector_store = VectorStore()  # Reinitialize after delete

    vector_store.add_chunks(chunks, embeddings)
    stats = vector_store.get_collection_stats()
    logger.info("   Stored %d chunks in collection '%s'", stats.get("count", 0), stats.get("name", "unknown"))

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("✅ INGESTION COMPLETE")
    logger.info("=" * 70)
    logger.info("   Documents loaded:  %d", len(documents))
    logger.info("   Documents processed: %d", len(processed_docs))
    logger.info("   Chunks created:    %d", len(chunks))
    logger.info("   Embeddings stored: %d", len(embeddings))
    logger.info("   Total time:        %.2fs", elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
