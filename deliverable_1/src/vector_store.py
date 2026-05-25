"""
Vector Store Module - ChromaDB Wrapper
=======================================

Provides a thin abstraction over ChromaDB for storing, searching, and
managing legal document chunk embeddings.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent ChromaDB-backed vector store for legal document chunks.

    Args:
        persist_directory: Path where ChromaDB stores its data on disk.
        collection_name: Name of the ChromaDB collection to use.
    """

    def __init__(
        self,
        persist_directory: str = "./data/vectordb",
        collection_name: str = "uae_legal_docs",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        logger.info(
            "VectorStore configured – dir=%s, collection=%s",
            persist_directory,
            collection_name,
        )

    # ------------------------------------------------------------------
    # Lazy init – delays heavy ChromaDB import until first access
    # ------------------------------------------------------------------
    def _ensure_client(self):
        """Initialise the ChromaDB client and collection if needed."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                persist_path = Path(self.persist_directory)
                persist_path.mkdir(parents=True, exist_ok=True)

                self._client = chromadb.PersistentClient(path=str(persist_path))
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "ChromaDB ready – collection '%s' has %d documents",
                    self.collection_name,
                    self._collection.count(),
                )
            except ImportError:
                logger.error("chromadb is not installed. Run: pip install chromadb")
                raise
            except Exception as exc:
                logger.error("Failed to initialise ChromaDB: %s", exc)
                raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_chunks(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add chunk embeddings to the collection."""
        self._ensure_client()
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Added %d chunks to vector store.", len(ids))

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Search for nearest neighbours of *query_embedding*.

        Returns:
            ChromaDB results dict with keys: ids, documents, metadatas, distances.
        """
        self._ensure_client()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return basic statistics about the collection."""
        self._ensure_client()
        count = self._collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_directory": str(self.persist_directory),
        }

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        self._ensure_client()
        return self._collection.count()
