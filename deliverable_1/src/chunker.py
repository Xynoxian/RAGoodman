"""
Chunker Module — UAE Legal RAG System ("Better Call Saul AI")
=============================================================

Splits preprocessed Documents into smaller Chunks suitable for embedding
and retrieval.  The chunking strategy is critical for RAG quality:

    - Too large → retrieved context is noisy, wastes LLM tokens
    - Too small → loses context, fragments article meaning
    - Wrong boundaries → splits mid-sentence, breaks legal references

We provide three strategies:

    1. **ArticleBasedChunker** (recommended for legal text)
       Splits at Article boundaries, keeping each article as one chunk.
       If an article exceeds the size limit, it's split with overlap.

    2. **RecursiveChunker**
       Recursively splits text by paragraph → sentence → character,
       similar to LangChain's RecursiveCharacterTextSplitter.

    3. **SlidingWindowChunker**
       Fixed-size window with configurable overlap. Simple but effective
       baseline for comparison.

Usage:
    from deliverable_1.src.chunker import chunk_documents
    chunks = chunk_documents(documents, strategy='article')
"""

from __future__ import annotations

import re
import sys
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path fix — allow imports from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import CHUNK_SIZE, CHUNK_OVERLAP               # noqa: E402
from shared.utils import Document, Chunk, setup_logging            # noqa: E402
from shared.constants import ARTICLE_PATTERN, CHAPTER_PATTERN      # noqa: E402

logger = setup_logging(__name__)


# =============================================================================
# Base Chunker (Abstract)
# =============================================================================

class BaseChunker(ABC):
    """Abstract base class for all chunking strategies.

    Defines the interface that every chunker must implement.  This lets
    the rest of the pipeline swap chunkers without code changes.

    Args:
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Validate: overlap must be less than chunk size (otherwise we'd never advance)
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})."
            )

    @abstractmethod
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a single Document into Chunks.

        Args:
            document: The Document to chunk.

        Returns:
            List of Chunk objects with populated metadata.
        """
        ...

    def _create_chunk(
        self,
        text: str,
        document: Document,
        chunk_index: int,
        article_number: str = "",
        chapter: str = "",
    ) -> Chunk:
        """Create a Chunk object with proper metadata.

        This helper ensures every chunk has consistent metadata
        regardless of which chunking strategy produced it.

        Args:
            text: The chunk text content.
            document: The parent Document (for inheriting metadata).
            chunk_index: Position of this chunk within the document.
            article_number: Article number if known.
            chapter: Chapter name/number if known.

        Returns:
            A fully populated Chunk object.
        """
        # Generate a deterministic chunk ID from document ID + index
        chunk_id_input = f"{document.doc_id}:{chunk_index}"
        chunk_hash = hashlib.md5(chunk_id_input.encode()).hexdigest()[:8]
        chunk_id = f"chunk_{chunk_hash}"

        # Inherit parent document metadata and add chunk-specific fields
        metadata = {
            **document.metadata,
            "chunk_index": str(chunk_index),
            "chunk_size": str(len(text)),
            "article_number": article_number or document.metadata.get("article_number", ""),
            "chapter": chapter or document.metadata.get("chapter", ""),
        }

        return Chunk(text=text, metadata=metadata, chunk_id=chunk_id)


# =============================================================================
# Strategy 1: Article-Based Chunker (Primary for Legal Text)
# =============================================================================

class ArticleBasedChunker(BaseChunker):
    """Splits legal text at Article boundaries.

    This is the recommended strategy for UAE legal documents because:
        - Articles are natural semantic units in legal text
        - Users typically ask about specific articles
        - Each article covers one topic/rule, making retrieval precise
        - Embedding an entire article captures its full meaning

    If an article exceeds the chunk_size, it's split with overlap to
    ensure no information is lost.

    Handles edge cases:
        - Preamble text before the first article → separate chunk
        - Very long articles → sub-chunked with overlap
        - Empty articles → skipped
    """

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split a document by Article boundaries.

        Args:
            document: A preprocessed Document with legal text.

        Returns:
            List of Chunks, one per article (or sub-article for long ones).
        """
        text = document.text
        chunks: List[Chunk] = []
        chunk_index = 0

        # Find the current chapter context (tracks which chapter each article belongs to)
        current_chapter = ""

        # Find all article start positions
        article_starts = list(re.finditer(
            r'(?:^|\n)\s*Article\s+(\d+)\s*[:\.]?\s*',
            text,
            re.IGNORECASE,
        ))

        if not article_starts:
            # No articles found — fall back to sliding window for this document
            logger.warning(
                "No articles found in '%s' — falling back to sliding window.",
                document.metadata.get("source", "unknown"),
            )
            fallback = SlidingWindowChunker(self.chunk_size, self.chunk_overlap)
            return fallback.chunk_document(document)

        # Handle preamble text (everything before the first article)
        preamble_end = article_starts[0].start()
        if preamble_end > 0:
            preamble_text = text[:preamble_end].strip()
            if preamble_text and len(preamble_text) > 50:
                # Only create a chunk if the preamble is substantial
                chunks.append(self._create_chunk(
                    text=preamble_text,
                    document=document,
                    chunk_index=chunk_index,
                    article_number="preamble",
                ))
                chunk_index += 1

        # Process each article
        for i, match in enumerate(article_starts):
            article_number = match.group(1)

            # Determine article text boundaries
            start_pos = match.start()
            end_pos = (
                article_starts[i + 1].start()
                if i + 1 < len(article_starts)
                else len(text)
            )

            article_text = text[start_pos:end_pos].strip()

            if not article_text:
                continue

            # Track current chapter for metadata
            # Look backwards from this article to find the most recent Chapter header
            text_before_article = text[:start_pos]
            chapter_matches = list(CHAPTER_PATTERN.finditer(text_before_article))
            if chapter_matches:
                last_chapter = chapter_matches[-1]
                current_chapter = f"Chapter {last_chapter.group(1)}: {last_chapter.group(2)}".strip()

            # Check if article fits in one chunk
            if len(article_text) <= self.chunk_size:
                chunks.append(self._create_chunk(
                    text=article_text,
                    document=document,
                    chunk_index=chunk_index,
                    article_number=article_number,
                    chapter=current_chapter,
                ))
                chunk_index += 1
            else:
                # Article is too long — sub-chunk with overlap
                sub_chunks = self._split_long_text(article_text)
                for sub_text in sub_chunks:
                    chunks.append(self._create_chunk(
                        text=sub_text,
                        document=document,
                        chunk_index=chunk_index,
                        article_number=article_number,
                        chapter=current_chapter,
                    ))
                    chunk_index += 1

        logger.info(
            "ArticleBasedChunker: '%s' → %d chunks from %d articles",
            document.metadata.get("source", "unknown"),
            len(chunks),
            len(article_starts),
        )

        return chunks

    def _split_long_text(self, text: str) -> List[str]:
        """Split a long article into overlapping sub-chunks.

        Tries to split at sentence boundaries (periods followed by spaces)
        to avoid mid-sentence breaks.

        Args:
            text: Article text that exceeds chunk_size.

        Returns:
            List of sub-chunk strings with overlap.
        """
        sub_chunks: List[str] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                # Last chunk — take everything remaining
                sub_chunks.append(text[start:].strip())
                break

            # Try to find a sentence boundary (". ") near the end
            # Search backwards from 'end' to find a clean break point
            boundary = text.rfind('. ', start + self.chunk_size // 2, end)
            if boundary != -1:
                end = boundary + 1  # Include the period

            sub_chunks.append(text[start:end].strip())

            # Advance by (chunk_size - overlap), ensuring we always move forward
            start = end - self.chunk_overlap

        return sub_chunks


# =============================================================================
# Strategy 2: Recursive Character Chunker
# =============================================================================

class RecursiveChunker(BaseChunker):
    """Recursively splits text by paragraph → sentence → character.

    This mirrors LangChain's RecursiveCharacterTextSplitter approach.
    It tries to split at the most natural boundary possible:
        1. Double newlines (paragraph breaks)
        2. Single newlines (line breaks)
        3. Sentences (periods followed by space)
        4. Words (spaces)
        5. Characters (last resort)

    Good for general-purpose text but less ideal than ArticleBasedChunker
    for structured legal documents.
    """

    # Separators ordered from most preferred to least preferred
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Recursively chunk a document.

        Args:
            document: The Document to chunk.

        Returns:
            List of Chunk objects.
        """
        text = document.text
        raw_chunks = self._recursive_split(text, self.SEPARATORS)
        chunks: List[Chunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Try to detect article number from chunk content
            article_match = ARTICLE_PATTERN.search(chunk_text)
            article_number = article_match.group(1) if article_match else ""

            chunks.append(self._create_chunk(
                text=chunk_text,
                document=document,
                chunk_index=idx,
                article_number=article_number,
            ))

        logger.info(
            "RecursiveChunker: '%s' → %d chunks",
            document.metadata.get("source", "unknown"),
            len(chunks),
        )
        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using progressively finer separators.

        Args:
            text: Text to split.
            separators: Ordered list of separators to try.

        Returns:
            List of text chunks within the size limit.
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Base case: no separators left, hard-split by character count
            return self._hard_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Empty separator means split by character
            return self._hard_split(text)

        # Split by this separator
        parts = text.split(separator)

        # Merge small parts together until they reach chunk_size
        merged_chunks: List[str] = []
        current_chunk = ""

        for part in parts:
            # Would adding this part exceed the limit?
            test_chunk = (
                current_chunk + separator + part
                if current_chunk
                else part
            )

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it has content
                if current_chunk.strip():
                    merged_chunks.append(current_chunk.strip())

                # If the part itself is too large, recurse with finer separator
                if len(part) > self.chunk_size:
                    sub_chunks = self._recursive_split(part, remaining_separators)
                    merged_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        # Don't forget the last chunk
        if current_chunk.strip():
            merged_chunks.append(current_chunk.strip())

        return merged_chunks

    def _hard_split(self, text: str) -> List[str]:
        """Last-resort character-level splitting with overlap.

        Args:
            text: Text to split by character count.

        Returns:
            List of text chunks.
        """
        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        return [c for c in chunks if c]


# =============================================================================
# Strategy 3: Sliding Window Chunker
# =============================================================================

class SlidingWindowChunker(BaseChunker):
    """Fixed-size sliding window chunker with configurable overlap.

    The simplest chunking strategy: moves a fixed-size window across the text
    with a configurable overlap.  Good as a baseline for comparison.

    Overlap ensures that information at chunk boundaries isn't lost.
    A typical overlap of 50 chars is ~10 words, enough to preserve context.
    """

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a document using a sliding window.

        Args:
            document: The Document to chunk.

        Returns:
            List of Chunk objects.
        """
        text = document.text
        chunks: List[Chunk] = []
        chunk_index = 0

        if not text.strip():
            return []

        # Effective step size = chunk_size - overlap
        step = self.chunk_size - self.chunk_overlap
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                # Try to detect article number in this window
                article_match = ARTICLE_PATTERN.search(chunk_text)
                article_number = article_match.group(1) if article_match else ""

                chunks.append(self._create_chunk(
                    text=chunk_text,
                    document=document,
                    chunk_index=chunk_index,
                    article_number=article_number,
                ))
                chunk_index += 1

            start += step

            # Safety check: if step is somehow 0, break to avoid infinite loop
            if step <= 0:
                logger.error("Sliding window step is <= 0! Breaking.")
                break

        logger.info(
            "SlidingWindowChunker: '%s' → %d chunks (window=%d, overlap=%d)",
            document.metadata.get("source", "unknown"),
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )

        return chunks


# =============================================================================
# Factory Function
# =============================================================================

def chunk_documents(
    documents: List[Document],
    strategy: str = "article",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """Split a list of Documents into Chunks using the specified strategy.

    This is the main entry point for chunking.  It selects the appropriate
    chunker class and applies it to each document.

    Args:
        documents: List of preprocessed Document objects.
        strategy: Chunking strategy — one of 'article', 'recursive', 'sliding'.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of all Chunk objects from all documents.

    Raises:
        ValueError: If strategy is not recognised.

    Example:
        >>> from deliverable_1.src.chunker import chunk_documents
        >>> chunks = chunk_documents(docs, strategy='article', chunk_size=512)
        >>> print(f"Created {len(chunks)} chunks")
    """
    # Strategy selection — maps string names to chunker classes
    strategy_map = {
        "article": ArticleBasedChunker,
        "recursive": RecursiveChunker,
        "sliding": SlidingWindowChunker,
    }

    strategy_lower = strategy.lower()
    if strategy_lower not in strategy_map:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Choose from: {list(strategy_map.keys())}"
        )

    # Instantiate the chosen chunker
    chunker = strategy_map[strategy_lower](
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    logger.info(
        "Chunking %d documents with '%s' strategy (size=%d, overlap=%d)...",
        len(documents), strategy_lower, chunk_size, chunk_overlap,
    )

    all_chunks: List[Chunk] = []

    for doc in documents:
        try:
            doc_chunks = chunker.chunk_document(doc)
            all_chunks.extend(doc_chunks)
        except Exception as exc:
            logger.error(
                "Failed to chunk document '%s': %s — skipping.",
                doc.metadata.get("source", "unknown"),
                exc,
            )
            continue

    logger.info(
        "📦 Chunking complete: %d documents → %d chunks.",
        len(documents), len(all_chunks),
    )

    return all_chunks


# =============================================================================
# CLI Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Chunker — Smoke Test")
    print("=" * 60)

    sample_text = """UAE FEDERAL PENAL CODE
Federal Decree-Law No. 31 of 2021

PART ONE: GENERAL PROVISIONS

CHAPTER 1: Scope and Application

Article 1:
This Law shall apply to all crimes committed within the territory of the United Arab Emirates,
including its territorial waters and airspace.

Article 2:
The provisions of this Law shall apply to any person who commits, within the territory of the
State, a crime that is punishable under the provisions hereof.

Article 3:
The provisions of this Law shall also apply to crimes committed outside the State in the
following cases: (a) If the crime is committed on board a UAE-registered aircraft or vessel.
(b) If the perpetrator is a UAE national and the act constitutes a crime under both this Law
and the law of the country where it was committed. This article ensures that UAE citizens
remain subject to UAE criminal jurisdiction even when abroad, provided dual criminality
exists. The rationale is to prevent UAE nationals from evading justice by committing crimes
in jurisdictions with weaker legal frameworks.

CHAPTER 2: Criminal Responsibility

Article 4:
Criminal responsibility is personal. No person shall be punished for an offence committed by
another person. Each individual bears responsibility only for their own criminal acts.
"""

    doc = Document(
        text=sample_text,
        metadata={"source": "test_penal_code.txt", "law_name": "Test Penal Code"},
        doc_id="test_001",
    )

    # Test each strategy
    for strategy in ["article", "recursive", "sliding"]:
        print(f"\n{'─' * 40}")
        print(f"Strategy: {strategy}")
        print(f"{'─' * 40}")

        chunks = chunk_documents([doc], strategy=strategy, chunk_size=300, chunk_overlap=50)

        for chunk in chunks:
            print(f"\n  Chunk {chunk.metadata['chunk_index']} "
                  f"(Article {chunk.metadata.get('article_number', 'N/A')}, "
                  f"{chunk.metadata['chunk_size']} chars):")
            print(f"    {chunk.text[:80]}...")

    print(f"\n✅ Smoke test complete!")
