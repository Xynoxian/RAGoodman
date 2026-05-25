"""
Advanced Chunking Strategies — Semantic & Hierarchical
=======================================================

Standard fixed-size chunking (Deliverable 1) splits text at arbitrary
character boundaries, which can break legal articles in the middle of a
sentence.  This module provides two smarter alternatives:

1. **SemanticChunker** — Uses embedding similarity between consecutive
   sentences to find natural "break points".  Sentences that are
   semantically similar stay together, producing more coherent chunks.

2. **HierarchicalChunker** — Exploits the known structure of UAE legal
   documents (Part → Chapter → Article → Clause) to create a parent-child
   chunk hierarchy.  Each chunk knows its position in the law's structure,
   enabling hierarchical retrieval (e.g., retrieve an article's clause,
   but also know which chapter it belongs to).

Both chunkers implement the same ``chunk(documents) -> list[Chunk]``
interface, making them drop-in replacements for the basic chunker.

Usage::

    from deliverable_2.src.advanced_chunker import SemanticChunker, HierarchicalChunker
    from deliverable_1.src.embedder import EmbeddingModel

    embedder = EmbeddingModel()
    semantic = SemanticChunker(embedder)
    chunks = semantic.chunk(documents)

    hierarchical = HierarchicalChunker()
    chunks = hierarchical.chunk(documents)
"""

import re
import sys
import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from shared.utils import Document, Chunk, setup_logging

logger = setup_logging(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: sentence splitting
# ═══════════════════════════════════════════════════════════════════════════

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex-based heuristics.

    Handles common legal text patterns:
        - Period/question/exclamation followed by space + uppercase
        - Numbered lists (e.g., "1.", "a)")
        - Keeps short fragments joined to avoid single-word "sentences"

    Args:
        text: Raw text to split.

    Returns:
        List of sentence strings, stripped of extra whitespace.
    """
    # Split on sentence-ending punctuation followed by whitespace
    # This regex handles periods, question marks, exclamation marks
    # while being careful not to split on abbreviations like "Art." or "No."
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)

    # Filter out empty strings and strip whitespace
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    # Merge very short sentences (< 20 chars) with the next sentence
    # to avoid creating tiny, meaningless chunks
    merged: List[str] = []
    buffer = ""
    for sent in sentences:
        if buffer:
            buffer = buffer + " " + sent
            if len(buffer) >= 20:
                merged.append(buffer)
                buffer = ""
        elif len(sent) < 20 and not merged:
            # First sentence is short — buffer it
            buffer = sent
        else:
            merged.append(sent)

    # Don't lose any buffered text
    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    return merged


def _generate_chunk_id(text: str, index: int) -> str:
    """Generate a deterministic chunk ID from content hash + index.

    Args:
        text: Chunk text content.
        index: Sequential chunk index.

    Returns:
        A short hex string like 'sem_a3f2b1c0_003'.
    """
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"sem_{content_hash}_{index:03d}"


# ═══════════════════════════════════════════════════════════════════════════
# Semantic Chunker
# ═══════════════════════════════════════════════════════════════════════════

class SemanticChunker:
    """Chunks documents by grouping semantically similar sentences.

    **How it works:**

    1. Split the document into individual sentences.
    2. Embed each sentence using the provided embedding model.
    3. Compute cosine similarity between consecutive sentence embeddings.
    4. Identify "break points" where similarity drops below a threshold —
       these are natural topic boundaries.
    5. Group sentences between break points into chunks.
    6. Optionally merge very small chunks or split very large ones.

    This produces chunks that respect semantic boundaries rather than
    arbitrary character counts, leading to more coherent retrieval.

    Attributes:
        embedder: Embedding model for computing sentence vectors.
        similarity_threshold: Cosine similarity below which a break is inserted.
        max_chunk_size: Maximum characters per chunk (hard limit).
        min_chunk_size: Minimum characters per chunk (merge threshold).
    """

    def __init__(
        self,
        embedder,
        similarity_threshold: float = 0.5,
        max_chunk_size: int = CHUNK_SIZE * 2,
        min_chunk_size: int = 100,
    ) -> None:
        """Initialise the SemanticChunker.

        Args:
            embedder: An object with ``embed_text(str) -> list[float]`` and
                ``embed_texts(list[str]) -> list[list[float]]`` methods.
                Typically an ``EmbeddingModel`` instance from deliverable_1.
            similarity_threshold: Cosine similarity threshold for detecting
                break points.  Lower values = fewer, larger chunks.  Higher
                values = more, smaller chunks.  Default 0.5 works well for
                legal text.
            max_chunk_size: Maximum allowed chunk size in characters.
            min_chunk_size: Chunks below this size are merged with neighbours.
        """
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

        logger.info(
            "SemanticChunker initialised — threshold=%.2f, max_size=%d, min_size=%d",
            similarity_threshold, max_chunk_size, min_chunk_size,
        )

    # ------------------------------------------------------------------
    # Cosine similarity helper
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec_a: First embedding vector.
            vec_b: Second embedding vector.

        Returns:
            Cosine similarity in [-1, 1].  Higher = more similar.
        """
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        # Guard against zero-norm vectors
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # Find semantic break points
    # ------------------------------------------------------------------
    def _find_break_points(
        self, sentences: List[str]
    ) -> List[int]:
        """Identify indices where the topic shifts between sentences.

        Embeds all sentences, computes pairwise cosine similarity between
        consecutive sentences, and marks break points where similarity
        drops below the threshold.

        Args:
            sentences: List of sentence strings.

        Returns:
            List of indices (into ``sentences``) where breaks should occur.
            A break at index ``i`` means a new chunk starts at sentence ``i``.
        """
        if len(sentences) <= 1:
            return []

        # Batch-embed all sentences for efficiency
        logger.debug("Embedding %d sentences for break-point detection…", len(sentences))
        embeddings = self.embedder.embed_texts(sentences)

        # Compute consecutive similarities
        similarities: List[float] = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find break points: where similarity drops below threshold
        break_points: List[int] = []
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                break_points.append(i + 1)  # break BEFORE sentence i+1

        logger.debug(
            "Found %d break points among %d sentences (threshold=%.2f)",
            len(break_points), len(sentences), self.similarity_threshold,
        )
        return break_points

    # ------------------------------------------------------------------
    # Group sentences into chunks
    # ------------------------------------------------------------------
    def _group_sentences(
        self, sentences: List[str], break_points: List[int]
    ) -> List[str]:
        """Group sentences into chunk texts based on break points.

        Also handles post-processing:
        - Merges chunks that are too small (< min_chunk_size) with neighbours
        - Splits chunks that are too large (> max_chunk_size)

        Args:
            sentences: Original sentence list.
            break_points: Indices where breaks occur.

        Returns:
            List of chunk text strings.
        """
        # Create initial groups
        groups: List[List[str]] = []
        prev_break = 0
        for bp in break_points:
            group = sentences[prev_break:bp]
            if group:
                groups.append(group)
            prev_break = bp
        # Don't forget the last group
        last_group = sentences[prev_break:]
        if last_group:
            groups.append(last_group)

        # Join sentences within each group
        chunk_texts = [" ".join(group) for group in groups]

        # Post-processing: merge small chunks, split large ones
        chunk_texts = self._merge_small_chunks(chunk_texts)
        chunk_texts = self._split_large_chunks(chunk_texts)

        return chunk_texts

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks smaller than min_chunk_size with their neighbour.

        Args:
            chunks: List of chunk text strings.

        Returns:
            List with small chunks merged into neighbours.
        """
        if not chunks:
            return chunks

        merged: List[str] = [chunks[0]]
        for chunk in chunks[1:]:
            if len(merged[-1]) < self.min_chunk_size:
                # Merge with previous chunk
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)

        # Check if the last chunk is too small
        if len(merged) > 1 and len(merged[-1]) < self.min_chunk_size:
            merged[-2] = merged[-2] + " " + merged[-1]
            merged.pop()

        return merged

    def _split_large_chunks(self, chunks: List[str]) -> List[str]:
        """Split chunks larger than max_chunk_size into smaller pieces.

        Uses sentence boundaries within the chunk for splitting.

        Args:
            chunks: List of chunk text strings.

        Returns:
            List with large chunks split at sentence boundaries.
        """
        result: List[str] = []
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                result.append(chunk)
                continue

            # Re-split into sentences and re-group by size
            sentences = _split_sentences(chunk)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) + 1 > self.max_chunk_size and current:
                    result.append(current.strip())
                    current = sent
                else:
                    current = (current + " " + sent).strip()
            if current:
                result.append(current)

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Chunk a list of documents using semantic similarity.

        This is the main entry point.  For each document:
        1. Split into sentences
        2. Find semantic break points via embedding similarity
        3. Group sentences into coherent chunks
        4. Create Chunk objects with inherited metadata

        Args:
            documents: List of Document objects to chunk.

        Returns:
            List of Chunk objects with metadata inherited from the source
            documents plus ``chunk_index``, ``chunk_size``, and
            ``chunking_method = 'semantic'``.
        """
        all_chunks: List[Chunk] = []
        total_idx = 0

        for doc in documents:
            logger.info(
                "Semantic chunking document: %s",
                doc.metadata.get("law_name", doc.doc_id or "unknown"),
            )

            # Step 1: Split into sentences
            sentences = _split_sentences(doc.text)
            if not sentences:
                logger.warning("No sentences found in document %s", doc.doc_id)
                continue

            # Step 2: Find semantic break points
            break_points = self._find_break_points(sentences)

            # Step 3: Group into chunks
            chunk_texts = self._group_sentences(sentences, break_points)

            # Step 4: Create Chunk objects
            for i, text in enumerate(chunk_texts):
                chunk_meta = {
                    **doc.metadata,
                    "chunk_index": i,
                    "chunk_size": len(text),
                    "chunking_method": "semantic",
                    "doc_id": doc.doc_id,
                }
                chunk = Chunk(
                    text=text,
                    metadata=chunk_meta,
                    chunk_id=_generate_chunk_id(text, total_idx),
                )
                all_chunks.append(chunk)
                total_idx += 1

        logger.info(
            "SemanticChunker produced %d chunks from %d documents.",
            len(all_chunks), len(documents),
        )
        return all_chunks


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchical Chunker
# ═══════════════════════════════════════════════════════════════════════════

class HierarchicalChunker:
    """Chunks UAE legal documents respecting their structural hierarchy.

    **UAE legal document structure:**

    Most UAE laws follow this hierarchy::

        Part (الباب)
        └── Chapter (الفصل)
            └── Article (المادة)
                └── Clause (1., 2., a., b.)

    This chunker:
    1. Detects structural headings via regex patterns.
    2. Creates chunks at the Article level (the natural unit of legal reference).
    3. Stores the full hierarchy path in metadata so the retriever knows
       the context (e.g., "Part 3 > Chapter 2 > Article 45").
    4. Optionally creates parent-level (chapter/part) summary chunks for
       broader retrieval.

    Attributes:
        include_parent_chunks: If True, create summary chunks at Part/Chapter
            level in addition to Article-level chunks.
        max_chunk_size: Maximum characters per chunk.
    """

    # Regex patterns for UAE legal document structure markers
    # These patterns detect Arabic and English headings
    PART_PATTERN = re.compile(
        r"(?:Part|PART|الباب)\s*(\d+|[IVXLC]+)",
        re.IGNORECASE | re.UNICODE,
    )
    CHAPTER_PATTERN = re.compile(
        r"(?:Chapter|CHAPTER|الفصل)\s*(\d+|[IVXLC]+)",
        re.IGNORECASE | re.UNICODE,
    )
    ARTICLE_PATTERN = re.compile(
        r"(?:Article|ARTICLE|المادة)\s*\(?(\d+)\)?",
        re.IGNORECASE | re.UNICODE,
    )
    SECTION_PATTERN = re.compile(
        r"(?:Section|SECTION|القسم)\s*(\d+|[IVXLC]+)",
        re.IGNORECASE | re.UNICODE,
    )

    def __init__(
        self,
        include_parent_chunks: bool = True,
        max_chunk_size: int = CHUNK_SIZE * 3,
    ) -> None:
        """Initialise the HierarchicalChunker.

        Args:
            include_parent_chunks: Whether to also create higher-level
                (Part/Chapter) summary chunks for broad retrieval.
            max_chunk_size: Maximum chunk size in characters.  Articles
                exceeding this are split into sub-chunks.
        """
        self.include_parent_chunks = include_parent_chunks
        self.max_chunk_size = max_chunk_size
        logger.info(
            "HierarchicalChunker initialised — parent_chunks=%s, max_size=%d",
            include_parent_chunks, max_chunk_size,
        )

    # ------------------------------------------------------------------
    # Structure detection
    # ------------------------------------------------------------------
    def _detect_structure(self, text: str) -> List[dict]:
        """Parse the document text and identify structural elements.

        Scans the text line by line, detecting Part/Chapter/Section/Article
        headings and tracking the current hierarchy context.

        Args:
            text: Full document text.

        Returns:
            List of dicts, each representing a structural element::

                {
                    "type": "article",      # part, chapter, section, article
                    "number": "45",
                    "title": "Article 45",  # The heading line
                    "text": "...",          # Content until next heading
                    "hierarchy": {          # Full path context
                        "part": "Part 3",
                        "chapter": "Chapter 2",
                        "section": "",
                        "article": "45",
                    }
                }
        """
        lines = text.split("\n")
        elements: List[dict] = []

        # Current hierarchy context — tracks where we are in the document
        current_part = ""
        current_chapter = ""
        current_section = ""
        current_article = ""

        # Accumulate text for the current element
        current_element: Optional[dict] = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_element:
                    current_element["text"] += "\n"
                continue

            # Check for Part heading
            part_match = self.PART_PATTERN.search(stripped)
            if part_match and len(stripped) < 100:  # Headings are short
                # Save previous element if any
                if current_element:
                    elements.append(current_element)

                current_part = f"Part {part_match.group(1)}"
                current_chapter = ""
                current_section = ""
                current_article = ""

                current_element = {
                    "type": "part",
                    "number": part_match.group(1),
                    "title": stripped,
                    "text": "",
                    "hierarchy": {
                        "part": current_part,
                        "chapter": "",
                        "section": "",
                        "article": "",
                    },
                }
                continue

            # Check for Chapter heading
            chapter_match = self.CHAPTER_PATTERN.search(stripped)
            if chapter_match and len(stripped) < 100:
                if current_element:
                    elements.append(current_element)

                current_chapter = f"Chapter {chapter_match.group(1)}"
                current_section = ""
                current_article = ""

                current_element = {
                    "type": "chapter",
                    "number": chapter_match.group(1),
                    "title": stripped,
                    "text": "",
                    "hierarchy": {
                        "part": current_part,
                        "chapter": current_chapter,
                        "section": "",
                        "article": "",
                    },
                }
                continue

            # Check for Section heading
            section_match = self.SECTION_PATTERN.search(stripped)
            if section_match and len(stripped) < 100:
                if current_element:
                    elements.append(current_element)

                current_section = f"Section {section_match.group(1)}"
                current_article = ""

                current_element = {
                    "type": "section",
                    "number": section_match.group(1),
                    "title": stripped,
                    "text": "",
                    "hierarchy": {
                        "part": current_part,
                        "chapter": current_chapter,
                        "section": current_section,
                        "article": "",
                    },
                }
                continue

            # Check for Article heading
            article_match = self.ARTICLE_PATTERN.search(stripped)
            if article_match and len(stripped) < 150:
                if current_element:
                    elements.append(current_element)

                current_article = article_match.group(1)

                current_element = {
                    "type": "article",
                    "number": current_article,
                    "title": stripped,
                    "text": "",
                    "hierarchy": {
                        "part": current_part,
                        "chapter": current_chapter,
                        "section": current_section,
                        "article": current_article,
                    },
                }
                continue

            # Regular content line — append to current element
            if current_element:
                current_element["text"] += stripped + "\n"
            # else: text before any heading — skip or create a preamble element

        # Don't forget the last element
        if current_element:
            elements.append(current_element)

        logger.debug(
            "Detected %d structural elements (%d parts, %d chapters, %d articles)",
            len(elements),
            sum(1 for e in elements if e["type"] == "part"),
            sum(1 for e in elements if e["type"] == "chapter"),
            sum(1 for e in elements if e["type"] == "article"),
        )
        return elements

    # ------------------------------------------------------------------
    # Split oversized articles
    # ------------------------------------------------------------------
    def _split_article_text(self, text: str) -> List[str]:
        """Split an article's text into sub-chunks if it exceeds max_chunk_size.

        Splits at clause boundaries (numbered items like "1.", "2.", "(a)")
        or, failing that, at sentence boundaries.

        Args:
            text: Article text content.

        Returns:
            List of text sub-chunks.
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        # Try splitting at clause boundaries first (numbered items)
        clause_pattern = re.compile(r'\n(?=\d+[.)]\s|\([a-z]\)\s)')
        clauses = clause_pattern.split(text)

        if len(clauses) > 1:
            # Re-group clauses to fit within max_chunk_size
            result: List[str] = []
            current = ""
            for clause in clauses:
                if len(current) + len(clause) + 1 > self.max_chunk_size and current:
                    result.append(current.strip())
                    current = clause
                else:
                    current = (current + "\n" + clause).strip()
            if current:
                result.append(current)
            return result

        # Fallback: split at sentence boundaries
        sentences = _split_sentences(text)
        result = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 > self.max_chunk_size and current:
                result.append(current.strip())
                current = sent
            else:
                current = (current + " " + sent).strip()
        if current:
            result.append(current)

        return result if result else [text]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """Chunk documents using hierarchical structure detection.

        For each document:
        1. Detect structural elements (Part/Chapter/Article).
        2. Create a chunk for each Article (the core retrieval unit).
        3. Optionally create parent-level chunks for Parts and Chapters.
        4. Include full hierarchy path in metadata for context.

        Args:
            documents: List of Document objects to chunk.

        Returns:
            List of Chunk objects with rich hierarchical metadata including:
            - ``chunk_index``, ``chunk_size``, ``chunking_method``
            - ``hierarchy_level``: 'part', 'chapter', 'section', or 'article'
            - ``hierarchy_path``: e.g., "Part 3 > Chapter 2 > Article 45"
            - ``parent_part``, ``parent_chapter``, ``parent_section``
            - ``article_number`` (for article-level chunks)
        """
        all_chunks: List[Chunk] = []
        total_idx = 0

        for doc in documents:
            law_name = doc.metadata.get("law_name", doc.doc_id or "unknown")
            logger.info("Hierarchical chunking document: %s", law_name)

            # Step 1: Detect structure
            elements = self._detect_structure(doc.text)

            if not elements:
                # Fallback: if no structure detected, treat entire doc as one chunk
                logger.warning(
                    "No structure detected in '%s' — creating single chunk.", law_name
                )
                chunk = Chunk(
                    text=doc.text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": 0,
                        "chunk_size": len(doc.text),
                        "chunking_method": "hierarchical",
                        "hierarchy_level": "document",
                        "hierarchy_path": law_name,
                    },
                    chunk_id=f"hier_{total_idx:04d}",
                )
                all_chunks.append(chunk)
                total_idx += 1
                continue

            # Step 2: Create chunks from structural elements
            for elem in elements:
                # Build hierarchy path string
                path_parts = [
                    v for v in [
                        elem["hierarchy"]["part"],
                        elem["hierarchy"]["chapter"],
                        elem["hierarchy"]["section"],
                        f"Article {elem['hierarchy']['article']}"
                        if elem["hierarchy"]["article"] else "",
                    ] if v
                ]
                hierarchy_path = " > ".join(path_parts) if path_parts else law_name

                # Full text = heading + content
                full_text = (elem["title"] + "\n" + elem["text"]).strip()

                # Skip empty elements
                if len(full_text) < 10:
                    continue

                # For articles: always create chunks
                # For parts/chapters: only if include_parent_chunks is True
                if elem["type"] in ("part", "chapter", "section"):
                    if not self.include_parent_chunks:
                        continue
                    # Create a summary-level parent chunk (just the heading area)
                    # Truncate to avoid massive parent chunks
                    parent_text = full_text[:self.max_chunk_size]
                    chunk_meta = {
                        **doc.metadata,
                        "chunk_index": total_idx,
                        "chunk_size": len(parent_text),
                        "chunking_method": "hierarchical",
                        "hierarchy_level": elem["type"],
                        "hierarchy_path": hierarchy_path,
                        "parent_part": elem["hierarchy"]["part"],
                        "parent_chapter": elem["hierarchy"]["chapter"],
                        "parent_section": elem["hierarchy"]["section"],
                        "article_number": "",
                        "doc_id": doc.doc_id,
                    }
                    chunk = Chunk(
                        text=parent_text,
                        metadata=chunk_meta,
                        chunk_id=f"hier_{elem['type']}_{total_idx:04d}",
                    )
                    all_chunks.append(chunk)
                    total_idx += 1
                else:
                    # Article-level chunk — the primary retrieval unit
                    sub_texts = self._split_article_text(full_text)
                    for sub_idx, sub_text in enumerate(sub_texts):
                        chunk_meta = {
                            **doc.metadata,
                            "chunk_index": total_idx,
                            "chunk_size": len(sub_text),
                            "chunking_method": "hierarchical",
                            "hierarchy_level": "article",
                            "hierarchy_path": hierarchy_path,
                            "parent_part": elem["hierarchy"]["part"],
                            "parent_chapter": elem["hierarchy"]["chapter"],
                            "parent_section": elem["hierarchy"]["section"],
                            "article_number": elem["number"],
                            "sub_chunk_index": sub_idx if len(sub_texts) > 1 else -1,
                            "doc_id": doc.doc_id,
                        }
                        chunk = Chunk(
                            text=sub_text,
                            metadata=chunk_meta,
                            chunk_id=f"hier_art{elem['number']}_{total_idx:04d}",
                        )
                        all_chunks.append(chunk)
                        total_idx += 1

        logger.info(
            "HierarchicalChunker produced %d chunks from %d documents.",
            len(all_chunks), len(documents),
        )
        return all_chunks


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Chunker — Smoke Test")
    print("=" * 60)

    # Create a sample legal document
    sample_doc = Document(
        text="""Part 1 - General Provisions
Chapter 1 - Definitions
Article 1
This law shall be known as the UAE Labour Law. It governs the relationship between employers and employees in the private sector.
Article 2
The provisions of this law shall apply to all employers and employees in the UAE. Government employees are excluded from this law.
Chapter 2 - Employment Contracts
Article 3
An employment contract shall be in writing and in duplicate. Each party shall retain a copy. The contract must specify the type of work, remuneration, and duration.
Article 4
The probation period shall not exceed six months. During probation, either party may terminate the contract with 14 days written notice.
Part 2 - Working Hours and Leave
Chapter 3 - Working Hours
Article 5
The maximum working hours shall be eight hours per day or forty-eight hours per week. During Ramadan, working hours shall be reduced by two hours per day.
""",
        metadata={"source": "test_law.txt", "law_name": "UAE Labour Law"},
        doc_id="test_001",
    )

    # Test HierarchicalChunker
    print("\n--- Hierarchical Chunker ---")
    hier = HierarchicalChunker(include_parent_chunks=True)
    hier_chunks = hier.chunk([sample_doc])
    for c in hier_chunks:
        print(f"  [{c.metadata.get('hierarchy_level', '?')}] "
              f"{c.metadata.get('hierarchy_path', '?')} "
              f"({c.metadata.get('chunk_size', '?')} chars)")

    print(f"\nTotal hierarchical chunks: {len(hier_chunks)}")
    print("\n✅ Smoke test completed!")
