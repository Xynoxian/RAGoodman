"""
Preprocessor Module — UAE Legal RAG System ("Better Call Saul AI")
==================================================================

Transforms raw legal text into clean, structured form ready for chunking.

Pipeline stages:
    1. ``clean_text()`` — Remove noise (extra whitespace, encoding artefacts)
    2. ``parse_legal_structure()`` — Identify Parts, Chapters, Sections, Articles
    3. ``extract_articles()`` — Pull out individual (article_number, article_text) pairs
    4. ``normalize_text()`` — Expand abbreviations, standardise formatting
    5. ``preprocess_documents()`` — Full pipeline over a list of Documents

Design Notes:
    - All functions are pure (no side effects) except logging
    - We preserve the original structure as much as possible — the chunker
      relies on Article boundaries for intelligent splitting
    - Encoding issues are common in legal PDFs (smart quotes, em-dashes, etc.)
      so we handle those explicitly

Usage:
    from deliverable_1.src.preprocessor import preprocess_documents
    cleaned_docs = preprocess_documents(raw_docs)
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Path fix — allow imports from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils import Document, setup_logging                       # noqa: E402
from shared.constants import (                                          # noqa: E402
    ARTICLE_PATTERN,
    CHAPTER_PATTERN,
    PART_PATTERN,
    SECTION_PATTERN,
    LEGAL_ABBREVIATIONS,
)

logger = setup_logging(__name__)


# =============================================================================
# Stage 1: Text Cleaning
# =============================================================================

def clean_text(text: str) -> str:
    """Remove noise from raw legal text while preserving structure.

    What we clean:
        - Unicode normalisation (NFKD → NFC) to fix encoding oddities from PDFs
        - Smart quotes → straight quotes (legal text should be plain ASCII)
        - Multiple consecutive blank lines → single blank line
        - Leading/trailing whitespace on each line
        - Non-breaking spaces → regular spaces
        - Control characters (except newline/tab)

    What we preserve:
        - Paragraph breaks (single blank lines between sections)
        - Article numbering format ("Article 1:", "Article 23:")
        - Indentation structure (important for sub-articles)

    Args:
        text: Raw text from a legal document.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Step 1: Unicode normalisation — converts composed characters to their
    # canonical form (e.g., é as single char vs e+combining accent)
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Replace smart quotes and other typographic characters
    # PDFs from government sources often use these
    replacements = {
        "\u2018": "'",   # Left single quote
        "\u2019": "'",   # Right single quote
        "\u201c": '"',   # Left double quote
        "\u201d": '"',   # Right double quote
        "\u2013": "-",   # En dash
        "\u2014": "--",  # Em dash
        "\u2026": "...", # Ellipsis
        "\u00a0": " ",   # Non-breaking space
        "\u200b": "",    # Zero-width space (common in Arabic-English PDFs)
        "\ufeff": "",    # BOM character
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Step 3: Remove control characters (keep newlines and tabs)
    text = "".join(
        char for char in text
        if char in ('\n', '\t', '\r') or not unicodedata.category(char).startswith('C')
    )

    # Step 4: Normalise whitespace within lines (but preserve newlines)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Strip leading/trailing whitespace from each line
        stripped = line.strip()
        # Collapse multiple spaces within a line to single space
        stripped = re.sub(r' {2,}', ' ', stripped)
        cleaned_lines.append(stripped)

    text = '\n'.join(cleaned_lines)

    # Step 5: Collapse multiple blank lines into at most two
    # (we keep double-newline as paragraph separator)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Step 6: Strip leading/trailing whitespace from entire document
    text = text.strip()

    return text


# =============================================================================
# Stage 2: Legal Structure Parsing
# =============================================================================

def parse_legal_structure(text: str) -> Dict[str, List[Dict]]:
    """Parse a legal text into its hierarchical structure.

    Identifies Parts, Chapters, Sections, and Articles and returns them
    in a structured dictionary.  This is useful for:
        - Understanding the document layout before chunking
        - Populating metadata fields (chapter, section) for each chunk
        - Building a table of contents for the web UI

    Args:
        text: Cleaned legal text.

    Returns:
        Dictionary with keys 'parts', 'chapters', 'sections', 'articles',
        each containing a list of dicts with 'number', 'title', 'position'.

    Example:
        >>> structure = parse_legal_structure(law_text)
        >>> print(structure['articles'][0])
        {'number': '1', 'title': 'This Law shall apply to...', 'position': 245}
    """
    structure: Dict[str, List[Dict]] = {
        "parts": [],
        "chapters": [],
        "sections": [],
        "articles": [],
    }

    # Find all Parts
    for match in PART_PATTERN.finditer(text):
        structure["parts"].append({
            "number": match.group(1),
            "title": match.group(2).strip() if match.group(2) else "",
            "position": match.start(),
        })

    # Find all Chapters
    for match in CHAPTER_PATTERN.finditer(text):
        structure["chapters"].append({
            "number": match.group(1),
            "title": match.group(2).strip() if match.group(2) else "",
            "position": match.start(),
        })

    # Find all Sections
    for match in SECTION_PATTERN.finditer(text):
        structure["sections"].append({
            "number": match.group(1),
            "title": match.group(2).strip() if match.group(2) else "",
            "position": match.start(),
        })

    # Find all Articles
    for match in ARTICLE_PATTERN.finditer(text):
        structure["articles"].append({
            "number": match.group(1),
            "title": "",  # Article title is extracted separately
            "position": match.start(),
        })

    logger.info(
        "Parsed structure: %d parts, %d chapters, %d sections, %d articles",
        len(structure["parts"]),
        len(structure["chapters"]),
        len(structure["sections"]),
        len(structure["articles"]),
    )

    return structure


# =============================================================================
# Stage 3: Article Extraction
# =============================================================================

def extract_articles(text: str) -> List[Tuple[str, str]]:
    """Extract individual articles from a legal text.

    Splits the text at "Article N:" boundaries and returns a list of
    (article_number, article_text) tuples.

    Why not just split on "Article"?
        Because article text can reference other articles (e.g.,
        "as specified in Article 45"). We use regex to find article
        *headers* specifically (at the start of a line or after a blank line).

    Args:
        text: Cleaned legal text containing Article markers.

    Returns:
        List of (article_number, article_text) tuples.
        The article_text includes the "Article N:" header line.

    Example:
        >>> articles = extract_articles(law_text)
        >>> print(articles[0])
        ('1', 'Article 1:\\nThis Law shall apply to ...')
    """
    if not text:
        return []

    # Find all article start positions using a more specific pattern:
    # "Article N:" must appear at start of line or after blank line
    article_starts = list(re.finditer(
        r'(?:^|\n)\s*Article\s+(\d+)\s*[:\.]?\s*\n?',
        text,
        re.IGNORECASE
    ))

    if not article_starts:
        logger.warning("No articles found in text (length=%d).", len(text))
        return []

    articles: List[Tuple[str, str]] = []

    for i, match in enumerate(article_starts):
        article_number = match.group(1)

        # Article text runs from this match to the start of the next article
        start_pos = match.start()
        end_pos = article_starts[i + 1].start() if i + 1 < len(article_starts) else len(text)

        article_text = text[start_pos:end_pos].strip()

        # Skip empty articles (shouldn't happen, but defensive programming)
        if article_text:
            articles.append((article_number, article_text))

    logger.info("Extracted %d articles from text.", len(articles))
    return articles


# =============================================================================
# Stage 4: Text Normalisation
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalise legal text for consistent embedding and retrieval.

    Normalisation steps:
        1. Expand common legal abbreviations (so embeddings capture full meaning)
        2. Standardise number formatting (e.g., "1,000" → "1000" for consistency)
        3. Normalise article references ("Art. 5" → "Article 5")
        4. Lowercase for consistent matching (but preserve original in metadata)

    Note: We do NOT lowercase the text because:
        - Proper nouns matter in legal context (e.g., "UAE", "Minister")
        - The embedding model handles case-insensitivity internally
        - Users see the original text in search results

    Args:
        text: Cleaned legal text.

    Returns:
        Normalised text string.
    """
    if not text:
        return ""

    # Step 1: Expand abbreviations
    # Only expand standalone abbreviations (word boundaries) to avoid
    # false positives like "AEDucation"
    for abbr, expansion in LEGAL_ABBREVIATIONS.items():
        # Use word boundary matching for safety
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b')
        # Don't expand if the text already has the full form nearby
        # (avoids "Article Article" type duplication)
        if expansion not in text:
            # Replace abbreviation but keep original for readability
            # e.g., "Art. 5" → "Article 5" (but only for "Art.")
            if abbr in ("Art.", "No.", "Ch.", "Sec.", "Pt."):
                text = pattern.sub(expansion, text)

    # Step 2: Standardise article reference format
    # "art. 5", "ART 5", "ARTICLE 5" → "Article 5"
    text = re.sub(
        r'\b(?:art\.?|article)\s*(\d+)',
        r'Article \1',
        text,
        flags=re.IGNORECASE,
    )

    # Step 3: Normalise "AED" currency references for consistency
    # "AED 10,000" → "AED 10000" (no commas in numbers)
    text = re.sub(
        r'AED\s*([\d,]+)',
        lambda m: f"AED {m.group(1).replace(',', '')}",
        text,
    )

    # Step 4: Standardise dash-separated number ranges
    # "3-5 years" is fine, but "3 - 5 years" → "3-5 years"
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', text)

    return text


# =============================================================================
# Stage 5: Full Preprocessing Pipeline
# =============================================================================

def preprocess_documents(documents: List[Document]) -> List[Document]:
    """Apply the full preprocessing pipeline to a list of Documents.

    Pipeline:
        1. Clean text (remove noise, fix encoding)
        2. Normalise text (expand abbreviations, standardise format)

    Note: We do NOT extract articles or parse structure here — that's
    the chunker's job.  The preprocessor ensures the text is clean and
    normalised so the chunker can work with reliable input.

    Args:
        documents: List of raw Document objects from the data loader.

    Returns:
        List of preprocessed Document objects (new objects, originals unchanged).
    """
    if not documents:
        logger.warning("preprocess_documents called with empty list.")
        return []

    logger.info("Preprocessing %d documents...", len(documents))
    processed: List[Document] = []

    for doc in documents:
        try:
            # Apply cleaning and normalisation sequentially
            cleaned_text = clean_text(doc.text)
            normalised_text = normalize_text(cleaned_text)

            # Parse structure to log statistics (doesn't modify the text)
            structure = parse_legal_structure(normalised_text)

            # Create a new Document with the processed text
            # (we don't modify the original — immutability is safer)
            processed_doc = Document(
                text=normalised_text,
                metadata={
                    **doc.metadata,  # Preserve all original metadata
                    # Add structure statistics for downstream use
                    "num_articles": str(len(structure["articles"])),
                    "num_chapters": str(len(structure["chapters"])),
                },
                doc_id=doc.doc_id,
            )

            original_len = len(doc.text)
            processed_len = len(normalised_text)
            reduction = (1 - processed_len / original_len) * 100 if original_len > 0 else 0

            logger.info(
                "  ✓ %s: %d→%d chars (%.1f%% reduction), %d articles found",
                doc.metadata.get("source", "unknown"),
                original_len,
                processed_len,
                reduction,
                len(structure["articles"]),
            )

            processed.append(processed_doc)

        except Exception as exc:
            logger.error(
                "Failed to preprocess document '%s': %s — skipping.",
                doc.metadata.get("source", "unknown"),
                exc,
            )
            continue

    logger.info(
        "📝 Preprocessing complete: %d/%d documents processed.",
        len(processed), len(documents),
    )

    return processed


# =============================================================================
# CLI Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Preprocessor — Smoke Test")
    print("=" * 60)

    # Create a sample document to test
    sample_text = """
    UAE FEDERAL PENAL CODE
    Federal Decree-Law No.  31 of 2021

    PART ONE: GENERAL   PROVISIONS

    CHAPTER 1: Scope and Application

    Article 1:
    This Law shall apply to crimes committed in the   territory of the UAE.
    The provisions of Art. 2 shall also apply to crimes committed abroad.

    Article 2:
    The provisions of this Law shall apply to anyone who commits a crime
    on board a UAE-registered aircraft or vessel.

    Article 3:
    Any person who commits a crime punishable by a fine of AED 10,000 or
    imprisonment for 3 - 5 years shall be subject to the penalties herein.
    """

    doc = Document(
        text=sample_text,
        metadata={"source": "test_penal_code.txt", "law_name": "Test Law"},
        doc_id="test_001",
    )

    # Run preprocessing
    processed = preprocess_documents([doc])

    if processed:
        print(f"\nOriginal length: {len(sample_text)} chars")
        print(f"Processed length: {len(processed[0].text)} chars")
        print(f"\n--- Processed text ---")
        print(processed[0].text[:500])

        # Test article extraction
        articles = extract_articles(processed[0].text)
        print(f"\n--- Extracted articles ---")
        for num, text in articles:
            print(f"Article {num}: {text[:80]}...")

    print("\n✅ Smoke test complete!")
