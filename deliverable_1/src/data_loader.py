"""
Data Loader Module — UAE Legal RAG System ("Better Call Saul AI")
=================================================================

Responsible for loading raw legal documents from multiple sources into the
pipeline's standard ``Document`` data class.

Supported sources:
    1. **Plain text files** (.txt) — our primary format for UAE law texts
    2. **PDF files** (.pdf) — scanned/digital legal documents via pdfplumber
    3. **HuggingFace datasets** — e.g. 'obadabaq/structured-uae-laws'

Design Notes:
    - Each loader function returns a list of Document objects with populated
      metadata (source filename, law_name, etc.)
    - The law_name is inferred from the filename using the UAE_LAW_CATEGORIES
      mapping (e.g. "uae_penal_code.txt" → "UAE Federal Penal Code ...")
    - All functions are stateless and side-effect-free (pure data loading)

Usage:
    from deliverable_1.src.data_loader import load_all_documents
    docs = load_all_documents()
"""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path fix — allow imports from the project root regardless of working dir
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import DATA_RAW_DIR                          # noqa: E402
from shared.utils import Document, setup_logging                # noqa: E402
from shared.constants import UAE_LAW_CATEGORIES, SUPPORTED_FILE_TYPES  # noqa: E402

logger = setup_logging(__name__)


# =============================================================================
# Helper: Infer law name from filename
# =============================================================================

def _infer_law_name(filepath: Path) -> str:
    """Infer the official law name from the filename.

    We match the filename stem against keys in UAE_LAW_CATEGORIES.
    For example:
        "uae_penal_code.txt" → stem "uae_penal_code"
        Check if any category key is a substring → "penal_code" matches.

    Args:
        filepath: Path to the file.

    Returns:
        The official law name string, or the filename stem as fallback.
    """
    stem = filepath.stem.lower()

    for key, law_name in UAE_LAW_CATEGORIES.items():
        # Check if the category key appears in the filename
        # e.g. "penal_code" in "uae_penal_code"
        if key in stem:
            return law_name

    # Fallback: use the filename itself as the law name
    logger.warning(
        "Could not infer law name for '%s' — using filename as law_name.",
        filepath.name,
    )
    return stem.replace("_", " ").title()


def _generate_doc_id(filepath: Path, index: int = 0) -> str:
    """Generate a deterministic document ID from file path and index.

    Uses MD5 hashing to create a short, unique ID that is reproducible
    across runs (important for idempotent re-ingestion).

    Args:
        filepath: Source file path.
        index: Document index within the file (for multi-doc files).

    Returns:
        A hex string like "doc_a3f2b1c9".
    """
    content = f"{filepath.resolve()}:{index}"
    hash_hex = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"doc_{hash_hex}"


# =============================================================================
# Loader: Plain Text Files
# =============================================================================

def load_text_file(filepath: Path) -> List[Document]:
    """Load a plain text file into a list of Document objects.

    Each text file becomes a single Document. The full file content is read
    as one text blob — the chunker will split it later.

    Why one Document per file?
        Legal texts are structured (Part → Chapter → Article), and we want
        the preprocessor and chunker to see the full structure before splitting.

    Args:
        filepath: Path to the .txt file.

    Returns:
        List containing a single Document (or empty list on error).

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        raise FileNotFoundError(f"Text file not found: {filepath}")

    if not filepath.suffix.lower() == '.txt':
        logger.warning("Expected .txt file, got '%s' — loading anyway.", filepath.suffix)

    try:
        # Try UTF-8 first (standard), fall back to latin-1 (handles most edge cases)
        try:
            text = filepath.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed for %s — trying latin-1.", filepath.name)
            text = filepath.read_text(encoding='latin-1')

        if not text.strip():
            logger.warning("File is empty: %s", filepath.name)
            return []

        law_name = _infer_law_name(filepath)
        doc_id = _generate_doc_id(filepath)

        doc = Document(
            text=text,
            metadata={
                "source": filepath.name,
                "law_name": law_name,
                "article_number": "",       # Will be filled during preprocessing
                "chapter": "",
                "section": "",
            },
            doc_id=doc_id,
        )

        logger.info(
            "Loaded text file: %s (%d chars, law='%s')",
            filepath.name, len(text), law_name,
        )
        return [doc]

    except Exception as exc:
        logger.error("Failed to load text file '%s': %s", filepath.name, exc)
        raise


# =============================================================================
# Loader: PDF Files
# =============================================================================

def load_pdf_file(filepath: Path) -> List[Document]:
    """Load a PDF file into a list of Document objects using pdfplumber.

    Why pdfplumber over PyPDF2?
        pdfplumber is better at extracting text from complex layouts (tables,
        multi-column) which is common in official UAE legal documents.

    Args:
        filepath: Path to the .pdf file.

    Returns:
        List containing a single Document with all pages concatenated.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ImportError: If pdfplumber is not installed.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error("PDF file not found: %s", filepath)
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    try:
        import pdfplumber
    except ImportError:
        logger.error(
            "pdfplumber is not installed. Run: pip install pdfplumber"
        )
        raise ImportError(
            "pdfplumber is required for PDF loading. "
            "Install it with: pip install pdfplumber"
        )

    try:
        pages_text = []

        with pdfplumber.open(filepath) as pdf:
            logger.info("Opened PDF: %s (%d pages)", filepath.name, len(pdf.pages))

            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
                else:
                    logger.debug("Page %d of %s yielded no text.", i + 1, filepath.name)

        if not pages_text:
            logger.warning("No text extracted from PDF: %s", filepath.name)
            return []

        # Join all pages with double newline (preserves page boundaries)
        full_text = "\n\n".join(pages_text)
        law_name = _infer_law_name(filepath)
        doc_id = _generate_doc_id(filepath)

        doc = Document(
            text=full_text,
            metadata={
                "source": filepath.name,
                "law_name": law_name,
                "article_number": "",
                "chapter": "",
                "section": "",
            },
            doc_id=doc_id,
        )

        logger.info(
            "Loaded PDF: %s (%d chars from %d pages)",
            filepath.name, len(full_text), len(pages_text),
        )
        return [doc]

    except Exception as exc:
        logger.error("Failed to load PDF '%s': %s", filepath.name, exc)
        raise


# =============================================================================
# Loader: HuggingFace Dataset
# =============================================================================

def load_huggingface_dataset(
    dataset_name: str = "obadabaq/structured-uae-laws",
) -> List[Document]:
    """Load UAE legal documents from a HuggingFace dataset.

    This is useful for supplementing our local text files with additional
    structured legal data.  The dataset is expected to have 'text' and
    optional metadata columns.

    Args:
        dataset_name: HuggingFace dataset identifier.

    Returns:
        List of Document objects, one per dataset row.

    Raises:
        ImportError: If the 'datasets' library is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' library is not installed. "
            "Run: pip install datasets"
        )
        raise ImportError(
            "HuggingFace 'datasets' library is required. "
            "Install it with: pip install datasets"
        )

    try:
        logger.info("Loading HuggingFace dataset: %s", dataset_name)
        dataset = load_dataset(dataset_name, split="train")

        documents = []
        for idx, row in enumerate(dataset):
            # Try common column names for the text field
            text = (
                row.get("text")
                or row.get("content")
                or row.get("article_text")
                or ""
            )

            if not text.strip():
                continue

            # Extract metadata from available columns
            metadata = {
                "source": f"huggingface:{dataset_name}",
                "law_name": row.get("law_name", row.get("title", "")),
                "article_number": str(row.get("article_number", row.get("article_no", ""))),
                "chapter": row.get("chapter", row.get("chapter_name", "")),
                "section": row.get("section", row.get("section_name", "")),
            }

            doc = Document(
                text=text,
                metadata=metadata,
                doc_id=f"hf_{dataset_name.replace('/', '_')}_{idx}",
            )
            documents.append(doc)

        logger.info(
            "Loaded %d documents from HuggingFace dataset '%s'.",
            len(documents), dataset_name,
        )
        return documents

    except Exception as exc:
        logger.error("Failed to load HuggingFace dataset '%s': %s", dataset_name, exc)
        raise


# =============================================================================
# Master Loader: Load All Documents from a Directory
# =============================================================================

def load_all_documents(data_dir: Optional[Path] = None) -> List[Document]:
    """Load all supported documents from a directory.

    Scans the directory for files with supported extensions (.txt, .pdf, .json)
    and loads each one using the appropriate loader function.

    This is the main entry point for the ingestion pipeline.

    Args:
        data_dir: Directory to scan. Defaults to DATA_RAW_DIR from config.

    Returns:
        List of all loaded Document objects, sorted by source filename.

    Example:
        >>> docs = load_all_documents()
        >>> print(f"Loaded {len(docs)} documents")
        Loaded 5 documents
    """
    data_dir = Path(data_dir) if data_dir else DATA_RAW_DIR

    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        logger.info("Creating directory: %s", data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        return []

    all_documents: List[Document] = []

    # Collect all supported files, sorted for deterministic ordering
    files = sorted([
        f for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FILE_TYPES
    ])

    if not files:
        logger.warning(
            "No supported files found in %s. "
            "Supported types: %s",
            data_dir, SUPPORTED_FILE_TYPES,
        )
        return []

    logger.info("Found %d files to load from %s", len(files), data_dir)

    # Dispatch each file to the appropriate loader
    for filepath in files:
        try:
            if filepath.suffix.lower() == '.txt':
                docs = load_text_file(filepath)
            elif filepath.suffix.lower() == '.pdf':
                docs = load_pdf_file(filepath)
            elif filepath.suffix.lower() in ('.json', '.csv'):
                # JSON/CSV files could be HuggingFace exports — treat as text
                logger.info("Loading %s as text file.", filepath.name)
                docs = load_text_file(filepath)
            else:
                logger.debug("Skipping unsupported file: %s", filepath.name)
                continue

            all_documents.extend(docs)

        except Exception as exc:
            # Log error but continue with remaining files
            # (one bad file shouldn't stop the entire pipeline)
            logger.error(
                "Failed to load '%s': %s — skipping.",
                filepath.name, exc,
            )
            continue

    logger.info(
        "📚 Loaded %d documents total from %d files.",
        len(all_documents), len(files),
    )

    return all_documents


# =============================================================================
# CLI Smoke Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Data Loader — Smoke Test")
    print("=" * 60)

    docs = load_all_documents()

    for doc in docs:
        print(f"\n📄 {doc.metadata['source']}")
        print(f"   Law: {doc.metadata['law_name']}")
        print(f"   ID:  {doc.doc_id}")
        print(f"   Length: {len(doc.text)} chars")
        print(f"   Preview: {doc.text[:100]}...")

    print(f"\n✅ Total documents loaded: {len(docs)}")
