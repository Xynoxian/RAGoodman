"""
Constants - UAE Legal RAG System ("Better Call Saul AI")
========================================================

Project-wide constants that don't change between runs. These are distinct from
config.py values (which can be overridden via environment variables). Constants
are truly fixed — they define the structure and vocabulary of our legal domain.

Why separate from config?
    - Config = tuneable settings (chunk size, model name, API keys)
    - Constants = domain knowledge (law categories, regex patterns, disclaimers)

Usage:
    from shared.constants import UAE_LAW_CATEGORIES, ARTICLE_PATTERN
"""

import re

# =============================================================================
# UAE Law Category Mapping
# =============================================================================
# Maps short identifiers (used in filenames) to official law titles.
# This is used by the data_loader to tag documents with their law_name metadata,
# and by the web UI to display user-friendly labels.
UAE_LAW_CATEGORIES = {
    "penal_code": "UAE Federal Penal Code (Federal Decree-Law No. 31 of 2021)",
    "labour_law": "UAE Labour Law (Federal Decree-Law No. 33 of 2021)",
    "civil_code": "UAE Civil Code (Federal Law No. 5 of 1985)",
    "cybercrime_law": "UAE Cybercrime Law (Federal Decree-Law No. 34 of 2021)",
    "personal_status_law": "UAE Personal Status Law (Federal Law No. 28 of 2005)",
}

# =============================================================================
# Legal Text Parsing Patterns
# =============================================================================
# These regex patterns are used by the preprocessor and chunker to identify
# structural elements in legal text.  We compile them once here for efficiency
# (compiled regexes are ~10x faster when called repeatedly).

# Matches "Article 1", "Article 23", "ARTICLE 100" etc.
ARTICLE_PATTERN = re.compile(r'Article\s+(\d+)', re.IGNORECASE)

# Matches "Chapter 1:", "CHAPTER IV:", etc. (Roman or Arabic numerals)
CHAPTER_PATTERN = re.compile(
    r'Chapter\s+(\d+|[IVXLC]+)\s*[:.]?\s*(.*)',
    re.IGNORECASE
)

# Matches "Part One:", "PART TWO:", "Part 1:" etc.
PART_PATTERN = re.compile(
    r'Part\s+(One|Two|Three|Four|Five|Six|\d+)\s*[:.]?\s*(.*)',
    re.IGNORECASE
)

# Matches "Section 1:", "SECTION 2:" etc.
SECTION_PATTERN = re.compile(
    r'Section\s+(\d+)\s*[:.]?\s*(.*)',
    re.IGNORECASE
)

# =============================================================================
# Supported File Types
# =============================================================================
# The data_loader will only process files with these extensions.
# We support plain text (primary), PDF, JSON (HuggingFace exports), and CSV.
SUPPORTED_FILE_TYPES = ['.txt', '.pdf', '.json', '.csv']

# =============================================================================
# Legal Disclaimer
# =============================================================================
# This disclaimer is shown to users in EVERY response.  It's legally important
# because our system must NOT be mistaken for actual legal advice.  The UAE has
# strict regulations around unauthorised legal practice.
DEFAULT_DISCLAIMER = (
    "⚠️ DISCLAIMER: This is an AI-generated response for informational purposes only. "
    "It does NOT constitute legal advice. The information is based on UAE legal texts "
    "but may not reflect the most recent amendments. For legal matters, always consult "
    "a qualified legal professional licensed in the UAE."
)

# =============================================================================
# Query Difficulty Levels (for Evaluation)
# =============================================================================
# Used in deliverable_2 to categorise test questions by complexity.
# This helps us measure whether our RAG system handles nuanced queries well.
QUERY_DIFFICULTY = {
    "simple": "Direct factual question with a single clear answer",
    "moderate": "Question requiring synthesis of multiple articles",
    "complex": "Multi-hop reasoning or cross-law question",
    "adversarial": "Deliberately tricky or out-of-scope question",
}

# =============================================================================
# Metadata Keys
# =============================================================================
# Standard keys expected in Document and Chunk metadata dicts.
# Using constants instead of magic strings prevents typos and enables IDE autocomplete.
METADATA_KEY_SOURCE = "source"
METADATA_KEY_LAW_NAME = "law_name"
METADATA_KEY_ARTICLE_NUMBER = "article_number"
METADATA_KEY_CHAPTER = "chapter"
METADATA_KEY_SECTION = "section"
METADATA_KEY_CHUNK_INDEX = "chunk_index"
METADATA_KEY_CHUNK_SIZE = "chunk_size"

# Standard metadata keys as a tuple (for validation)
DOCUMENT_METADATA_KEYS = (
    METADATA_KEY_SOURCE,
    METADATA_KEY_LAW_NAME,
    METADATA_KEY_ARTICLE_NUMBER,
    METADATA_KEY_CHAPTER,
    METADATA_KEY_SECTION,
)

# =============================================================================
# Penalty Type Labels
# =============================================================================
# Common penalty types found in UAE law — used for entity extraction and tagging.
PENALTY_TYPES = [
    "imprisonment",
    "temporary imprisonment",
    "life imprisonment",
    "death penalty",
    "fine",
    "deportation",
    "community service",
    "confiscation",
    "closure of establishment",
]

# =============================================================================
# Common Legal Abbreviations (UAE Context)
# =============================================================================
# Used by the preprocessor to normalize text.
LEGAL_ABBREVIATIONS = {
    "AED": "Arab Emirates Dirham",
    "UAE": "United Arab Emirates",
    "Art.": "Article",
    "No.": "Number",
    "Ch.": "Chapter",
    "Sec.": "Section",
    "Pt.": "Part",
    "Para.": "Paragraph",
    "Sub-para.": "Sub-paragraph",
}
