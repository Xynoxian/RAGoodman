# Deliverable 1: System Design + Baseline (30%)

> **Deadline**: Week 6 — Report (PDF) + Code + Evidence

## Overview

This deliverable implements the complete baseline RAG system for UAE legal document Q&A. It covers the full data pipeline from raw legal texts through to generating answers with the Saul Goodman AI persona.

---

## Components

### 1. Problem Definition

- **Domain**: UAE Legal System — Penal Code, Labour Law, Civil Code, Cybercrime Law, Personal Status Law
- **Target Users**: UAE residents, expatriates, legal students, and professionals
- **Why RAG?**: UAE laws require precise article citations. Direct LLMs hallucinate specific article numbers, penalties, and procedures. RAG grounds every response in actual legal text.

### 2. Dataset Description

**Primary Sources** (in `data/raw/`):
| File | Law | Articles |
|------|-----|----------|
| `uae_penal_code.txt` | Federal Decree-Law No. 31 of 2021 | ~55 articles |
| `uae_labour_law.txt` | Federal Decree-Law No. 33 of 2021 | ~45 articles |
| `uae_civil_code.txt` | Federal Law No. 5 of 1985 | ~35 articles |
| `uae_cybercrime_law.txt` | Federal Decree-Law No. 34 of 2021 | ~28 articles |
| `uae_personal_status_law.txt` | Federal Law No. 28 of 2005 | ~28 articles |

**Supplementary**: HuggingFace dataset `obadabaq/structured-uae-laws` (9,446 Q&A pairs, MIT License)

### 3. System Architecture

```
Raw Legal PDFs/Text
        │
        ▼
┌─────────────────┐
│   Data Loader    │  ◄── data_loader.py
│  (txt, PDF, HF)  │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Preprocessor    │  ◄── preprocessor.py
│ (clean, parse)   │
└────────┬────────┘
         ▼
┌─────────────────┐
│    Chunker       │  ◄── chunker.py
│ (article-based)  │      (ArticleBasedChunker, RecursiveChunker, SlidingWindowChunker)
└────────┬────────┘
         ▼
┌─────────────────┐
│    Embedder      │  ◄── embedder.py
│  (MiniLM-L6-v2)  │      (sentence-transformers)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Vector Store    │  ◄── vector_store.py
│   (ChromaDB)     │      (persistent collection)
└────────┬────────┘
         ▼
┌─────────────────┐
│   Retriever      │  ◄── retriever.py
│  (dense search)  │      (cosine similarity, top-K)
└────────┬────────┘
         ▼
┌─────────────────┐
│   Generator      │  ◄── generator.py
│  (Gemini/GPT)    │      (Saul Goodman persona)
└────────┬────────┘
         ▼
┌─────────────────┐
│  RAG Pipeline    │  ◄── rag_pipeline.py
│ (orchestration)  │      (timing, logging, debug mode)
└─────────────────┘
```

### 4. Design Justification

| Decision | Choice | Why |
|----------|--------|-----|
| Embedding Model | MiniLM-L6-v2 | Fast on CPU, 384-dim, good quality for semantic search |
| Vector DB | ChromaDB | Python-native, persistent, simple API, good for project scope |
| Chunking | Article-based (primary) | Legal documents have natural article boundaries — preserves semantic integrity |
| LLM | Gemini Flash | Free tier, fast, high quality generation |

### 5. Baseline Implementation

The baseline system uses:
- **Dense retrieval only** (cosine similarity on sentence-transformer embeddings)
- **Article-based chunking** (one chunk per article)
- **Top-5 retrieval** with 0.3 similarity threshold
- **Saul Goodman persona** prompt for generation

### 6. Initial Evaluation

Run the baseline evaluation:
```bash
python scripts/run_baseline.py
```

This tests 20 queries across all 5 law categories and reports:
- Retrieved article relevance
- Answer quality
- Citation accuracy
- Response timing

---

## How to Run

```bash
# 1. From project root, ensure .env is configured
copy ..\.env.example ..\.env

# 2. Ingest documents into vector store
python scripts/ingest_documents.py

# 3. Run baseline evaluation
python scripts/run_baseline.py
```

---

## Code Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `src/data_loader.py` | Load documents from files/APIs | `load_text_file()`, `load_all_documents()` |
| `src/preprocessor.py` | Clean and parse legal text | `clean_text()`, `extract_articles()` |
| `src/chunker.py` | Split docs into chunks | `ArticleBasedChunker`, `RecursiveChunker` |
| `src/embedder.py` | Generate embeddings | `EmbeddingModel` |
| `src/vector_store.py` | Manage ChromaDB | `VectorStore` |
| `src/retriever.py` | Retrieve relevant chunks | `BaseRetriever` |
| `src/generator.py` | Generate LLM responses | `LLMGenerator` |
| `src/rag_pipeline.py` | Orchestrate pipeline | `RAGPipeline` |
