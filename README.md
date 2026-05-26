# ⚖️ Better Call Saul AI — UAE Legal RAG System

> *"In legal trouble? Better Call Saul AI!"*

A Retrieval-Augmented Generation (RAG) system that serves as an AI legal assistant for UAE law, styled after the iconic Saul Goodman. The system retrieves relevant legal articles from UAE laws and generates accurate, cited responses with a touch of personality.

---

## 📋 Project Overview

| Component | Description |
|-----------|-------------|
| **Domain** | UAE Legal System (Penal Code, Labour Law, Civil Code, Cybercrime Law, Personal Status Law) |
| **Architecture** | RAG (Retrieval-Augmented Generation) |
| **Embedding** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector DB** | ChromaDB (persistent) |
| **LLM** | Google Gemini / OpenAI / Ollama (configurable) |
| **Web Interface** | Flask + Vanilla JS with Saul Goodman theme |
| **Language** | Python 3.10+ |

### Why RAG over Direct LLM?

1. **Accuracy**: UAE laws require precise article citations — LLMs hallucinate specific article numbers
2. **Currency**: Laws change frequently; RAG can be updated without retraining
3. **Transparency**: Every answer shows exactly which articles were used
4. **Trust**: Users can verify claims against the cited source documents

---

## 📁 Project Structure

```
NLP Project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
│
├── shared/                            # Shared utilities across deliverables
│   ├── config.py                      # Central configuration
│   ├── utils.py                       # Data classes (Document, Chunk, RAGResponse)
│   ├── prompts.py                     # All prompt templates (Saul persona variants)
│   └── constants.py                   # Project constants
│
├── deliverable_1/                     # ── DELIVERABLE 1: System Design + Baseline ──
│   ├── README.md                      # D1 documentation
│   ├── data/
│   │   └── raw/                       # UAE law text files (5 laws)
│   ├── src/
│   │   ├── data_loader.py             # Document loading (txt, PDF, HuggingFace)
│   │   ├── preprocessor.py            # Text cleaning & article extraction
│   │   ├── chunker.py                 # Chunking (article-based, recursive, sliding)
│   │   ├── embedder.py                # Sentence-transformer embeddings
│   │   ├── vector_store.py            # ChromaDB vector store
│   │   ├── retriever.py               # Dense similarity retrieval
│   │   ├── generator.py               # LLM generation (Gemini/OpenAI/Ollama)
│   │   └── rag_pipeline.py            # End-to-end RAG pipeline
│   └── scripts/
│       ├── ingest_documents.py        # Data ingestion script
│       └── run_baseline.py            # Baseline evaluation
│
├── deliverable_2/                     # ── DELIVERABLE 2: Experiments + Evaluation ──
│   ├── README.md                      # D2 documentation
│   ├── src/
│   │   ├── hybrid_retriever.py        # BM25 + Dense hybrid search     [BONUS]
│   │   ├── advanced_chunker.py        # Semantic & hierarchical chunking
│   │   ├── reranker.py                # Cross-encoder reranking
│   │   ├── hallucination_detector.py  # Hallucination detection         [BONUS]
│   │   ├── evaluator.py               # Evaluation framework
│   │   └── experiment_runner.py       # Automated experiment runner
│   └── evaluation/
│       ├── test_queries.json          # 30 test queries
│       └── ground_truth.json          # Ground truth answers
│
└── deliverable_3/                     # ── DELIVERABLE 3: Demo + Web Interface ──
    ├── README.md                      # D3 documentation
    └── app/
        ├── app.py                     # Flask web application
        ├── templates/
        │   └── index.html             # Chat interface
        └── static/
            ├── css/style.css          # Saul Goodman themed styling
            └── js/main.js             # Frontend interactivity
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # macOS/Linux

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example environment file
copy .env.example .env         # Windows
# cp .env.example .env        # macOS/Linux

# Edit .env and add your API key:
# GEMINI_API_KEY=your_key_here
# OR use Ollama for free local inference:
# LLM_PROVIDER=ollama
```

### 3. Ingest Documents

```bash
# Process UAE law documents and build the vector store
python deliverable_1/scripts/ingest_documents.py
```

### 4. Run Baseline Evaluation

```bash
# Test the RAG pipeline with sample questions
python deliverable_1/scripts/run_baseline.py
```

### 5. Launch Web Interface

```bash
# Start the Saul Goodman AI chat interface
python deliverable_3/app/app.py
# Open http://localhost:5000 in your browser
```

---

## ⚙️ Configuration

All settings are controlled via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | LLM backend: `gemini`, `openai`, or `ollama` |
| `GEMINI_API_KEY` | - | Google Gemini API key |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum relevance score |

---

## 🏗️ System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  User Query  │───▶│   Embedder   │───▶│  Retriever   │───▶│  Generator   │
│             │    │ (MiniLM-L6)  │    │ (ChromaDB)   │    │ (Gemini/GPT) │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                              │                    │
                                              ▼                    ▼
                                       ┌─────────────┐    ┌──────────────┐
                                       │  Retrieved   │    │    Saul AI    │
                                       │  Articles    │───▶│   Response    │
                                       └─────────────┘    └──────────────┘
```

**Data Flow:**
1. **User Query** → Natural language question about UAE law
2. **Embedding** → Query converted to 384-dim vector using MiniLM-L6-v2
3. **Retrieval** → Top-K most similar chunks found via ChromaDB cosine similarity
4. **Generation** → LLM generates answer using retrieved context + Saul Goodman persona
5. **Response** → Cited answer with source articles, confidence score, and disclaimer

---

## 📊 Deliverables Summary

### Deliverable 1: System Design + Baseline (30%)
- Complete data pipeline (loading → preprocessing → chunking → embedding → storage)
- Working baseline RAG system with dense retrieval
- Initial evaluation with sample queries
- [Full documentation →](deliverable_1/README.md)

### Deliverable 2: Experimental Investigation (50%)
- **4 controlled experiments**: chunk size, embedding model, retrieval strategy, prompt design
- **Evaluation framework**: faithfulness, relevancy, precision, recall, citation accuracy
- **Bonus**: Hybrid search (BM25 + dense), hallucination detection
- **Failure analysis** with categorized error cases
- [Full documentation →](deliverable_2/README.md)

### Deliverable 3: Presentation + Demo (20%)
- Interactive web interface with Saul Goodman theme
- Live demo capability
- [Full documentation →](deliverable_3/README.md)

---

## 🎭 Bonus Features

| Feature | Description | Location |
|---------|-------------|----------|
| **Interactive Web App** | Flask chat interface with glassmorphism dark theme | `deliverable_3/app/` |
| **Hybrid Search** | BM25 + Dense retrieval with Reciprocal Rank Fusion | `deliverable_2/src/hybrid_retriever.py` |
| **Hallucination Detection** | Claim extraction and verification against source docs | `deliverable_2/src/hallucination_detector.py` |



*"I'm not saying you need a lawyer, but you definitely need Saul Goodman AI."* ⚖️
