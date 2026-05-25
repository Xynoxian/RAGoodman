# Deliverable 2: Experimental Investigation + Evaluation (50%)

> **Deadline**: Week 9 — Report (PDF) + Code + Evidence

## Overview

This deliverable extends the baseline RAG system with systematic experiments, advanced features, and a comprehensive evaluation framework.

---

## Experiments

### Experiment 1: Chunk Size Variation
- **Hypothesis**: Smaller chunks improve retrieval accuracy by reducing noise
- **Variable**: `CHUNK_SIZE` = [256, 512, 1024, 2048] characters
- **Metrics**: Context precision, answer relevancy, retrieval time

### Experiment 2: Embedding Model Comparison
- **Hypothesis**: Larger embedding models produce better retrieval quality
- **Variable**: Model = [all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L3-v2]
- **Metrics**: Context precision, context recall, embedding time

### Experiment 3: Retrieval Strategy
- **Hypothesis**: Hybrid search (BM25 + dense) improves recall for keyword-heavy legal queries
- **Variable**: Strategy = [Dense only, BM25 only, Hybrid (RRF)]
- **Metrics**: Context precision, context recall, answer relevancy

### Experiment 4: Prompt Design
- **Hypothesis**: The Saul Goodman persona does not reduce answer accuracy
- **Variable**: Prompt = [Base legal, Saul standard, Saul formal, Saul casual]
- **Metrics**: Faithfulness, answer relevancy, citation accuracy

---

## Evaluation Framework

Located in `src/evaluator.py`:

| Metric | What It Measures | Range |
|--------|-----------------|-------|
| **Faithfulness** | Is the answer grounded in retrieved context? | 0.0 – 1.0 |
| **Answer Relevancy** | Does the answer address the question? | 0.0 – 1.0 |
| **Context Precision** | Are retrieved documents relevant? | 0.0 – 1.0 |
| **Context Recall** | Were all relevant documents retrieved? | 0.0 – 1.0 |
| **Citation Accuracy** | Are cited article numbers correct? | 0.0 – 1.0 |

---

## Bonus Features

### Hybrid Search (`src/hybrid_retriever.py`)
- BM25 sparse retrieval using `rank_bm25`
- Dense retrieval via ChromaDB
- Reciprocal Rank Fusion (RRF) to combine both
- Configurable dense vs. sparse weights

### Cross-Encoder Reranking (`src/reranker.py`)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking top-K results
- Improves precision by rescoring with full query-document attention

### Hallucination Detection (`src/hallucination_detector.py`)
- Claim extraction: breaks responses into atomic claims
- Claim verification: checks each claim against source context
- Confidence scoring and flagging of unsupported claims

---

## Test Data

- `evaluation/test_queries.json` — 30 queries across 5 UAE law categories
- `evaluation/ground_truth.json` — Ground truth answers with expected articles

---

## How to Run

```bash
# Run all experiments
python deliverable_2/src/experiment_runner.py

# Run evaluation only
python deliverable_2/src/evaluator.py

# Results and plots saved to deliverable_2/experiments/results/
```

---

## Code Files

| File | Purpose |
|------|---------|
| `src/hybrid_retriever.py` | BM25 + Dense hybrid search with RRF |
| `src/advanced_chunker.py` | Semantic and hierarchical chunking |
| `src/reranker.py` | Cross-encoder reranking |
| `src/hallucination_detector.py` | Claim extraction and verification |
| `src/evaluator.py` | 5-metric evaluation framework |
| `src/experiment_runner.py` | Automated experiment runner with plotting |
