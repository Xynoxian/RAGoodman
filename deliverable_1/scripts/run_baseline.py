#!/usr/bin/env python3
"""
Baseline RAG Evaluation — Better Call Saul AI
===============================================

Tests the baseline RAG pipeline with a curated set of UAE legal questions.
This script is used for Deliverable 1's "Initial Evaluation" section.

For each question, it shows:
    - The query
    - Retrieved source articles
    - Generated answer (in Saul Goodman's voice)
    - Confidence score
    - Timing breakdown

Usage:
    # First, run the ingestion pipeline:
    python deliverable_1/scripts/ingest_documents.py

    # Then run this baseline evaluation:
    python deliverable_1/scripts/run_baseline.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils import setup_logging
from shared.config import PROJECT_ROOT
from deliverable_1.src.rag_pipeline import RAGPipeline

logger = setup_logging("baseline_eval")

# =============================================================================
# Test Questions
# =============================================================================
# These cover all 5 UAE law documents with varying difficulty levels.
# Each question should ideally retrieve relevant articles and produce
# a grounded, cited response.
# =============================================================================

TEST_QUESTIONS = [
    # --- Penal Code ---
    "What is the punishment for theft in the UAE?",
    "What are the penalties for murder under UAE law?",
    "Is defamation a criminal offense in the UAE?",
    "What happens if someone commits fraud in the UAE?",
    # --- Labour Law ---
    "How many days of annual leave am I entitled to under UAE labour law?",
    "What are the rules for probation periods in UAE employment?",
    "How is end-of-service gratuity calculated in the UAE?",
    "Can my employer terminate me without notice in the UAE?",
    # --- Civil Code ---
    "What are the requirements for a valid contract under UAE civil law?",
    "How is compensation for damages determined in UAE law?",
    "What are the rules about property ownership in the UAE?",
    # --- Cybercrime Law ---
    "What is the penalty for hacking into someone's account in the UAE?",
    "Is online defamation illegal in the UAE?",
    "What are the consequences of sharing someone's private information online?",
    # --- Personal Status Law ---
    "How is child custody determined in UAE divorce cases?",
    "What are the rules for inheritance under UAE law?",
    "What are the grounds for divorce in the UAE?",
    # --- Cross-law / Complex ---
    "What are my rights as an employee if I face workplace harassment?",
    "What legal protections exist for personal data in the UAE?",
    # --- Adversarial / Edge Case ---
    "What is the speed limit on UAE highways?",  # Out-of-scope
]


def main():
    """
    Run baseline evaluation across all test questions.

    Process:
        1. Initialize the RAG pipeline (loads embedder, vector store, etc.)
        2. For each test question:
           a. Run the full RAG pipeline
           b. Display results (answer, sources, confidence, timing)
        3. Save all results to a JSON file for analysis
    """
    logger.info("=" * 70)
    logger.info("⚖️  BETTER CALL SAUL AI — Baseline Evaluation")
    logger.info("=" * 70)
    logger.info("Testing %d questions across 5 UAE law categories\n", len(TEST_QUESTIONS))

    # Initialize pipeline
    logger.info("🔧 Initializing RAG pipeline...")
    try:
        pipeline = RAGPipeline()
        logger.info("✅ Pipeline ready\n")
    except Exception as e:
        logger.error("❌ Failed to initialize pipeline: %s", e)
        logger.error("   Did you run ingest_documents.py first?")
        sys.exit(1)

    results = []
    total_start = time.time()

    for i, question in enumerate(TEST_QUESTIONS, 1):
        logger.info("-" * 70)
        logger.info("📝 Question %d/%d: %s", i, len(TEST_QUESTIONS), question)
        logger.info("-" * 70)

        try:
            # Run the full RAG pipeline
            response = pipeline.query(question)

            # Display results
            logger.info("🎯 Answer:")
            # Print answer with wrapping
            answer_lines = response.answer.split("\n")
            for line in answer_lines[:10]:  # Show first 10 lines
                logger.info("   %s", line)
            if len(answer_lines) > 10:
                logger.info("   ... (truncated, %d total lines)", len(answer_lines))

            logger.info("\n📚 Sources (%d retrieved):", response.source_count)
            for j, source in enumerate(response.sources[:3], 1):  # Show top 3
                src_law = source.chunk.metadata.get("law_name", "Unknown")
                src_article = source.chunk.metadata.get("article_number", "?")
                logger.info("   %d. %s, Article %s (score: %.3f)", j, src_law, src_article, source.score)

            logger.info("\n📊 Metrics:")
            logger.info("   Confidence:      %.2f", response.confidence)
            logger.info("   Retrieval time:  %.3fs", response.retrieval_time)
            logger.info("   Generation time: %.3fs", response.generation_time)
            logger.info("   Total time:      %.3fs", response.total_time)

            if response.hallucination_flags:
                logger.warning("⚠️  Hallucination flags: %s", response.hallucination_flags)

            # Store result for JSON export
            results.append({
                "question_id": i,
                "question": question,
                "answer": response.answer,
                "confidence": response.confidence,
                "source_count": response.source_count,
                "sources": [
                    {
                        "law_name": s.chunk.metadata.get("law_name", ""),
                        "article_number": s.chunk.metadata.get("article_number", ""),
                        "score": round(s.score, 4),
                        "text_preview": s.chunk.text[:150] + "...",
                    }
                    for s in response.sources
                ],
                "retrieval_time": round(response.retrieval_time, 4),
                "generation_time": round(response.generation_time, 4),
                "total_time": round(response.total_time, 4),
                "hallucination_flags": response.hallucination_flags,
            })

        except Exception as e:
            logger.error("❌ Error processing question %d: %s", i, e)
            results.append({
                "question_id": i,
                "question": question,
                "error": str(e),
            })

        logger.info("")  # Blank line between questions

    # Save results to JSON
    total_elapsed = time.time() - total_start
    output_dir = PROJECT_ROOT / "deliverable_1" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        "evaluation": "baseline",
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(TEST_QUESTIONS),
        "total_time_seconds": round(total_elapsed, 2),
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Summary
    successful = [r for r in results if "error" not in r]
    avg_confidence = sum(r["confidence"] for r in successful) / len(successful) if successful else 0
    avg_time = sum(r["total_time"] for r in successful) / len(successful) if successful else 0

    logger.info("=" * 70)
    logger.info("✅ BASELINE EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info("   Total questions:    %d", len(TEST_QUESTIONS))
    logger.info("   Successful:         %d", len(successful))
    logger.info("   Failed:             %d", len(results) - len(successful))
    logger.info("   Avg confidence:     %.2f", avg_confidence)
    logger.info("   Avg response time:  %.3fs", avg_time)
    logger.info("   Total eval time:    %.2fs", total_elapsed)
    logger.info("   Results saved to:   %s", output_file)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
