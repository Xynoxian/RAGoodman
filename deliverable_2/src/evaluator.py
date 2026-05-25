"""
RAG Evaluation Framework — Comprehensive Quality Metrics
==========================================================

Evaluating a RAG system requires measuring quality at multiple stages:

1. **Retrieval quality** — Are we finding the right chunks?
   - Context Precision: Of the retrieved docs, how many are actually relevant?
   - Context Recall: Of all relevant docs, how many did we retrieve?

2. **Generation quality** — Is the answer correct and grounded?
   - Faithfulness: Is the answer supported by the retrieved context?
   - Answer Relevancy: Does the answer actually address the question?

3. **Legal-specific metrics** — Domain-specific quality checks
   - Citation Accuracy: Are article numbers and law references correct?

This framework uses lightweight, embedding-based metrics that don't
require additional LLM calls (keeping evaluation fast and cheap).

Usage::

    from deliverable_2.src.evaluator import RAGEvaluator
    evaluator = RAGEvaluator(rag_pipeline)
    results = evaluator.run_evaluation("test_queries.json", "ground_truth.json")
    evaluator.generate_report(results, "evaluation_report.md")
"""

import json
import re
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import EXPERIMENTS_DIR
from shared.utils import RetrievalResult, RAGResponse, Chunk, setup_logging

logger = setup_logging(__name__)


class RAGEvaluator:
    """Comprehensive evaluation framework for the UAE Legal RAG system.

    Measures retrieval quality, generation quality, and legal-domain-specific
    metrics.  All metrics return scores in [0, 1] where higher is better.

    The evaluator works with the standard ``RAGPipeline`` from Deliverable 1,
    running test queries through it and comparing outputs against ground
    truth annotations.

    Attributes:
        pipeline: The RAG pipeline to evaluate (must have a ``query()`` method).
        embedder: Optional embedding model for semantic similarity metrics.
    """

    def __init__(self, pipeline=None, embedder=None) -> None:
        """Initialise the evaluator.

        Args:
            pipeline: A RAG pipeline with a ``query(str) -> RAGResponse``
                method.  If None, individual metric methods can still be
                called directly with pre-computed results.
            embedder: An embedding model with ``embed_text(str) -> list[float]``
                for computing semantic similarity metrics.  If None,
                semantic metrics fall back to keyword overlap.
        """
        self.pipeline = pipeline
        self.embedder = embedder
        logger.info("RAGEvaluator initialised (pipeline=%s)", "yes" if pipeline else "no")

    # ==================================================================
    # Metric 1: Faithfulness
    # ==================================================================

    def evaluate_faithfulness(self, response: str, context: str) -> float:
        """Measure how faithful the response is to the retrieved context.

        Faithfulness = proportion of response claims that are supported
        by the context.  Uses keyword overlap as a proxy for entailment.

        A faithful response only contains information present in the context.
        Fabricated or unsupported claims reduce the score.

        Args:
            response: The generated answer text.
            context: Concatenated retrieved context chunks.

        Returns:
            Float in [0, 1].  1.0 = fully faithful, 0.0 = fully hallucinated.
        """
        if not response.strip() or not context.strip():
            return 0.0

        # Extract factual sentences from the response
        response_sentences = [
            s.strip() for s in re.split(r'[.!?]+', response) if len(s.strip()) > 15
        ]

        if not response_sentences:
            return 1.0  # No factual sentences = nothing to verify

        context_lower = context.lower()
        supported_count = 0

        for sentence in response_sentences:
            # Extract key terms from the sentence
            terms = set(re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower()))
            stop_words = {
                "this", "that", "with", "from", "have", "been", "were", "will",
                "shall", "would", "could", "should", "into", "upon", "also",
                "such", "than", "other", "more", "most", "only", "each",
            }
            terms -= stop_words

            if not terms:
                supported_count += 1
                continue

            # Check what fraction of key terms appear in context
            found = sum(1 for t in terms if t in context_lower)
            overlap = found / len(terms)

            if overlap >= 0.5:  # At least half the terms found
                supported_count += 1

        faithfulness = supported_count / len(response_sentences)
        return round(faithfulness, 4)

    # ==================================================================
    # Metric 2: Answer Relevancy
    # ==================================================================

    def evaluate_relevancy(self, response: str, query: str) -> float:
        """Measure how relevant the response is to the original query.

        Answer relevancy checks if the response actually addresses what
        was asked.  Uses keyword overlap between query and response terms.

        If an embedder is available, also includes semantic similarity.

        Args:
            response: The generated answer text.
            query: The original user question.

        Returns:
            Float in [0, 1].  1.0 = perfectly relevant to the query.
        """
        if not response.strip() or not query.strip():
            return 0.0

        # Keyword-based relevancy
        query_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        response_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', response.lower()))

        stop_words = {
            "the", "and", "for", "are", "was", "were", "been", "have", "has",
            "had", "will", "shall", "can", "may", "what", "how", "who", "which",
            "this", "that", "with", "from", "into", "does",
        }
        query_terms -= stop_words
        response_terms -= stop_words

        if not query_terms:
            return 0.5  # Can't assess relevancy without query terms

        # What fraction of query terms appear in the response?
        found = sum(1 for t in query_terms if t in response_terms)
        keyword_score = found / len(query_terms)

        # Semantic similarity (if embedder available)
        semantic_score = 0.0
        if self.embedder is not None:
            try:
                import numpy as np
                q_vec = np.array(self.embedder.embed_text(query))
                r_vec = np.array(self.embedder.embed_text(response[:512]))  # truncate long responses
                norm_q = np.linalg.norm(q_vec)
                norm_r = np.linalg.norm(r_vec)
                if norm_q > 0 and norm_r > 0:
                    semantic_score = float(np.dot(q_vec, r_vec) / (norm_q * norm_r))
            except Exception:
                pass

        # Combine keyword and semantic scores
        if self.embedder is not None:
            relevancy = 0.4 * keyword_score + 0.6 * semantic_score
        else:
            relevancy = keyword_score

        return round(max(0.0, min(1.0, relevancy)), 4)

    # ==================================================================
    # Metric 3: Context Precision
    # ==================================================================

    def evaluate_context_precision(
        self,
        retrieved_docs: List[RetrievalResult],
        relevant_docs: List[str],
    ) -> float:
        """Measure what fraction of retrieved documents are actually relevant.

        Context Precision = |retrieved ∩ relevant| / |retrieved|

        This tells us how much noise is in our retrieval results.
        High precision means most retrieved chunks are useful.

        Args:
            retrieved_docs: List of RetrievalResult objects from the retriever.
            relevant_docs: List of relevant document/article identifiers
                (e.g., article numbers like "461", "29").

        Returns:
            Float in [0, 1].  1.0 = all retrieved docs are relevant.
        """
        if not retrieved_docs:
            return 0.0

        if not relevant_docs:
            return 0.0  # No ground truth = can't assess

        relevant_set = set(str(d).lower().strip() for d in relevant_docs)

        relevant_count = 0
        for result in retrieved_docs:
            meta = result.chunk.metadata
            # Check article number match
            article_num = str(meta.get("article_number", "")).strip()
            law_name = str(meta.get("law_name", "")).lower().strip()
            source = str(meta.get("source", "")).lower().strip()

            # A retrieved doc is relevant if its article number or source
            # matches any item in the relevant_docs list
            for ref in relevant_set:
                if (article_num and article_num == ref) or \
                   (ref in law_name) or (ref in source) or \
                   (ref in result.chunk.text.lower()):
                    relevant_count += 1
                    break

        precision = relevant_count / len(retrieved_docs)
        return round(precision, 4)

    # ==================================================================
    # Metric 4: Context Recall
    # ==================================================================

    def evaluate_context_recall(
        self,
        retrieved_docs: List[RetrievalResult],
        relevant_docs: List[str],
    ) -> float:
        """Measure what fraction of relevant documents were actually retrieved.

        Context Recall = |retrieved ∩ relevant| / |relevant|

        This tells us if we're missing important chunks.
        High recall means we found most of what we needed.

        Args:
            retrieved_docs: List of RetrievalResult objects from the retriever.
            relevant_docs: List of relevant document/article identifiers.

        Returns:
            Float in [0, 1].  1.0 = all relevant docs were retrieved.
        """
        if not relevant_docs:
            return 1.0  # Nothing to retrieve = perfect recall (vacuously true)

        if not retrieved_docs:
            return 0.0

        relevant_set = set(str(d).lower().strip() for d in relevant_docs)

        # Build a set of all identifiers from retrieved docs
        retrieved_identifiers: set = set()
        for result in retrieved_docs:
            meta = result.chunk.metadata
            article_num = str(meta.get("article_number", "")).strip()
            if article_num:
                retrieved_identifiers.add(article_num)
            # Also check text for article references
            text_articles = re.findall(r'Article\s+(\d+)', result.chunk.text, re.IGNORECASE)
            for a in text_articles:
                retrieved_identifiers.add(a)

        # How many relevant docs were found?
        found = 0
        for ref in relevant_set:
            # Check exact match or substring match in retrieved identifiers
            for rid in retrieved_identifiers:
                if ref == rid or ref in rid:
                    found += 1
                    break

        recall = found / len(relevant_set)
        return round(recall, 4)

    # ==================================================================
    # Metric 5: Citation Accuracy (Legal-Specific)
    # ==================================================================

    def evaluate_citation_accuracy(self, response: str, context: str) -> float:
        """Measure accuracy of legal citations (article numbers, law names).

        This is a domain-specific metric that checks whether article
        numbers and law references mentioned in the response actually
        appear in the source context.

        Particularly important for legal applications where citing
        non-existent articles could be misleading or harmful.

        Args:
            response: The generated answer text.
            context: The source context text.

        Returns:
            Float in [0, 1].  1.0 = all citations verified.
            Returns 1.0 if no citations are found (nothing to check).
        """
        # Extract article references from response
        response_articles = set(re.findall(
            r'(?:Article|Art\.?)\s*\(?(\d+)\)?', response, re.IGNORECASE
        ))

        if not response_articles:
            return 1.0  # No citations to check

        # Extract article references from context
        context_articles = set(re.findall(
            r'(?:Article|Art\.?)\s*\(?(\d+)\)?', context, re.IGNORECASE
        ))

        # Check how many response articles exist in the context
        verified = sum(1 for a in response_articles if a in context_articles)
        accuracy = verified / len(response_articles)

        if accuracy < 1.0:
            unverified = response_articles - context_articles
            logger.warning("Unverified article citations: %s", unverified)

        return round(accuracy, 4)

    # ==================================================================
    # Full Evaluation Pipeline
    # ==================================================================

    def run_evaluation(
        self,
        test_queries_path: str,
        ground_truths_path: str,
    ) -> Dict[str, Any]:
        """Run a full evaluation over a test dataset.

        Reads test queries and ground truth from JSON files, runs each
        query through the pipeline, and computes all metrics.

        Args:
            test_queries_path: Path to JSON file with test queries.
                Expected format::

                    [{"id": "q1", "query": "...", "expected_articles": ["461"], ...}, ...]

            ground_truths_path: Path to JSON file with ground truth answers.
                Expected format::

                    [{"id": "q1", "reference_answer": "...", "relevant_articles": ["461"], ...}, ...]

        Returns:
            Dict with overall scores and per-query results::

                {
                    "overall": {"faithfulness": 0.85, "relevancy": 0.78, ...},
                    "per_query": [{"id": "q1", "query": "...", "metrics": {...}}, ...],
                    "timestamp": "2024-...",
                    "num_queries": 30,
                }

        Raises:
            ValueError: If pipeline is not set or files don't exist.
        """
        if self.pipeline is None:
            raise ValueError(
                "RAGPipeline not set. Pass a pipeline to RAGEvaluator() or "
                "use individual metric methods directly."
            )

        # Load test data
        test_queries_file = Path(test_queries_path)
        ground_truths_file = Path(ground_truths_path)

        if not test_queries_file.exists():
            raise FileNotFoundError(f"Test queries file not found: {test_queries_path}")
        if not ground_truths_file.exists():
            raise FileNotFoundError(f"Ground truths file not found: {ground_truths_path}")

        with open(test_queries_file, "r", encoding="utf-8") as f:
            test_queries = json.load(f)
        with open(ground_truths_file, "r", encoding="utf-8") as f:
            ground_truths = json.load(f)

        # Index ground truths by query ID for quick lookup
        gt_by_id = {gt["id"]: gt for gt in ground_truths}

        logger.info("Running evaluation on %d test queries…", len(test_queries))

        # Aggregate metrics
        all_metrics = {
            "faithfulness": [],
            "relevancy": [],
            "context_precision": [],
            "context_recall": [],
            "citation_accuracy": [],
        }
        per_query_results: List[Dict] = []

        for i, tq in enumerate(test_queries):
            qid = tq["id"]
            query = tq["query"]
            gt = gt_by_id.get(qid, {})

            logger.info("Evaluating query %d/%d: %s", i + 1, len(test_queries), qid)

            try:
                # Run through pipeline
                start = time.time()
                rag_response: RAGResponse = self.pipeline.query(query)
                elapsed = time.time() - start

                # Build context string from sources
                context = "\n\n".join(r.chunk.text for r in rag_response.sources)

                # Compute metrics
                faithfulness = self.evaluate_faithfulness(rag_response.answer, context)
                relevancy = self.evaluate_relevancy(rag_response.answer, query)
                citation_accuracy = self.evaluate_citation_accuracy(rag_response.answer, context)

                # Context precision/recall need ground truth articles
                expected_articles = tq.get("expected_articles", gt.get("relevant_articles", []))
                precision = self.evaluate_context_precision(rag_response.sources, expected_articles)
                recall = self.evaluate_context_recall(rag_response.sources, expected_articles)

                # Record metrics
                metrics = {
                    "faithfulness": faithfulness,
                    "relevancy": relevancy,
                    "context_precision": precision,
                    "context_recall": recall,
                    "citation_accuracy": citation_accuracy,
                    "confidence": rag_response.confidence,
                    "num_sources": rag_response.source_count,
                    "response_time": round(elapsed, 3),
                    "hallucination_flags": rag_response.hallucination_flags,
                }

                for key in all_metrics:
                    all_metrics[key].append(metrics[key])

                per_query_results.append({
                    "id": qid,
                    "query": query,
                    "category": tq.get("category", "unknown"),
                    "difficulty": tq.get("difficulty", "unknown"),
                    "answer_preview": rag_response.answer[:200],
                    "metrics": metrics,
                })

            except Exception as exc:
                logger.error("Error evaluating query %s: %s", qid, exc)
                per_query_results.append({
                    "id": qid,
                    "query": query,
                    "error": str(exc),
                    "metrics": {k: 0.0 for k in all_metrics},
                })
                for key in all_metrics:
                    all_metrics[key].append(0.0)

        # Compute overall averages
        overall = {
            key: round(sum(vals) / len(vals), 4) if vals else 0.0
            for key, vals in all_metrics.items()
        }

        results = {
            "overall": overall,
            "per_query": per_query_results,
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(test_queries),
        }

        logger.info(
            "Evaluation complete — Overall: faith=%.3f, rel=%.3f, prec=%.3f, recall=%.3f, cite=%.3f",
            overall["faithfulness"], overall["relevancy"],
            overall["context_precision"], overall["context_recall"],
            overall["citation_accuracy"],
        )

        return results

    # ==================================================================
    # Report Generation
    # ==================================================================

    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a markdown evaluation report.

        Creates a comprehensive, human-readable report with:
        - Overall scores summary table
        - Per-query results table
        - Analysis by category and difficulty
        - Identified issues and recommendations

        Args:
            results: Results dict from ``run_evaluation()``.
            output_path: Path to write the markdown report.  If None,
                defaults to ``EXPERIMENTS_DIR / evaluation_report.md``.

        Returns:
            The markdown report as a string.
        """
        if output_path is None:
            EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
            output_path = str(EXPERIMENTS_DIR / "evaluation_report.md")

        overall = results["overall"]
        per_query = results.get("per_query", [])

        # Build report
        lines: List[str] = []
        lines.append("# 📊 UAE Legal RAG — Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {results.get('timestamp', 'N/A')}")
        lines.append(f"**Total queries evaluated:** {results.get('num_queries', 0)}")
        lines.append("")

        # Overall scores table
        lines.append("## Overall Scores")
        lines.append("")
        lines.append("| Metric | Score | Rating |")
        lines.append("|--------|-------|--------|")

        for metric, score in overall.items():
            rating = "🟢 Good" if score >= 0.7 else "🟡 Fair" if score >= 0.4 else "🔴 Poor"
            display_name = metric.replace("_", " ").title()
            lines.append(f"| {display_name} | {score:.4f} | {rating} |")

        lines.append("")

        # Per-query results
        lines.append("## Per-Query Results")
        lines.append("")
        lines.append("| ID | Category | Difficulty | Faith. | Relev. | Prec. | Recall | Cite. |")
        lines.append("|-----|----------|------------|--------|--------|-------|--------|-------|")

        for qr in per_query:
            if "error" in qr:
                lines.append(f"| {qr['id']} | — | — | ❌ Error: {qr['error'][:30]} | | | | |")
                continue

            m = qr["metrics"]
            lines.append(
                f"| {qr['id']} | {qr.get('category', '—')} | {qr.get('difficulty', '—')} | "
                f"{m['faithfulness']:.2f} | {m['relevancy']:.2f} | "
                f"{m['context_precision']:.2f} | {m['context_recall']:.2f} | "
                f"{m['citation_accuracy']:.2f} |"
            )

        lines.append("")

        # Analysis by category
        categories: Dict[str, List[Dict]] = {}
        for qr in per_query:
            cat = qr.get("category", "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(qr)

        if categories:
            lines.append("## Analysis by Category")
            lines.append("")
            lines.append("| Category | Avg Faithfulness | Avg Relevancy | Count |")
            lines.append("|----------|-----------------|---------------|-------|")

            for cat, queries in sorted(categories.items()):
                valid_queries = [q for q in queries if "metrics" in q and "error" not in q]
                if valid_queries:
                    avg_faith = sum(q["metrics"]["faithfulness"] for q in valid_queries) / len(valid_queries)
                    avg_rel = sum(q["metrics"]["relevancy"] for q in valid_queries) / len(valid_queries)
                    lines.append(f"| {cat} | {avg_faith:.4f} | {avg_rel:.4f} | {len(valid_queries)} |")

            lines.append("")

        # Analysis by difficulty
        difficulties: Dict[str, List[Dict]] = {}
        for qr in per_query:
            diff = qr.get("difficulty", "unknown")
            if diff not in difficulties:
                difficulties[diff] = []
            difficulties[diff].append(qr)

        if difficulties:
            lines.append("## Analysis by Difficulty")
            lines.append("")
            lines.append("| Difficulty | Avg Faithfulness | Avg Relevancy | Count |")
            lines.append("|------------|-----------------|---------------|-------|")

            for diff, queries in sorted(difficulties.items()):
                valid_queries = [q for q in queries if "metrics" in q and "error" not in q]
                if valid_queries:
                    avg_faith = sum(q["metrics"]["faithfulness"] for q in valid_queries) / len(valid_queries)
                    avg_rel = sum(q["metrics"]["relevancy"] for q in valid_queries) / len(valid_queries)
                    lines.append(f"| {diff} | {avg_faith:.4f} | {avg_rel:.4f} | {len(valid_queries)} |")

            lines.append("")

        # Hallucination issues
        flagged_queries = [
            qr for qr in per_query
            if "metrics" in qr and qr["metrics"].get("hallucination_flags")
        ]
        if flagged_queries:
            lines.append("## ⚠️ Hallucination Flags")
            lines.append("")
            for qr in flagged_queries:
                lines.append(f"- **{qr['id']}**: {qr['metrics']['hallucination_flags']}")
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        if overall.get("faithfulness", 0) < 0.7:
            lines.append("- 🔴 **Faithfulness is low** — Consider improving the prompt to emphasize staying grounded in context.")
        if overall.get("context_recall", 0) < 0.5:
            lines.append("- 🔴 **Context recall is low** — Try increasing TOP_K or using hybrid retrieval.")
        if overall.get("context_precision", 0) < 0.5:
            lines.append("- 🟡 **Context precision is low** — Consider adding a reranker to filter irrelevant chunks.")
        if overall.get("citation_accuracy", 0) < 0.8:
            lines.append("- 🟡 **Citation accuracy needs improvement** — Strengthen the prompt's instruction to only cite articles from context.")
        if all(v >= 0.7 for v in overall.values()):
            lines.append("- 🟢 **All metrics look healthy!** Consider fine-tuning for even better performance.")
        lines.append("")

        # Write report
        report_text = "\n".join(lines)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info("Evaluation report saved to: %s", output_path)
        return report_text


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("RAGEvaluator — Smoke Test (individual metrics)")
    print("=" * 60)

    evaluator = RAGEvaluator()

    # Test faithfulness
    context = "Article 461: Theft is punished by imprisonment up to 3 years."
    response = "According to Article 461, theft carries up to 3 years imprisonment."
    print(f"\nFaithfulness: {evaluator.evaluate_faithfulness(response, context)}")

    # Test relevancy
    query = "What is the punishment for theft?"
    print(f"Relevancy: {evaluator.evaluate_relevancy(response, query)}")

    # Test citation accuracy
    print(f"Citation accuracy: {evaluator.evaluate_citation_accuracy(response, context)}")

    # Test context precision with mock results
    mock_results = [
        RetrievalResult(
            chunk=Chunk(text="Article 461...", metadata={"article_number": "461"}, chunk_id="c1"),
            score=0.9,
        ),
        RetrievalResult(
            chunk=Chunk(text="Article 30...", metadata={"article_number": "30"}, chunk_id="c2"),
            score=0.7,
        ),
    ]
    print(f"Context precision: {evaluator.evaluate_context_precision(mock_results, ['461'])}")
    print(f"Context recall: {evaluator.evaluate_context_recall(mock_results, ['461', '462'])}")

    print("\n✅ Smoke test completed!")
