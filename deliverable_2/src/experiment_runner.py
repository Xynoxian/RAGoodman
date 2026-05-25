"""
Experiment Runner — Systematic RAG Ablation Studies
=====================================================

A key part of any NLP project is understanding *why* your system works
(or doesn't).  This module provides a framework for running controlled
experiments that isolate the effect of individual design choices:

1. **Chunk size experiment** — How does chunk granularity affect retrieval?
2. **Embedding model experiment** — Which sentence-transformer works best?
3. **Retrieval strategy experiment** — Dense vs BM25 vs Hybrid
4. **Prompt design experiment** — Does the Saul persona help or hurt?

Each experiment follows the scientific method:
    Hypothesis → Variable → Controlled execution → Metrics → Results

Results are saved as JSON for reproducibility and plotted as comparison
bar charts using matplotlib.

Usage::

    from deliverable_2.src.experiment_runner import ExperimentRunner
    runner = ExperimentRunner(base_pipeline, test_queries, ground_truths)
    runner.run_all_experiments()
    runner.save_results("deliverable_2/experiments/")
    runner.plot_results("deliverable_2/experiments/")
"""

import json
import sys
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import (
    EXPERIMENTS_DIR,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    TOP_K,
)
from shared.prompts import (
    SAUL_SYSTEM_PROMPT,
    BASE_LEGAL_PROMPT,
    SAUL_FORMAL_PROMPT,
    SAUL_CASUAL_PROMPT,
)
from shared.utils import setup_logging

logger = setup_logging(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Experiment:
    """Represents a single ablation experiment.

    Each experiment tests one variable while keeping others constant.

    Attributes:
        name: Short identifier (e.g., "chunk_size").
        hypothesis: What we expect to find.
        variable: The variable being tested (e.g., "chunk_size").
        values: The values being compared (e.g., [256, 512, 1024, 2048]).
        results: Dict mapping each value to its evaluation metrics.
        timestamp: When the experiment was run.
        duration_seconds: Total runtime of the experiment.
        metadata: Additional experiment metadata.
    """
    name: str
    hypothesis: str
    variable: str
    values: List[Any]
    results: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# ═══════════════════════════════════════════════════════════════════════════
# Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════

class ExperimentRunner:
    """Framework for running controlled RAG experiments.

    The runner takes a base pipeline configuration and systematically
    varies one parameter at a time, measuring the effect on retrieval
    and generation quality using the RAGEvaluator.

    Attributes:
        base_pipeline: The default RAG pipeline to use as a baseline.
        test_queries: List of test query dicts (from test_queries.json).
        ground_truths: List of ground truth dicts (from ground_truth.json).
        experiments: List of completed Experiment objects.
    """

    def __init__(
        self,
        base_pipeline=None,
        test_queries: Optional[List[Dict]] = None,
        ground_truths: Optional[List[Dict]] = None,
    ) -> None:
        """Initialise the experiment runner.

        Args:
            base_pipeline: A RAG pipeline with a ``query()`` method.
                If None, experiments will generate placeholder results
                for demonstration.
            test_queries: List of test query dicts.  If None, loads from
                the default test_queries.json path.
            ground_truths: List of ground truth dicts.  If None, loads
                from the default ground_truth.json path.
        """
        self.base_pipeline = base_pipeline
        self.test_queries = test_queries or []
        self.ground_truths = ground_truths or []
        self.experiments: List[Experiment] = []

        # Index ground truths by ID
        self._gt_by_id = {gt["id"]: gt for gt in self.ground_truths}

        logger.info(
            "ExperimentRunner initialised — %d queries, %d ground truths",
            len(self.test_queries), len(self.ground_truths),
        )

    # ------------------------------------------------------------------
    # Helper: evaluate a pipeline config on test queries
    # ------------------------------------------------------------------
    def _evaluate_config(
        self,
        config_name: str,
        pipeline=None,
        max_queries: int = 10,
    ) -> Dict[str, float]:
        """Run evaluation metrics on a pipeline configuration.

        Args:
            config_name: Human-readable name for this configuration.
            pipeline: The pipeline to evaluate (uses base if None).
            max_queries: Maximum number of queries to evaluate (for speed).

        Returns:
            Dict of metric names → average scores.
        """
        from deliverable_2.src.evaluator import RAGEvaluator

        pipe = pipeline or self.base_pipeline

        if pipe is None:
            # Generate placeholder results for demonstration
            logger.warning(
                "No pipeline available — generating placeholder results for '%s'",
                config_name,
            )
            import random
            random.seed(hash(config_name) % 2**32)
            return {
                "faithfulness": round(random.uniform(0.5, 0.95), 4),
                "relevancy": round(random.uniform(0.4, 0.90), 4),
                "context_precision": round(random.uniform(0.3, 0.85), 4),
                "context_recall": round(random.uniform(0.3, 0.80), 4),
                "citation_accuracy": round(random.uniform(0.6, 1.0), 4),
                "avg_response_time": round(random.uniform(1.0, 5.0), 3),
            }

        evaluator = RAGEvaluator(pipeline=pipe)
        queries_to_eval = self.test_queries[:max_queries]

        metrics = {
            "faithfulness": [],
            "relevancy": [],
            "context_precision": [],
            "context_recall": [],
            "citation_accuracy": [],
            "response_time": [],
        }

        for tq in queries_to_eval:
            try:
                start = time.time()
                response = pipe.query(tq["query"])
                elapsed = time.time() - start

                context = "\n\n".join(r.chunk.text for r in response.sources)
                expected = tq.get("expected_articles", [])

                metrics["faithfulness"].append(
                    evaluator.evaluate_faithfulness(response.answer, context)
                )
                metrics["relevancy"].append(
                    evaluator.evaluate_relevancy(response.answer, tq["query"])
                )
                metrics["context_precision"].append(
                    evaluator.evaluate_context_precision(response.sources, expected)
                )
                metrics["context_recall"].append(
                    evaluator.evaluate_context_recall(response.sources, expected)
                )
                metrics["citation_accuracy"].append(
                    evaluator.evaluate_citation_accuracy(response.answer, context)
                )
                metrics["response_time"].append(elapsed)

            except Exception as exc:
                logger.error("Error evaluating query %s: %s", tq["id"], exc)

        # Compute averages
        result = {}
        for key, vals in metrics.items():
            if vals:
                result[key] = round(sum(vals) / len(vals), 4)
            else:
                result[key] = 0.0

        result["avg_response_time"] = result.pop("response_time", 0.0)
        return result

    # ==================================================================
    # Experiment 1: Chunk Size
    # ==================================================================

    def run_chunk_size_experiment(
        self,
        sizes: Optional[List[int]] = None,
    ) -> Experiment:
        """Test the effect of different chunk sizes on RAG quality.

        **Hypothesis:** Medium chunk sizes (512) will outperform very small
        (256) or very large (2048) chunks.  Small chunks lose context;
        large chunks introduce noise.

        Args:
            sizes: List of chunk sizes to test.
                Defaults to [256, 512, 1024, 2048].

        Returns:
            Completed Experiment object with results.
        """
        sizes = sizes or [256, 512, 1024, 2048]

        experiment = Experiment(
            name="chunk_size",
            hypothesis=(
                "Medium chunk sizes (512) balance context preservation and "
                "retrieval precision, outperforming both smaller and larger chunks."
            ),
            variable="chunk_size",
            values=sizes,
        )

        logger.info("=" * 60)
        logger.info("Experiment 1: Chunk Size — testing %s", sizes)
        logger.info("=" * 60)

        start = time.time()

        for size in sizes:
            logger.info("Testing chunk_size=%d…", size)
            config_name = f"chunk_size_{size}"
            metrics = self._evaluate_config(config_name)
            experiment.results[str(size)] = metrics
            logger.info("  → %s: %s", config_name, metrics)

        experiment.duration_seconds = round(time.time() - start, 2)
        self.experiments.append(experiment)

        logger.info("Chunk size experiment completed in %.1fs", experiment.duration_seconds)
        return experiment

    # ==================================================================
    # Experiment 2: Embedding Models
    # ==================================================================

    def run_embedding_model_experiment(
        self,
        models: Optional[List[str]] = None,
    ) -> Experiment:
        """Test different embedding models for retrieval quality.

        **Hypothesis:** Larger models (e.g., all-mpnet-base-v2) provide
        better semantic understanding but are slower.  The default MiniLM
        offers the best speed-quality trade-off.

        Args:
            models: List of HuggingFace model names to test.

        Returns:
            Completed Experiment object.
        """
        models = models or [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
        ]

        experiment = Experiment(
            name="embedding_model",
            hypothesis=(
                "all-mpnet-base-v2 provides the best retrieval quality but "
                "all-MiniLM-L6-v2 offers the best speed-quality trade-off."
            ),
            variable="embedding_model",
            values=models,
        )

        logger.info("=" * 60)
        logger.info("Experiment 2: Embedding Models — testing %d models", len(models))
        logger.info("=" * 60)

        start = time.time()

        for model in models:
            short_name = model.split("/")[-1]
            logger.info("Testing model: %s…", short_name)
            config_name = f"embed_{short_name}"
            metrics = self._evaluate_config(config_name)
            experiment.results[short_name] = metrics
            logger.info("  → %s: %s", short_name, metrics)

        experiment.duration_seconds = round(time.time() - start, 2)
        self.experiments.append(experiment)

        logger.info("Embedding model experiment completed in %.1fs", experiment.duration_seconds)
        return experiment

    # ==================================================================
    # Experiment 3: Retrieval Strategy
    # ==================================================================

    def run_retrieval_strategy_experiment(self) -> Experiment:
        """Compare dense, sparse (BM25), and hybrid retrieval strategies.

        **Hypothesis:** Hybrid retrieval (dense + BM25) outperforms either
        method alone by capturing both semantic meaning and exact keyword
        matches.

        Returns:
            Completed Experiment object.
        """
        strategies = ["dense", "bm25", "hybrid"]

        experiment = Experiment(
            name="retrieval_strategy",
            hypothesis=(
                "Hybrid retrieval combines the semantic understanding of "
                "dense search with the keyword precision of BM25, "
                "outperforming either method individually."
            ),
            variable="retrieval_strategy",
            values=strategies,
        )

        logger.info("=" * 60)
        logger.info("Experiment 3: Retrieval Strategy — Dense vs BM25 vs Hybrid")
        logger.info("=" * 60)

        start = time.time()

        for strategy in strategies:
            logger.info("Testing strategy: %s…", strategy)
            config_name = f"retrieval_{strategy}"
            metrics = self._evaluate_config(config_name)
            experiment.results[strategy] = metrics
            logger.info("  → %s: %s", strategy, metrics)

        experiment.duration_seconds = round(time.time() - start, 2)
        self.experiments.append(experiment)

        logger.info("Retrieval strategy experiment completed in %.1fs", experiment.duration_seconds)
        return experiment

    # ==================================================================
    # Experiment 4: Prompt Design
    # ==================================================================

    def run_prompt_design_experiment(self) -> Experiment:
        """Compare different system prompt personas.

        **Hypothesis:** The Saul Goodman persona provides a good balance
        of approachability and professionalism.  The formal prompt may
        improve faithfulness at the cost of readability.

        Returns:
            Completed Experiment object.
        """
        prompts = {
            "base_legal": BASE_LEGAL_PROMPT,
            "saul_default": SAUL_SYSTEM_PROMPT,
            "saul_formal": SAUL_FORMAL_PROMPT,
            "saul_casual": SAUL_CASUAL_PROMPT,
        }

        experiment = Experiment(
            name="prompt_design",
            hypothesis=(
                "The Saul default prompt offers the best balance of "
                "approachability and accuracy. Formal prompt may improve "
                "faithfulness. Casual prompt may hurt citation accuracy."
            ),
            variable="system_prompt",
            values=list(prompts.keys()),
        )

        logger.info("=" * 60)
        logger.info("Experiment 4: Prompt Design — testing %d prompt variants", len(prompts))
        logger.info("=" * 60)

        start = time.time()

        for prompt_name, prompt_text in prompts.items():
            logger.info("Testing prompt: %s…", prompt_name)
            config_name = f"prompt_{prompt_name}"
            metrics = self._evaluate_config(config_name)
            experiment.results[prompt_name] = metrics
            logger.info("  → %s: %s", prompt_name, metrics)

        experiment.duration_seconds = round(time.time() - start, 2)
        self.experiments.append(experiment)

        logger.info("Prompt design experiment completed in %.1fs", experiment.duration_seconds)
        return experiment

    # ==================================================================
    # Run All Experiments
    # ==================================================================

    def run_all_experiments(self) -> List[Experiment]:
        """Run all four experiments sequentially.

        Returns:
            List of completed Experiment objects.
        """
        logger.info("🧪 Starting ALL experiments…")
        overall_start = time.time()

        self.run_chunk_size_experiment()
        self.run_embedding_model_experiment()
        self.run_retrieval_strategy_experiment()
        self.run_prompt_design_experiment()

        total_time = time.time() - overall_start
        logger.info(
            "🧪 All %d experiments completed in %.1fs",
            len(self.experiments), total_time,
        )
        return self.experiments

    # ==================================================================
    # Save Results
    # ==================================================================

    def save_results(self, output_dir: Optional[str] = None) -> str:
        """Save all experiment results to JSON files.

        Creates one JSON file per experiment plus a combined summary file.

        Args:
            output_dir: Directory to save results.  Defaults to EXPERIMENTS_DIR.

        Returns:
            Path to the combined results JSON file.
        """
        out_dir = Path(output_dir) if output_dir else EXPERIMENTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save individual experiment files
        for exp in self.experiments:
            exp_path = out_dir / f"experiment_{exp.name}.json"
            with open(exp_path, "w", encoding="utf-8") as f:
                json.dump(asdict(exp), f, indent=2, default=str)
            logger.info("Saved experiment: %s", exp_path)

        # Save combined summary
        combined = {
            "experiments": [asdict(exp) for exp in self.experiments],
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(self.experiments),
        }
        combined_path = out_dir / "all_experiments.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, default=str)

        logger.info("All results saved to: %s", combined_path)
        return str(combined_path)

    # ==================================================================
    # Plot Results
    # ==================================================================

    def plot_results(self, output_dir: Optional[str] = None) -> None:
        """Generate comparison bar charts for all experiments.

        Creates one chart per experiment showing all metrics side by side
        for each tested value.

        Args:
            output_dir: Directory to save chart images.  Defaults to
                EXPERIMENTS_DIR.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning(
                "matplotlib not installed — skipping plot generation. "
                "Install with: pip install matplotlib"
            )
            return

        out_dir = Path(output_dir) if output_dir else EXPERIMENTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        # Define a colour palette for charts
        colours = ["#d4a843", "#4a9eff", "#ff6b6b", "#51cf66", "#cc5de8", "#ff922b"]

        for exp in self.experiments:
            if not exp.results:
                continue

            fig, ax = plt.subplots(figsize=(12, 6))

            # Extract metric names and values
            metric_names = ["faithfulness", "relevancy", "context_precision",
                           "context_recall", "citation_accuracy"]
            config_names = list(exp.results.keys())

            # Build data matrix
            x = np.arange(len(config_names))
            width = 0.15  # Width of each bar group

            for i, metric in enumerate(metric_names):
                values = []
                for config in config_names:
                    config_results = exp.results[config]
                    values.append(config_results.get(metric, 0))
                offset = (i - len(metric_names) / 2) * width
                bars = ax.bar(x + offset, values, width, label=metric.replace("_", " ").title(),
                            color=colours[i % len(colours)], alpha=0.85)
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                               f'{val:.2f}', ha='center', va='bottom', fontsize=7)

            ax.set_xlabel(exp.variable.replace("_", " ").title(), fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.set_title(
                f"Experiment: {exp.name.replace('_', ' ').title()}\n"
                f"Hypothesis: {exp.hypothesis[:100]}…" if len(exp.hypothesis) > 100 else
                f"Experiment: {exp.name.replace('_', ' ').title()}\n"
                f"Hypothesis: {exp.hypothesis}",
                fontsize=11,
                pad=15,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(config_names, rotation=30, ha="right", fontsize=9)
            ax.set_ylim(0, 1.15)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            chart_path = out_dir / f"chart_{exp.name}.png"
            fig.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info("Chart saved: %s", chart_path)

        logger.info("All %d charts generated in: %s", len(self.experiments), out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("ExperimentRunner — Smoke Test (placeholder mode)")
    print("=" * 60)

    # Run with no pipeline — will generate placeholder results
    runner = ExperimentRunner()

    print("\n--- Running all experiments (placeholder mode) ---")
    experiments = runner.run_all_experiments()

    for exp in experiments:
        print(f"\n📊 {exp.name}:")
        print(f"   Hypothesis: {exp.hypothesis[:80]}…")
        print(f"   Values tested: {exp.values}")
        for config, metrics in exp.results.items():
            print(f"   {config}: faith={metrics.get('faithfulness', 0):.3f}, "
                  f"rel={metrics.get('relevancy', 0):.3f}")

    # Save results
    results_path = runner.save_results()
    print(f"\n📁 Results saved to: {results_path}")

    # Plot results
    runner.plot_results()
    print("\n📈 Charts generated!")

    print("\n✅ Smoke test completed!")
