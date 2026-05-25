"""
Flask Web Application — Better Call Saul AI
=============================================

A premium web interface for the UAE Legal RAG system, themed as
"Better Call Saul AI" — a witty, knowledgeable legal assistant.

Routes:
    GET  /           → Serves the main chat interface
    POST /api/query  → Accepts {query: str}, returns RAG response
    GET  /api/health → Health check endpoint
    GET  /api/stats  → Vector store statistics

Usage:
    python deliverable_3/app/app.py
    # Then open http://localhost:5000 in your browser
"""

import sys
import time
import traceback
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from shared.config import FLASK_PORT, FLASK_DEBUG
from shared.utils import setup_logging

logger = setup_logging("web_app")

# =============================================================================
# Flask Application Setup
# =============================================================================

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests for API flexibility

# Global pipeline reference — initialized on first request or at startup
pipeline = None


def get_pipeline():
    """
    Lazy-initialize the RAG pipeline.

    We use lazy initialization so the web server starts quickly,
    and the heavy model loading only happens when the first query arrives.
    """
    global pipeline
    if pipeline is None:
        logger.info("🔧 Initializing RAG pipeline (first request)...")
        try:
            from deliverable_1.src.rag_pipeline import RAGPipeline
            pipeline = RAGPipeline()
            logger.info("✅ RAG pipeline ready!")
        except Exception as e:
            logger.error("❌ Failed to initialize pipeline: %s", e)
            raise
    return pipeline


# =============================================================================
# Routes
# =============================================================================


@app.route("/")
def index():
    """Serve the main chat interface."""
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query():
    """
    Process a legal query through the RAG pipeline.

    Request body: {"query": "What is the punishment for theft?"}

    Response: {
        "answer": "...",
        "sources": [...],
        "confidence": 0.85,
        "hallucination_flags": [],
        "timing": {"retrieval": 0.23, "generation": 1.45, "total": 1.68}
    }
    """
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' field in request body"}), 400

        user_query = data["query"].strip()
        if not user_query:
            return jsonify({"error": "Query cannot be empty"}), 400

        logger.info("📝 Query received: %s", user_query[:100])

        # Get the pipeline (lazy init)
        rag = get_pipeline()

        # Run the RAG pipeline
        response = rag.query(user_query)

        # Format sources for the frontend
        sources = []
        for result in response.sources:
            sources.append({
                "text": result.chunk.text,
                "law_name": result.chunk.metadata.get("law_name", "Unknown"),
                "article_number": result.chunk.metadata.get("article_number", "N/A"),
                "chapter": result.chunk.metadata.get("chapter", ""),
                "score": round(result.score, 4),
            })

        return jsonify({
            "answer": response.answer,
            "sources": sources,
            "confidence": round(response.confidence, 3),
            "hallucination_flags": response.hallucination_flags,
            "timing": {
                "retrieval": round(response.retrieval_time, 3),
                "generation": round(response.generation_time, 3),
                "total": round(response.total_time, 3),
            },
        })

    except Exception as e:
        logger.error("❌ Error processing query: %s", traceback.format_exc())
        return jsonify({
            "error": "An error occurred while processing your query.",
            "details": str(e),
        }), 500


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Better Call Saul AI",
        "pipeline_loaded": pipeline is not None,
    })


@app.route("/api/stats")
def stats():
    """Return vector store statistics."""
    try:
        rag = get_pipeline()
        vs_stats = rag.vector_store.get_collection_stats()
        return jsonify({
            "vector_store": vs_stats,
            "pipeline_loaded": True,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("⚖️  BETTER CALL SAUL AI — Web Interface")
    logger.info("=" * 60)
    logger.info("Starting Flask server on http://localhost:%d", FLASK_PORT)
    logger.info("Press Ctrl+C to stop\n")

    app.run(
        host="0.0.0.0",
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
    )
