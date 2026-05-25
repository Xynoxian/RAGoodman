"""
Generator Module - LLM Response Generation
============================================

Takes retrieved context chunks and a user query, formats them into a
prompt, sends to the configured LLM provider, and returns a response.
Supports Gemini, OpenAI, and Ollama backends.
"""

import json
import logging
import time
from typing import List, Tuple

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import (
    LLM_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OLLAMA_MODEL,
)
from shared.prompts import SAUL_SYSTEM_PROMPT, RAG_USER_PROMPT, HALLUCINATION_CHECK_PROMPT
from shared.utils import RetrievalResult

logger = logging.getLogger(__name__)


class Generator:
    """Generates answers using an LLM with retrieved context.

    Supports multiple backends:
        - **gemini** – Google Generative AI
        - **openai** – OpenAI Chat Completions
        - **ollama** – Local Ollama server

    Args:
        provider: One of ``"gemini"``, ``"openai"``, ``"ollama"``.
    """

    def __init__(self, provider: str = LLM_PROVIDER):
        self.provider = provider.lower()
        logger.info("Generator initialised with provider: %s", self.provider)

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _format_context(results: List[RetrievalResult]) -> str:
        """Format retrieval results into a numbered context block."""
        parts = []
        for i, r in enumerate(results, 1):
            meta = r.chunk.metadata
            source = meta.get("law_name", meta.get("source", "Unknown"))
            article = meta.get("article_number", "N/A")
            parts.append(
                f"[Source {i}] {source} – Article {article}\n{r.chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # LLM call – dispatches to the right backend
    # ------------------------------------------------------------------
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the configured LLM and return the response text."""
        if self.provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt)
        elif self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        elif self.provider == "ollama":
            return self._call_ollama(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Call Google Gemini API."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                GEMINI_MODEL,
                system_instruction=system_prompt,
            )
            response = model.generate_content(user_prompt)
            return response.text
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            raise

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI Chat Completions API."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.error("OpenAI API call failed: %s", exc)
            raise

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call local Ollama server."""
        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as exc:
            logger.error("Ollama API call failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
    ) -> Tuple[str, float]:
        """Generate an answer for *query* using retrieved context.

        Returns:
            Tuple of (answer_text, generation_time_seconds).
        """
        context = self._format_context(retrieval_results)
        user_prompt = RAG_USER_PROMPT.format(context=context, question=query)

        start = time.time()
        answer = self._call_llm(SAUL_SYSTEM_PROMPT, user_prompt)
        gen_time = time.time() - start

        logger.info("Generated answer in %.2fs (%d chars)", gen_time, len(answer))
        return answer, gen_time

    def check_hallucination(
        self,
        answer: str,
        retrieval_results: List[RetrievalResult],
    ) -> List[str]:
        """Check whether the answer is grounded in the retrieved context.

        Returns:
            List of hallucination flag strings (empty if fully grounded).
        """
        context = self._format_context(retrieval_results)
        prompt = HALLUCINATION_CHECK_PROMPT.format(context=context, answer=answer)

        try:
            raw = self._call_llm(
                "You are a fact-checking assistant. Respond only with valid JSON.",
                prompt,
            )
            # Try to parse JSON from the response
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            flags = json.loads(raw)
            if isinstance(flags, list):
                return [str(f) for f in flags]
            return []
        except Exception as exc:
            logger.warning("Hallucination check failed: %s", exc)
            return []
