"""
Hallucination Detector — Faithfulness Verification (BONUS Feature)
===================================================================

A critical concern in RAG systems is **hallucination** — the LLM generating
claims that are not grounded in the retrieved context.  In legal applications
this is especially dangerous: a fabricated article number could lead to
incorrect legal advice.

This module provides a **local, LLM-free** hallucination detector that uses
a combination of:

1. **Claim extraction** — Splits the LLM response into atomic factual claims
   using rule-based heuristics (no LLM needed).
2. **Keyword verification** — Checks if key legal terms (article numbers,
   law names, penalties) mentioned in claims appear in the source context.
3. **Semantic similarity** — Uses embedding cosine similarity to check if
   the *meaning* of each claim is supported by the context.

This avoids the need for an external NLI (Natural Language Inference) model
or additional LLM calls, keeping the system self-contained and fast.

Usage::

    from deliverable_2.src.hallucination_detector import HallucinationDetector
    detector = HallucinationDetector()
    report = detector.get_hallucination_report(llm_response, context)
    print(report["is_faithful"])  # True/False
"""

import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils import setup_logging

logger = setup_logging(__name__)


class HallucinationDetector:
    """Detects potential hallucinations in LLM-generated legal responses.

    **Why is this important?**

    In legal RAG, the LLM might:
    - Cite an article number that doesn't exist in the context
    - State a penalty that differs from what the law actually says
    - Mix up provisions from different laws
    - Generate plausible-sounding but fabricated legal provisions

    This detector catches these issues using a combination of keyword
    matching and semantic similarity — no external NLI model required.

    Attributes:
        embedder: Optional embedding model for semantic similarity checks.
            If None, only keyword-based verification is used.
        keyword_weight: Weight for keyword-based verification score (0-1).
        semantic_weight: Weight for semantic similarity score (0-1).
        confidence_threshold: Minimum verification score to consider a
            claim as "supported" by the context.
    """

    # Regex patterns for extracting legal references from text
    ARTICLE_REF_PATTERN = re.compile(
        r"(?:Article|Art\.?|المادة)\s*\(?(\d+)\)?",
        re.IGNORECASE | re.UNICODE,
    )
    LAW_REF_PATTERN = re.compile(
        r"(?:UAE\s+)?(?:Federal\s+)?(?:Law|Decree|Code|Act)\s+(?:No\.?\s*)?(\d+)?[^.]*?(?:of\s+\d{4})?",
        re.IGNORECASE,
    )
    PENALTY_PATTERN = re.compile(
        r"(?:imprison(?:ment|ed)?|fine[sd]?|penalty|punish(?:ment|ed)?|sentenced?)"
        r"[^.]*?(?:\d+[^.]*?(?:years?|months?|days?|dirhams?|AED))",
        re.IGNORECASE,
    )
    NUMBER_PATTERN = re.compile(r"\b\d+(?:,\d+)*(?:\.\d+)?\b")

    def __init__(
        self,
        embedder=None,
        keyword_weight: float = 0.5,
        semantic_weight: float = 0.5,
        confidence_threshold: float = 0.4,
    ) -> None:
        """Initialise the HallucinationDetector.

        Args:
            embedder: An embedding model with ``embed_text(str) -> list[float]``
                method.  If None, only keyword verification is used and
                semantic_weight is redistributed to keyword_weight.
            keyword_weight: Weight for keyword-based verification (0-1).
            semantic_weight: Weight for semantic similarity verification (0-1).
            confidence_threshold: Minimum verification score for a claim
                to be considered "supported".
        """
        self.embedder = embedder
        self.confidence_threshold = confidence_threshold

        # If no embedder, redirect all weight to keyword matching
        if embedder is None:
            self.keyword_weight = 1.0
            self.semantic_weight = 0.0
            logger.info(
                "HallucinationDetector initialised (keyword-only mode, threshold=%.2f)",
                confidence_threshold,
            )
        else:
            self.keyword_weight = keyword_weight
            self.semantic_weight = semantic_weight
            logger.info(
                "HallucinationDetector initialised (kw_weight=%.2f, sem_weight=%.2f, threshold=%.2f)",
                keyword_weight, semantic_weight, confidence_threshold,
            )

    # ==================================================================
    # Step 1: Claim Extraction
    # ==================================================================

    def extract_claims(self, response: str) -> List[str]:
        """Split an LLM response into atomic factual claims.

        An "atomic claim" is a single verifiable statement.  For example:
            Input:  "Article 461 imposes up to 3 years imprisonment for theft,
                     while Article 462 covers aggravated theft with 7 years."
            Output: ["Article 461 imposes up to 3 years imprisonment for theft",
                     "Article 462 covers aggravated theft with 7 years"]

        The extraction uses heuristic rules:
        - Split on sentence boundaries
        - Split compound sentences on conjunctions ("and", "while", "however")
        - Filter out non-factual content (greetings, disclaimers, questions)

        Args:
            response: The LLM-generated response text.

        Returns:
            List of atomic claim strings.
        """
        if not response or not response.strip():
            return []

        # Step 1a: Remove markdown formatting
        clean_text = re.sub(r"\*\*|__|~~|`{1,3}", "", response)
        # Remove bullet points and numbered list markers
        clean_text = re.sub(r"^\s*[-•*]\s+", "", clean_text, flags=re.MULTILINE)
        clean_text = re.sub(r"^\s*\d+[.)]\s+", "", clean_text, flags=re.MULTILINE)

        # Step 1b: Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)

        # Step 1c: Split compound sentences on coordinating conjunctions
        atomic_claims: List[str] = []
        for sentence in sentences:
            # Split on conjunctions that introduce new facts
            sub_claims = re.split(
                r'\s*(?:,\s*(?:and|while|whereas|however|but|although|moreover|furthermore)\s+)',
                sentence,
                flags=re.IGNORECASE,
            )
            for claim in sub_claims:
                claim = claim.strip().rstrip(".")
                if claim:
                    atomic_claims.append(claim)

        # Step 1d: Filter out non-factual content
        filtered = []
        non_factual_patterns = [
            r"^(?:hi|hello|hey|dear|thank|please|hope|feel free)",  # Greetings
            r"disclaimer|not (?:a )?(?:legal|professional) advice",  # Disclaimers
            r"^(?:I'm|I am|as an? (?:AI|assistant))",  # AI self-references
            r"^(?:sure|of course|certainly|absolutely)",  # Filler
            r"\?$",  # Questions
        ]

        for claim in atomic_claims:
            is_factual = True
            for pattern in non_factual_patterns:
                if re.search(pattern, claim, re.IGNORECASE):
                    is_factual = False
                    break

            # Only keep claims with some substance (> 15 chars)
            if is_factual and len(claim) > 15:
                filtered.append(claim)

        logger.debug("Extracted %d claims from response (%d chars).", len(filtered), len(response))
        return filtered

    # ==================================================================
    # Step 2: Individual Claim Verification
    # ==================================================================

    def verify_claim(self, claim: str, context: str) -> Dict:
        """Verify whether a single claim is supported by the context.

        Combines two verification methods:
        1. **Keyword verification** — checks if key terms (article numbers,
           law names, specific numbers) from the claim appear in the context.
        2. **Semantic verification** — checks if the claim's embedding is
           similar to any part of the context (if embedder is available).

        Args:
            claim: A single atomic claim string.
            context: The source context (concatenated retrieved chunks).

        Returns:
            Dict with keys:
                - ``claim``: The original claim text.
                - ``is_supported``: Boolean — True if verification score
                  exceeds the confidence threshold.
                - ``confidence``: Float [0, 1] — combined verification score.
                - ``keyword_score``: Float [0, 1] — keyword match score.
                - ``semantic_score``: Float [0, 1] — semantic similarity score.
                - ``details``: List of strings explaining verification results.
        """
        details: List[str] = []

        # --- Keyword verification ---
        keyword_score = self._keyword_verify(claim, context, details)

        # --- Semantic verification ---
        semantic_score = self._semantic_verify(claim, context, details)

        # --- Combined score ---
        combined_score = (
            self.keyword_weight * keyword_score +
            self.semantic_weight * semantic_score
        )

        is_supported = combined_score >= self.confidence_threshold

        result = {
            "claim": claim,
            "is_supported": is_supported,
            "confidence": round(combined_score, 4),
            "keyword_score": round(keyword_score, 4),
            "semantic_score": round(semantic_score, 4),
            "details": details,
        }

        logger.debug(
            "Claim verification: supported=%s, confidence=%.4f — '%s'",
            is_supported, combined_score, claim[:80],
        )
        return result

    def _keyword_verify(
        self, claim: str, context: str, details: List[str]
    ) -> float:
        """Verify a claim using keyword/pattern matching.

        Checks multiple categories:
        1. Article number references — do they appear in the context?
        2. Key legal terms — are the important words found?
        3. Specific numbers (penalties, durations) — are they accurate?

        Args:
            claim: The claim to verify.
            context: Source context to check against.
            details: List to append verification detail messages to.

        Returns:
            Score in [0, 1] based on keyword matching.
        """
        scores: List[float] = []
        context_lower = context.lower()

        # Check 1: Article references
        article_refs = self.ARTICLE_REF_PATTERN.findall(claim)
        if article_refs:
            found = sum(1 for ref in article_refs if ref in context)
            article_score = found / len(article_refs) if article_refs else 1.0
            scores.append(article_score)
            if article_score < 1.0:
                missing = [ref for ref in article_refs if ref not in context]
                details.append(f"⚠️ Article reference(s) not found in context: {missing}")
            else:
                details.append(f"✓ Article reference(s) verified: {article_refs}")

        # Check 2: Key legal terms (extract important words from the claim)
        # Focus on nouns and domain-specific terms
        claim_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', claim.lower()))
        # Remove common stop words
        stop_words = {
            "this", "that", "with", "from", "have", "been", "were", "will",
            "shall", "would", "could", "should", "into", "upon", "also",
            "such", "than", "other", "more", "most", "only", "each", "every",
            "some", "according", "under", "above", "below", "between",
        }
        claim_words -= stop_words

        if claim_words:
            found_words = sum(1 for w in claim_words if w in context_lower)
            term_score = found_words / len(claim_words)
            scores.append(term_score)
            if term_score < 0.5:
                details.append(f"⚠️ Only {found_words}/{len(claim_words)} key terms found in context")
        else:
            scores.append(0.5)  # Neutral if no key terms extracted

        # Check 3: Specific numbers (penalties, durations, amounts)
        claim_numbers = set(self.NUMBER_PATTERN.findall(claim))
        if claim_numbers:
            found_numbers = sum(1 for n in claim_numbers if n in context)
            number_score = found_numbers / len(claim_numbers)
            scores.append(number_score)
            if number_score < 1.0:
                missing_nums = [n for n in claim_numbers if n not in context]
                details.append(f"⚠️ Number(s) not found in context: {missing_nums}")
            else:
                details.append(f"✓ Numeric references verified: {list(claim_numbers)}")

        # Average all sub-scores
        return sum(scores) / len(scores) if scores else 0.5

    def _semantic_verify(
        self, claim: str, context: str, details: List[str]
    ) -> float:
        """Verify a claim using semantic (embedding) similarity.

        Splits the context into paragraphs and checks if any paragraph
        is semantically similar to the claim.

        Args:
            claim: The claim to verify.
            context: Source context to check against.
            details: List to append verification detail messages to.

        Returns:
            Maximum cosine similarity between claim and any context paragraph.
            Returns 0.5 (neutral) if no embedder is available.
        """
        if self.embedder is None:
            return 0.5  # Neutral score when no embedder available

        try:
            import numpy as np

            # Split context into paragraphs for granular comparison
            paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [p.strip() for p in context.split("\n") if len(p.strip()) > 20]

            if not paragraphs:
                details.append("⚠️ No context paragraphs for semantic verification")
                return 0.5

            # Embed the claim
            claim_embedding = np.array(self.embedder.embed_text(claim))

            # Find the most similar paragraph
            max_similarity = 0.0
            for para in paragraphs:
                para_embedding = np.array(self.embedder.embed_text(para))

                # Cosine similarity
                norm_claim = np.linalg.norm(claim_embedding)
                norm_para = np.linalg.norm(para_embedding)
                if norm_claim > 0 and norm_para > 0:
                    sim = float(np.dot(claim_embedding, para_embedding) / (norm_claim * norm_para))
                    max_similarity = max(max_similarity, sim)

            details.append(f"Semantic similarity: {max_similarity:.4f}")
            return max_similarity

        except Exception as exc:
            logger.warning("Semantic verification failed: %s", exc)
            details.append(f"⚠️ Semantic verification failed: {exc}")
            return 0.5  # Neutral on failure

    # ==================================================================
    # Step 3: Full Faithfulness Check
    # ==================================================================

    def check_faithfulness(self, response: str, context: str) -> Dict:
        """Check whether an entire response is faithful to the context.

        This is the main entry point for simple yes/no faithfulness checking.
        It extracts claims, verifies each one, and produces an overall verdict.

        Args:
            response: The LLM-generated response text.
            context: The source context (concatenated retrieved chunks).

        Returns:
            Dict with keys:
                - ``is_faithful``: Boolean — True if the majority of claims
                  are supported.
                - ``confidence``: Float [0, 1] — overall faithfulness score.
                - ``flagged_claims``: List of claim strings that were NOT
                  supported by the context.
                - ``verified_claims``: List of claim strings that WERE
                  supported by the context.
                - ``total_claims``: Total number of claims extracted.
        """
        claims = self.extract_claims(response)

        if not claims:
            logger.info("No verifiable claims extracted — treating as faithful.")
            return {
                "is_faithful": True,
                "confidence": 1.0,
                "flagged_claims": [],
                "verified_claims": [],
                "total_claims": 0,
            }

        verified_claims: List[str] = []
        flagged_claims: List[str] = []
        total_confidence = 0.0

        for claim in claims:
            result = self.verify_claim(claim, context)
            total_confidence += result["confidence"]

            if result["is_supported"]:
                verified_claims.append(claim)
            else:
                flagged_claims.append(claim)

        # Overall faithfulness: ratio of supported claims
        avg_confidence = total_confidence / len(claims)
        # is_faithful if >70% of claims are supported
        faithfulness_ratio = len(verified_claims) / len(claims)
        is_faithful = faithfulness_ratio >= 0.7

        logger.info(
            "Faithfulness check: %d/%d claims verified (%.0f%%), faithful=%s",
            len(verified_claims), len(claims), faithfulness_ratio * 100, is_faithful,
        )

        return {
            "is_faithful": is_faithful,
            "confidence": round(avg_confidence, 4),
            "flagged_claims": flagged_claims,
            "verified_claims": verified_claims,
            "total_claims": len(claims),
        }

    # ==================================================================
    # Step 4: Comprehensive Report
    # ==================================================================

    def get_hallucination_report(self, response: str, context: str) -> Dict:
        """Generate a comprehensive hallucination detection report.

        This is the detailed version of ``check_faithfulness``, providing
        per-claim verification results and additional analysis.

        Args:
            response: The LLM-generated response text.
            context: The source context (concatenated retrieved chunks).

        Returns:
            Dict with keys:
                - ``is_faithful``: Boolean overall verdict.
                - ``confidence``: Float [0, 1] overall faithfulness score.
                - ``flagged_claims``: List of unsupported claim strings.
                - ``verified_claims``: List of supported claim strings.
                - ``total_claims``: Total number of claims.
                - ``faithfulness_ratio``: Float — proportion of supported claims.
                - ``claim_details``: List of per-claim verification dicts
                  (each from ``verify_claim``).
                - ``risk_level``: 'low', 'medium', or 'high' hallucination risk.
                - ``article_references_check``: Dict with article ref verification.
                - ``summary``: Human-readable summary string.
        """
        claims = self.extract_claims(response)
        claim_details: List[Dict] = []
        verified_claims: List[str] = []
        flagged_claims: List[str] = []

        for claim in claims:
            result = self.verify_claim(claim, context)
            claim_details.append(result)

            if result["is_supported"]:
                verified_claims.append(claim)
            else:
                flagged_claims.append(claim)

        # Calculate metrics
        total = len(claims)
        faithfulness_ratio = len(verified_claims) / total if total > 0 else 1.0
        avg_confidence = (
            sum(d["confidence"] for d in claim_details) / total if total > 0 else 1.0
        )

        # Determine risk level
        if faithfulness_ratio >= 0.9:
            risk_level = "low"
        elif faithfulness_ratio >= 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"

        # Check article references specifically
        article_refs_in_response = self.ARTICLE_REF_PATTERN.findall(response)
        article_refs_in_context = self.ARTICLE_REF_PATTERN.findall(context)
        unverified_articles = [
            ref for ref in article_refs_in_response
            if ref not in article_refs_in_context
        ]

        article_check = {
            "response_articles": article_refs_in_response,
            "context_articles": article_refs_in_context,
            "unverified_articles": unverified_articles,
            "all_verified": len(unverified_articles) == 0,
        }

        # Build human-readable summary
        if risk_level == "low":
            summary = (
                f"✅ Low hallucination risk: {len(verified_claims)}/{total} claims verified. "
                f"The response appears well-grounded in the source context."
            )
        elif risk_level == "medium":
            summary = (
                f"⚠️ Medium hallucination risk: {len(flagged_claims)}/{total} claims could not "
                f"be fully verified against the source context. Review flagged claims carefully."
            )
        else:
            summary = (
                f"🚨 High hallucination risk: {len(flagged_claims)}/{total} claims are not "
                f"supported by the source context. The response may contain fabricated information."
            )

        if unverified_articles:
            summary += f" ⚠️ Unverified article references: {unverified_articles}"

        report = {
            "is_faithful": faithfulness_ratio >= 0.7,
            "confidence": round(avg_confidence, 4),
            "flagged_claims": flagged_claims,
            "verified_claims": verified_claims,
            "total_claims": total,
            "faithfulness_ratio": round(faithfulness_ratio, 4),
            "claim_details": claim_details,
            "risk_level": risk_level,
            "article_references_check": article_check,
            "summary": summary,
        }

        logger.info(
            "Hallucination report: risk=%s, faithful=%.0f%%, %d flagged claims",
            risk_level, faithfulness_ratio * 100, len(flagged_claims),
        )
        return report


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("HallucinationDetector — Smoke Test")
    print("=" * 60)

    detector = HallucinationDetector()

    context = """
    Article 461: Whoever steals movable property owned by another shall be
    punished by imprisonment for a period not exceeding 3 years.

    Article 462: If the theft is committed at night, or by two or more persons,
    or with the use of violence, the penalty shall be imprisonment for a period
    not exceeding 7 years.
    """

    # Test 1: Faithful response
    print("\n--- Test 1: Faithful response ---")
    faithful_response = (
        "According to Article 461, the punishment for theft in the UAE is "
        "imprisonment for up to 3 years. For aggravated theft under Article 462, "
        "the penalty increases to up to 7 years."
    )
    result = detector.check_faithfulness(faithful_response, context)
    print(f"  Is faithful: {result['is_faithful']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Verified: {len(result['verified_claims'])}, Flagged: {len(result['flagged_claims'])}")

    # Test 2: Hallucinated response
    print("\n--- Test 2: Hallucinated response ---")
    hallucinated_response = (
        "Under Article 999, theft carries a penalty of 15 years imprisonment. "
        "The fine is 500,000 AED according to Article 1000."
    )
    result = detector.check_faithfulness(hallucinated_response, context)
    print(f"  Is faithful: {result['is_faithful']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Flagged claims: {result['flagged_claims']}")

    # Test 3: Full report
    print("\n--- Test 3: Comprehensive report ---")
    report = detector.get_hallucination_report(faithful_response, context)
    print(f"  Risk level: {report['risk_level']}")
    print(f"  Summary: {report['summary']}")
    print(f"  Article check: {report['article_references_check']}")

    print("\n✅ Smoke test completed!")
