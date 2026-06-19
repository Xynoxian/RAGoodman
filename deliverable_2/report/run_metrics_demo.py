import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deliverable_2.src.hallucination_detector import HallucinationDetector
from deliverable_2.src.evaluator import RAGEvaluator

det = HallucinationDetector()

context = (
    "Article 37: The penalty for theft shall be imprisonment for a term not less than "
    "three years and not exceeding seven years and a fine not less than AED 5,000. "
    "Article 38: The theft shall be considered aggravated and the punishment shall be "
    "imprisonment for a period not less than five years if committed at night, by two "
    "or more persons, or with a weapon."
)

faithful = (
    "Under Article 37, theft in the UAE carries imprisonment of 3 to 7 years and a "
    "fine of at least AED 5,000. Aggravated theft under Article 38 carries a minimum of 5 years."
)
hallucinated = (
    "Under Article 999, theft carries a 20 year sentence and a fine of AED 1,000,000. "
    "Article 1000 adds additional penalties for repeat offenders."
)

print("=== FAITHFUL RESPONSE ===")
r1 = det.check_faithfulness(faithful, context)
print(f"  is_faithful:     {r1['is_faithful']}")
print(f"  confidence:      {r1['confidence']}")
print(f"  verified claims: {len(r1['verified_claims'])}")
print(f"  flagged claims:  {len(r1['flagged_claims'])}")

print()
print("=== HALLUCINATED RESPONSE ===")
r2 = det.check_faithfulness(hallucinated, context)
print(f"  is_faithful:     {r2['is_faithful']}")
print(f"  confidence:      {r2['confidence']}")
print(f"  verified claims: {len(r2['verified_claims'])}")
print(f"  flagged claims:  {len(r2['flagged_claims'])}")
print(f"  flagged:         {r2['flagged_claims']}")

ev = RAGEvaluator()
print()
print("=== METRIC CALCULATIONS (live) ===")
print(f"  Faithfulness (faithful):         {ev.evaluate_faithfulness(faithful, context):.4f}")
print(f"  Faithfulness (hallucinated):     {ev.evaluate_faithfulness(hallucinated, context):.4f}")
print(f"  Relevancy (theft query):         {ev.evaluate_relevancy(faithful, 'What is the punishment for theft?'):.4f}")
print(f"  Citation accuracy (faithful):    {ev.evaluate_citation_accuracy(faithful, context):.4f}")
print(f"  Citation accuracy (hallucinated):{ev.evaluate_citation_accuracy(hallucinated, context):.4f}")
