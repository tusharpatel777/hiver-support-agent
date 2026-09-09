import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pydantic import BaseModel, Field
from src.config import (
    POLICY_SENSITIVE_INTENTS,
    SENSITIVE_KEYWORDS,
    REASON_LOW_INTENT_CONFIDENCE,
    REASON_POLICY_SENSITIVE_INTENT,
    REASON_NO_GROUNDING_EVIDENCE,
    REASON_LOW_CONFIDENCE_DRAFT,
    REASON_POLICY_TRIGGER_KEYWORD,
    REASON_NONE,
    DEFAULT_INTENT_CONFIDENCE_THRESHOLD,
    DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD,
    DEFAULT_DRAFT_CONFIDENCE_THRESHOLD,
    config
)


class RoutingDecision(BaseModel):
    decision: str = Field(description="'auto_handle' or 'escalate_to_human'")
    reason: str = Field(description="Deterministic reason string explaining the decision")
    confidence_score: float = Field(ge=0.0, le=1.0)
    risk_level: str = Field(default="low", description="'low', 'medium', 'high'")
    signals_evaluated: Dict[str, Any] = Field(default_factory=dict)


class RoutingPolicy:
    """
    Deterministic, rule-informed routing engine.
    """
    def __init__(
        self,
        intent_conf_threshold: float = DEFAULT_INTENT_CONFIDENCE_THRESHOLD,
        retrieval_sim_threshold: float = DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD,
        draft_conf_threshold: float = DEFAULT_DRAFT_CONFIDENCE_THRESHOLD
    ):
        self.intent_conf_threshold = intent_conf_threshold
        self.retrieval_sim_threshold = retrieval_sim_threshold
        self.draft_conf_threshold = draft_conf_threshold

    def evaluate(
        self,
        customer_tweet: str,
        predicted_intent: str,
        intent_confidence: float,
        max_retrieval_similarity: float,
        draft_confidence: float,
        contains_hedge: bool = False
    ) -> RoutingDecision:
        """
        Evaluates signals in priority order to determine routing.
        """
        signals = {
            "predicted_intent": predicted_intent,
            "intent_confidence": intent_confidence,
            "intent_conf_threshold": self.intent_conf_threshold,
            "max_retrieval_similarity": max_retrieval_similarity,
            "retrieval_sim_threshold": self.retrieval_sim_threshold,
            "draft_confidence": draft_confidence,
            "draft_conf_threshold": self.draft_conf_threshold,
            "contains_hedge": contains_hedge
        }

        tweet_lower = customer_tweet.lower()

        # Rule 1: High-Risk Policy Keywords (Legal, Fraud, Lawsuit, Police, Stolen)
        for kw in SENSITIVE_KEYWORDS:
            if kw in tweet_lower:
                return RoutingDecision(
                    decision="escalate_to_human",
                    reason=REASON_POLICY_TRIGGER_KEYWORD,
                    confidence_score=0.99,
                    risk_level="high",
                    signals_evaluated=signals
                )

        # Rule 2: Hard Policy Sensitive Intents (Account Security / Compromise, Billing Disputes)
        if predicted_intent in POLICY_SENSITIVE_INTENTS:
            return RoutingDecision(
                decision="escalate_to_human",
                reason=REASON_POLICY_SENSITIVE_INTENT,
                confidence_score=0.95,
                risk_level="high",
                signals_evaluated=signals
            )

        # Rule 3: Low Intent Confidence (Ambiguous or unclassifiable intent)
        if intent_confidence < self.intent_conf_threshold:
            return RoutingDecision(
                decision="escalate_to_human",
                reason=REASON_LOW_INTENT_CONFIDENCE,
                confidence_score=round(1.0 - intent_confidence, 3),
                risk_level="medium",
                signals_evaluated=signals
            )

        # Rule 4: Insufficient Grounding Evidence (Retrieval similarity below threshold)
        if max_retrieval_similarity < self.retrieval_sim_threshold:
            return RoutingDecision(
                decision="escalate_to_human",
                reason=REASON_NO_GROUNDING_EVIDENCE,
                confidence_score=round(1.0 - max_retrieval_similarity, 3),
                risk_level="medium",
                signals_evaluated=signals
            )

        # Rule 5: Draft Uncertainty / Apology-only Hedge
        if draft_confidence < self.draft_conf_threshold or contains_hedge:
            return RoutingDecision(
                decision="escalate_to_human",
                reason=REASON_LOW_CONFIDENCE_DRAFT,
                confidence_score=round(1.0 - draft_confidence, 3),
                risk_level="medium",
                signals_evaluated=signals
            )

        # Safe for Autonomous Handling
        return RoutingDecision(
            decision="auto_handle",
            reason=REASON_NONE,
            confidence_score=round((intent_confidence + max_retrieval_similarity + draft_confidence) / 3.0, 3),
            risk_level="low",
            signals_evaluated=signals
        )


if __name__ == "__main__":
    router = RoutingPolicy()
    
    # Test case 1: Standard tracking question
    d1 = router.evaluate(
        customer_tweet="Where is my order #102-8472910?",
        predicted_intent="order_status_tracking",
        intent_confidence=0.92,
        max_retrieval_similarity=0.75,
        draft_confidence=0.88,
        contains_hedge=False
    )
    print(f"Test 1: {d1.decision} (Reason: {d1.reason}, Risk: {d1.risk_level})")

    # Test case 2: Account security compromise
    d2 = router.evaluate(
        customer_tweet="Someone hacked my account and changed the password!",
        predicted_intent="account_access_security",
        intent_confidence=0.96,
        max_retrieval_similarity=0.80,
        draft_confidence=0.90,
        contains_hedge=False
    )
    print(f"Test 2: {d2.decision} (Reason: {d2.reason}, Risk: {d2.risk_level})")

    # Test case 3: Low confidence ambiguous tweet
    d3 = router.evaluate(
        customer_tweet="Hmm, strange stuff happening with things.",
        predicted_intent="general_feedback_other",
        intent_confidence=0.45,
        max_retrieval_similarity=0.30,
        draft_confidence=0.50,
        contains_hedge=True
    )
    print(f"Test 3: {d3.decision} (Reason: {d3.reason}, Risk: {d3.risk_level})")
