import sys
import pytest
from pathlib import Path

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import INTENT_TAXONOMY, REASON_POLICY_SENSITIVE_INTENT, REASON_POLICY_TRIGGER_KEYWORD, REASON_NONE
from src.classifier import BaselineIntentClassifier, LLMIntentClassifier
from src.retrieval import GroundingRetriever
from src.drafter import ReplyDrafter
from src.router import RoutingPolicy
from src.pipeline import ProductionAgent, SimpleBaseline, TrivialBaseline
from src.judge import LLMJudge, compute_judge_human_agreement


def test_intent_taxonomy_structure():
    assert len(INTENT_TAXONOMY) == 8
    assert "order_status_tracking" in INTENT_TAXONOMY
    assert "account_access_security" in INTENT_TAXONOMY
    assert "billing_and_payment_dispute" in INTENT_TAXONOMY


def test_baseline_intent_classifier():
    clf = BaselineIntentClassifier()
    res = clf.predict("Where is my order #112-9847192? Tracking says delayed.")
    assert res.intent in INTENT_TAXONOMY
    assert 0.0 <= res.confidence <= 1.0
    assert res.reasoning is not None


def test_llm_intent_classifier_security():
    clf = LLMIntentClassifier()
    res = clf.predict("Someone in another country hacked my Amazon account and changed my password!")
    assert res.intent == "account_access_security"
    assert res.confidence >= 0.90


def test_retrieval_intent_filtering():
    retriever = GroundingRetriever()
    query = "Where is my order #112-8947281?"
    intent = "order_status_tracking"
    exemplars = retriever.retrieve(query=query, predicted_intent=intent, top_k=3, filter_by_intent=True)
    assert len(exemplars) > 0
    for ex in exemplars:
        assert ex.intent == intent
        assert 0.0 <= ex.similarity_score <= 1.0


def test_reply_drafter():
    retriever = GroundingRetriever()
    drafter = ReplyDrafter()
    query = "Where is my package? Order 112-9847192 is missing."
    intent = "order_status_tracking"
    exemplars = retriever.retrieve(query=query, predicted_intent=intent, top_k=2)
    
    draft_res = drafter.draft(customer_tweet=query, predicted_intent=intent, exemplars=exemplars)
    assert len(draft_res.draft_reply) > 20
    assert len(draft_res.evidence_ids) > 0
    assert "http" in draft_res.draft_reply or "amzn.to" in draft_res.draft_reply


def test_router_sensitive_policy():
    router = RoutingPolicy()
    
    # Sensitive intent
    dec = router.evaluate(
        customer_tweet="Help with login OTP",
        predicted_intent="account_access_security",
        intent_confidence=0.95,
        max_retrieval_similarity=0.80,
        draft_confidence=0.90
    )
    assert dec.decision == "escalate_to_human"
    assert dec.reason == REASON_POLICY_SENSITIVE_INTENT

    # Sensitive keyword
    dec_kw = router.evaluate(
        customer_tweet="I will sue you in a lawsuit over this refund!",
        predicted_intent="refund_and_cancellation",
        intent_confidence=0.95,
        max_retrieval_similarity=0.80,
        draft_confidence=0.90
    )
    assert dec_kw.decision == "escalate_to_human"
    assert dec_kw.reason == REASON_POLICY_TRIGGER_KEYWORD

    # Safe auto-handle
    dec_safe = router.evaluate(
        customer_tweet="Where is my order #102-9847192?",
        predicted_intent="order_status_tracking",
        intent_confidence=0.92,
        max_retrieval_similarity=0.75,
        draft_confidence=0.90
    )
    assert dec_safe.decision == "auto_handle"
    assert dec_safe.reason == REASON_NONE


def test_end_to_end_production_agent():
    agent = ProductionAgent()
    res = agent.process("Where is my package #112-9847192? Was supposed to arrive yesterday.")
    assert res.system_name == "Production_Agent"
    assert res.intent in INTENT_TAXONOMY
    assert res.decision in ["auto_handle", "escalate_to_human"]
    assert len(res.draft_reply) > 10


def test_llm_judge_and_agreement():
    judge = LLMJudge()
    score = judge.score(
        customer_tweet="Where is my order?",
        predicted_intent="order_status_tracking",
        draft_reply="You can track your order at https://amzn.to/YourOrders or DM us.",
        retrieved_evidence=[{"historical_brand_resolution": "Track at https://amzn.to/YourOrders."}]
    )
    assert 1 <= score.factual_grounding <= 5
    assert 1 <= score.tone_and_brand_voice <= 5
    assert 1.0 <= score.overall_score <= 5.0

    agr = compute_judge_human_agreement([4.5, 5.0, 4.0], [5.0, 5.0, 4.0])
    assert agr["percent_within_1_point"] == 100.0
