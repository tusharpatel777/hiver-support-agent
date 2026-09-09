import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pydantic import BaseModel, Field

from src.config import TARGET_BRAND, REASON_NONE, config
from src.classifier import BaselineIntentClassifier, LLMIntentClassifier, IntentPrediction
from src.retrieval import GroundingRetriever, RetrievalExemplar
from src.drafter import ReplyDrafter, DraftOutput
from src.router import RoutingPolicy, RoutingDecision


class AgentResponse(BaseModel):
    system_name: str
    customer_tweet: str
    intent: str
    intent_confidence: float
    intent_reasoning: str
    draft_reply: str
    decision: str = Field(description="'auto_handle' or 'escalate_to_human'")
    reason: str
    evidence_ids: List[str]
    retrieval_exemplars: List[Dict[str, Any]] = Field(default_factory=list)
    max_retrieval_similarity: float = 0.0
    draft_confidence: float = 0.0
    risk_level: str = "low"


class TrivialBaseline:
    """
    Trivial Baseline (§6.3):
    Majority-class intent + static canned DM template + majority routing.
    Sets the minimum performance floor.
    """
    def __init__(self, brand_name: str = TARGET_BRAND):
        self.brand_name = brand_name
        self.majority_intent = "order_status_tracking"
        self.canned_reply = f"Thanks for reaching out to @{self.brand_name}! Please send us a direct message with your order details at https://amzn.to/help so we can assist."

    def process(self, tweet: str) -> AgentResponse:
        return AgentResponse(
            system_name="Trivial_Baseline",
            customer_tweet=tweet,
            intent=self.majority_intent,
            intent_confidence=0.50,
            intent_reasoning="Static majority-class baseline assumption.",
            draft_reply=self.canned_reply,
            decision="auto_handle",
            reason=REASON_NONE,
            evidence_ids=[],
            retrieval_exemplars=[],
            max_retrieval_similarity=0.0,
            draft_confidence=0.50,
            risk_level="low"
        )


class SimpleBaseline:
    """
    Simple Baseline (§6.3):
    TF-IDF/Logistic Regression intent classifier + nearest-neighbor verbatim historical resolution
    (without LLM generation) + simple confidence threshold routing.
    Isolates the exact incremental value of the LLM generation step.
    """
    def __init__(self):
        self.classifier = BaselineIntentClassifier()
        self.retriever = GroundingRetriever()

    def process(self, tweet: str) -> AgentResponse:
        # 1. Classical ML intent classification
        clf_pred = self.classifier.predict(tweet)
        
        # 2. Nearest neighbor retrieval (unfiltered global 1-NN)
        exemplars = self.retriever.retrieve(
            query=tweet,
            predicted_intent=clf_pred.intent,
            top_k=1,
            filter_by_intent=False
        )
        
        if exemplars:
            top_ex = exemplars[0]
            draft_reply = top_ex.historical_brand_resolution
            evidence_ids = [top_ex.evidence_id]
            max_sim = top_ex.similarity_score
            ex_list = [top_ex.model_dump()]
        else:
            draft_reply = "Please DM us at https://amzn.to/help."
            evidence_ids = []
            max_sim = 0.0
            ex_list = []

        # 3. Simple threshold routing: escalate if classifier confidence < 0.60
        if clf_pred.confidence < 0.60:
            decision = "escalate_to_human"
            reason = "low_intent_confidence"
        else:
            decision = "auto_handle"
            reason = REASON_NONE

        return AgentResponse(
            system_name="Simple_Baseline",
            customer_tweet=tweet,
            intent=clf_pred.intent,
            intent_confidence=clf_pred.confidence,
            intent_reasoning=clf_pred.reasoning,
            draft_reply=draft_reply,
            decision=decision,
            reason=reason,
            evidence_ids=evidence_ids,
            retrieval_exemplars=ex_list,
            max_retrieval_similarity=max_sim,
            draft_confidence=clf_pred.confidence,
            risk_level="medium" if decision == "escalate_to_human" else "low"
        )


class ProductionAgent:
    """
    Full Production Support Agent (§4):
    Few-shot LLM Intent Classification -> Intent-Filtered Vector Retrieval -> Grounded LLM Drafter -> Multi-Signal Router.
    """
    def __init__(
        self,
        intent_conf_threshold: float = config.intent_confidence_threshold,
        retrieval_sim_threshold: float = config.retrieval_similarity_threshold,
        draft_conf_threshold: float = config.draft_confidence_threshold,
        top_k: int = config.top_k_retrieval
    ):
        self.classifier = LLMIntentClassifier()
        self.retriever = GroundingRetriever()
        self.drafter = ReplyDrafter(brand_name=config.brand)
        self.router = RoutingPolicy(
            intent_conf_threshold=intent_conf_threshold,
            retrieval_sim_threshold=retrieval_sim_threshold,
            draft_conf_threshold=draft_conf_threshold
        )
        self.top_k = top_k

    def process(self, tweet: str, use_api: bool = True) -> AgentResponse:
        # Step 1: Intent Classification
        clf_pred: IntentPrediction = self.classifier.predict(tweet, use_api=use_api)

        # Step 2: Intent-Filtered Vector Retrieval (§4.2)
        exemplars: List[RetrievalExemplar] = self.retriever.retrieve(
            query=tweet,
            predicted_intent=clf_pred.intent,
            top_k=self.top_k,
            filter_by_intent=True
        )
        max_sim = self.retriever.get_max_similarity(exemplars)

        # Step 3: Grounded Reply Generation (§4.3)
        draft_out: DraftOutput = self.drafter.draft(
            customer_tweet=tweet,
            predicted_intent=clf_pred.intent,
            exemplars=exemplars,
            use_api=use_api
        )

        # Step 4: Multi-Signal Routing Policy (§4.4)
        routing_dec: RoutingDecision = self.router.evaluate(
            customer_tweet=tweet,
            predicted_intent=clf_pred.intent,
            intent_confidence=clf_pred.confidence,
            max_retrieval_similarity=max_sim,
            draft_confidence=draft_out.draft_confidence,
            contains_hedge=draft_out.contains_hedge
        )

        return AgentResponse(
            system_name="Production_Agent",
            customer_tweet=tweet,
            intent=clf_pred.intent,
            intent_confidence=clf_pred.confidence,
            intent_reasoning=clf_pred.reasoning,
            draft_reply=draft_out.draft_reply,
            decision=routing_dec.decision,
            reason=routing_dec.reason,
            evidence_ids=draft_out.evidence_ids,
            retrieval_exemplars=[ex.model_dump() for ex in exemplars],
            max_retrieval_similarity=max_sim,
            draft_confidence=draft_out.draft_confidence,
            risk_level=routing_dec.risk_level
        )


if __name__ == "__main__":
    agent = ProductionAgent()
    simple = SimpleBaseline()
    trivial = TrivialBaseline()

    sample = "Someone logged into my Amazon account and ordered $500 of gift cards with my credit card!"
    
    print("=" * 60)
    print(f"Customer Tweet: {sample}\n")
    
    print("[1] Trivial Baseline:")
    print(json.dumps(trivial.process(sample).model_dump(), indent=2))
    
    print("\n[2] Simple Baseline:")
    print(json.dumps(simple.process(sample).model_dump(), indent=2))
    
    print("\n[3] Production Agent:")
    print(json.dumps(agent.process(sample).model_dump(), indent=2))
