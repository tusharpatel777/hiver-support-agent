import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Set

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVAL_DIR = BASE_DIR / "eval"
REPORT_DIR = BASE_DIR / "report"

# Data file paths
RAW_TWCS_PATH = RAW_DATA_DIR / "twcs.csv"
PROCESSED_THREADS_PATH = PROCESSED_DATA_DIR / "threads.jsonl"
GOLDEN_SET_PATH = DATA_DIR / "golden_set.jsonl"
EVAL_METRICS_PATH = EVAL_DIR / "metrics_report.json"
HUMAN_JUDGE_SCORES_PATH = EVAL_DIR / "human_judge_scores.csv"

# Target Brand
TARGET_BRAND = "AmazonHelp"

# Intent Taxonomy (8 bottom-up classes derived from AmazonHelp dataset)
INTENT_TAXONOMY: List[str] = [
    "order_status_tracking",
    "shipping_delay_delivery_issue",
    "refund_and_cancellation",
    "damaged_or_wrong_item",
    "account_access_security",
    "billing_and_payment_dispute",
    "subscription_and_digital_services",
    "general_feedback_other"
]

# Taxonomy Descriptions
INTENT_DESCRIPTIONS = {
    "order_status_tracking": "Inquiries about order location, tracking numbers, dispatch status, or estimated delivery date.",
    "shipping_delay_delivery_issue": "Complaints about late shipments, missed delivery windows, courier delays, or marked delivered but not received.",
    "refund_and_cancellation": "Requests to cancel orders, check refund status, or return eligible items for money back.",
    "damaged_or_wrong_item": "Reports of receiving broken, defective, expired, missing, or incorrect products.",
    "account_access_security": "Issues with login, OTP/2FA, locked accounts, password reset, suspected unauthorized access or fraud.",
    "billing_and_payment_dispute": "Double charges, unrecognized credit/debit card transactions, failed payment methods, or currency/tax disputes.",
    "subscription_and_digital_services": "Inquiries regarding Amazon Prime membership, Kindle Unlimited, Prime Video streaming, or digital gift cards.",
    "general_feedback_other": "General praise, policy questions, app/website bugs, or complaints not covered by specific categories."
}

# Policy-Sensitive Intents (Mandatory Escalation to Human Agent)
POLICY_SENSITIVE_INTENTS: Set[str] = {
    "account_access_security",
    "billing_and_payment_dispute"
}

# Escalation Reason Codes
REASON_LOW_INTENT_CONFIDENCE = "low_intent_confidence"
REASON_POLICY_SENSITIVE_INTENT = "policy_sensitive_intent"
REASON_NO_GROUNDING_EVIDENCE = "no_grounding_evidence"
REASON_LOW_CONFIDENCE_DRAFT = "low_confidence_draft"
REASON_POLICY_TRIGGER_KEYWORD = "policy_trigger_keyword"
REASON_NONE = "none"

# Escalation Trigger Keywords (e.g. legal, lawsuit, fraud, chargeback, police)
SENSITIVE_KEYWORDS = [
    "lawsuit", "lawyer", "attorney", "legal action", "sue you", 
    "fraud", "police", "chargeback", "stolen identity", "unauthorized charge"
]

# Routing Policy Thresholds
DEFAULT_INTENT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD = 0.40
DEFAULT_DRAFT_CONFIDENCE_THRESHOLD = 0.65

# Cost matrix for routing decisions (Missed Escalation is 5x worse than False Escalation)
COST_FALSE_ESCALATION = 1.0   # Route to human when auto-handle was safe (wasted human agent time)
COST_MISSED_ESCALATION = 5.0  # Auto-handled when human escalation was required (brand risk, severe error)
COST_CORRECT_AUTO_HANDLE = 0.0
COST_CORRECT_ESCALATION = 0.0

class AgentConfig(BaseModel):
    brand: str = TARGET_BRAND
    intent_confidence_threshold: float = Field(default=DEFAULT_INTENT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
    retrieval_similarity_threshold: float = Field(default=DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD, ge=0.0, le=1.0)
    draft_confidence_threshold: float = Field(default=DEFAULT_DRAFT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
    top_k_retrieval: int = Field(default=3, ge=1, le=10)
    use_mock_llm: bool = False
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto") # auto, openai, gemini, anthropic, or local_rules
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

config = AgentConfig()
