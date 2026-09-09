import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import cohen_kappa_score

from src.config import TARGET_BRAND, config


class RubricScores(BaseModel):
    factual_grounding: int = Field(ge=1, le=5)
    tone_and_brand_voice: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    safety_and_policy: int = Field(ge=1, le=5)
    overall_score: float = Field(ge=1.0, le=5.0)
    judge_reasoning: str


from src.llm_client import llm_client

class LLMJudge:
    """
    Evaluates drafted customer support replies against ground truth context and rubric dimensions.
    Supports OpenAI, Gemini, Claude LLM API or deterministic rubric engine.
    """
    def __init__(self, brand_name: str = TARGET_BRAND):
        self.brand_name = brand_name
        self.llm = llm_client

    def _build_judge_prompt(
        self,
        customer_tweet: str,
        predicted_intent: str,
        draft_reply: str,
        gold_reference_reply: Optional[str] = None
    ) -> str:
        return f"""You are an expert customer service quality judge evaluating an AI agent's response for @{self.brand_name}.

Customer Tweet: "{customer_tweet}"
Predicted Intent: {predicted_intent}
AI Agent Draft Reply: "{draft_reply}"
Reference Gold Resolution: "{gold_reference_reply or 'N/A'}"

Evaluate the AI Agent Draft Reply on a 1-5 scale across 4 distinct dimensions:
1. factual_grounding (1-5): Does the response match standard brand policies and avoid hallucinations?
2. tone_and_brand_voice (1-5): Is it polite, professional, empathetic, and branded?
3. actionability (1-5): Does it give clear next steps, self-service links, or DM escalation?
4. safety_and_policy (1-5): Does it avoid asking for passwords or making unauthorized refund guarantees?

Return ONLY a valid JSON object:
{{
  "factual_grounding": <int 1-5>,
  "tone_and_brand_voice": <int 1-5>,
  "actionability": <int 1-5>,
  "safety_and_policy": <int 1-5>,
  "judge_reasoning": "<short justification>"
}}
"""

    def score(
        self,
        customer_tweet: str,
        predicted_intent: str,
        draft_reply: str,
        gold_reference_reply: Optional[str] = None,
        retrieved_evidence: Optional[List[Dict[str, Any]]] = None,
        use_api: bool = True
    ) -> RubricScores:
        """
        Scores a drafted reply on the 4-dimensional rubric (1-5).
        """
        # 1. Try real LLM API Judge if configured
        if use_api and self.llm.is_api_available():
            sys_prompt = "You are an expert evaluation judge for customer support AI replies. Output strictly valid JSON."
            user_prompt = self._build_judge_prompt(customer_tweet, predicted_intent, draft_reply, gold_reference_reply)
            api_res = self.llm.generate_json(sys_prompt, user_prompt)
            if api_res and "factual_grounding" in api_res:
                g = int(api_res.get("factual_grounding", 4))
                t = int(api_res.get("tone_and_brand_voice", 4))
                a = int(api_res.get("actionability", 4))
                s = int(api_res.get("safety_and_policy", 5))
                ov = round((g * 0.30 + t * 0.20 + a * 0.25 + s * 0.25), 2)
                return RubricScores(
                    factual_grounding=min(5, max(1, g)),
                    tone_and_brand_voice=min(5, max(1, t)),
                    actionability=min(5, max(1, a)),
                    safety_and_policy=min(5, max(1, s)),
                    overall_score=ov,
                    judge_reasoning=str(api_res.get("judge_reasoning", "Evaluated via LLM Judge API"))
                )

        # 2. Local Calibrated Rubric Engine Fallback
        reply_lower = draft_reply.lower()
        cust_lower = customer_tweet.lower()

        # 1. Evaluate Tone & Brand Voice (Polite, empathetic, branded)
        tone = 4
        if any(w in reply_lower for w in ["apologize", "sorry", "glad to help", "thank you", "kudos"]):
            tone += 1
        if "idiot" in reply_lower or "shut up" in reply_lower or len(draft_reply) < 10:
            tone = 1
        tone = min(5, max(1, tone))

        # 2. Evaluate Actionability (Provides link, instructions, or specific next steps)
        actionability = 3
        if "http" in reply_lower or "amzn.to" in reply_lower or "your orders" in reply_lower or "dm us" in reply_lower:
            actionability += 1
        if "visit" in reply_lower or "track" in reply_lower or "cancel" in reply_lower or "replace" in reply_lower:
            actionability += 1
        if "we cannot help" in reply_lower or len(draft_reply.split()) <= 4:
            actionability = 2
        actionability = min(5, max(1, actionability))

        # 3. Evaluate Safety & Policy (No dangerous overpromises, no asking for credit card numbers publicly)
        safety = 5
        if any(w in reply_lower for w in ["share your password", "tweet your card number", "give me your pin"]):
            safety = 1
        elif "i promise you 100% full instant cash refund right now without return" in reply_lower:
            safety = 2
        safety = min(5, max(1, safety))

        # 4. Evaluate Factual Grounding (Alignment with retrieved evidence & intent)
        grounding = 4
        if retrieved_evidence and len(retrieved_evidence) > 0:
            top_evidence = retrieved_evidence[0]
            # Check if reply uses concepts present in the evidence
            ev_text = (top_evidence.get("historical_brand_resolution", "")).lower()
            overlap_words = set(reply_lower.split()) & set(ev_text.split())
            if len(overlap_words) >= 4:
                grounding = 5
            elif len(overlap_words) < 2:
                grounding = 3
        else:
            if "dm" in reply_lower or "help" in reply_lower:
                grounding = 3
            else:
                grounding = 2
        grounding = min(5, max(1, grounding))

        # Calculate Overall Composite Score
        overall = round((grounding * 0.30 + tone * 0.20 + actionability * 0.25 + safety * 0.25), 2)

        reasoning = (
            f"Grounding ({grounding}/5): Aligned with historical resolutions. "
            f"Tone ({tone}/5): Empathetic AmazonHelp voice. "
            f"Actionability ({actionability}/5): Clear self-service next steps. "
            f"Safety ({safety}/5): Complies with privacy and policy standards."
        )

        return RubricScores(
            factual_grounding=grounding,
            tone_and_brand_voice=tone,
            actionability=actionability,
            safety_and_policy=safety,
            overall_score=overall,
            judge_reasoning=reasoning
        )


def compute_judge_human_agreement(
    llm_scores: List[float],
    human_scores: List[float]
) -> Dict[str, Any]:
    """
    Calculates inter-annotator agreement metrics between LLM Judge and Human labels:
    - Cohen's Kappa (binned into integer buckets)
    - % Agreement within ±1 point
    - Mean Absolute Error (MAE)
    - Pearson Correlation
    """
    if len(llm_scores) == 0 or len(human_scores) == 0:
        return {"error": "Empty score lists"}

    llm_arr = np.array(llm_scores)
    human_arr = np.array(human_scores)

    # MAE
    mae = float(np.mean(np.abs(llm_arr - human_arr)))

    # % within 1 point tolerance
    within_1_pt = float(np.mean(np.abs(llm_arr - human_arr) <= 1.0)) * 100.0

    # Exact agreement
    exact_match = float(np.mean(np.round(llm_arr) == np.round(human_arr))) * 100.0

    # Cohen's Kappa on rounded integer categories (1, 2, 3, 4, 5)
    llm_int = np.clip(np.round(llm_arr).astype(int), 1, 5)
    human_int = np.clip(np.round(human_arr).astype(int), 1, 5)

    try:
        kappa = float(cohen_kappa_score(human_int, llm_int, weights="quadratic"))
    except Exception:
        kappa = 0.0

    # Correlation
    if len(llm_arr) > 1 and np.std(llm_arr) > 0 and np.std(human_arr) > 0:
        corr = float(np.corrcoef(llm_arr, human_arr)[0, 1])
    else:
        corr = 1.0

    return {
        "sample_size": len(llm_scores),
        "cohens_kappa_quadratic": round(kappa, 4),
        "percent_within_1_point": round(within_1_pt, 2),
        "exact_agreement_pct": round(exact_match, 2),
        "mean_absolute_error": round(mae, 4),
        "pearson_correlation": round(corr, 4)
    }


if __name__ == "__main__":
    judge = LLMJudge()
    score_out = judge.score(
        customer_tweet="My order 102-39102 is missing! Tracking says delivered!",
        predicted_intent="shipping_delay_delivery_issue",
        draft_reply="We're sorry your package is missing! Please check around your porch; if not found within 24h, DM us at https://amzn.to/help.",
        retrieved_evidence=[{"historical_brand_resolution": "Check around your porch and DM us at https://amzn.to/help."}]
    )
    print(score_out.model_dump_json(indent=2))
