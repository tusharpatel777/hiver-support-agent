import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pydantic import BaseModel, Field
from src.retrieval import RetrievalExemplar


class DraftOutput(BaseModel):
    draft_reply: str
    evidence_ids: List[str]
    draft_confidence: float = Field(ge=0.0, le=1.0)
    contains_hedge: bool = False
    grounding_quality: str = "high"


from src.llm_client import llm_client

class ReplyDrafter:
    """
    Generates grounded, brand-compliant support replies.
    Supports OpenAI, Gemini, Claude LLM API or grounded synthesis fallback.
    """
    def __init__(self, brand_name: str = "AmazonHelp"):
        self.brand_name = brand_name
        self.llm = llm_client

    def _build_prompt_context(
        self,
        customer_tweet: str,
        predicted_intent: str,
        exemplars: List[RetrievalExemplar]
    ) -> str:
        lines = [
            f"You are drafting an official Twitter reply for @{self.brand_name}.",
            f"Customer Intent: {predicted_intent}",
            f"Incoming Customer Tweet: \"{customer_tweet}\"",
            "",
            "Grounding Historical Cases (from actual brand resolutions in the dataset):"
        ]
        for ex in exemplars:
            lines.append(f"- [ID: {ex.evidence_id}] (Similarity: {ex.similarity_score:.2f})")
            lines.append(f"  Customer asked: \"{ex.historical_customer_issue}\"")
            lines.append(f"  Brand answered: \"{ex.historical_brand_resolution}\"")

        lines.extend([
            "",
            "Brand Voice Guidelines:",
            "1. Be concise (under 280 characters if possible), polite, and empathetic.",
            "2. Ground your advice directly in the historical brand resolutions provided above.",
            "3. If order details/account info are needed, instruct them to connect via DM link: https://amzn.to/help.",
            "4. NEVER invent policy or guarantee unverified financial refunds without directing to official forms.",
            "5. Cite the evidence_ids used."
        ])
        return "\n".join(lines)

    def draft(
        self,
        customer_tweet: str,
        predicted_intent: str,
        exemplars: List[RetrievalExemplar],
        use_api: bool = True
    ) -> DraftOutput:
        """
        Drafts a grounded reply. If use_api is True and LLM API is available, calls LLM; else synthesizes grounded template.
        """
        if not exemplars:
            return DraftOutput(
                draft_reply="We apologize for the inconvenience! Please send us a direct message with your order details at https://amzn.to/help so our team can look into this.",
                evidence_ids=[],
                draft_confidence=0.40,
                contains_hedge=True,
                grounding_quality="none"
            )

        top_ex = exemplars[0]
        evidence_ids = [top_ex.evidence_id]

        # 1. Try real LLM API generation if configured
        if use_api and self.llm.is_api_available():
            sys_prompt = self._build_prompt_context(customer_tweet, predicted_intent, exemplars)
            user_prompt = f"Draft the response for customer tweet: \"{customer_tweet}\""
            api_reply = self.llm.generate_text(sys_prompt, user_prompt)
            if api_reply and len(api_reply) > 15:
                return DraftOutput(
                    draft_reply=api_reply,
                    evidence_ids=evidence_ids,
                    draft_confidence=0.92,
                    contains_hedge=False,
                    grounding_quality="high"
                )

        # 2. Local Grounded Synthesis Fallback
        urls = re.findall(r'https?://\S+', top_ex.historical_brand_resolution)
        primary_link = urls[0] if urls else "https://amzn.to/help"

        cust_lower = customer_tweet.lower()

        # Tailor reply grounded in the specific intent and retrieved exemplar
        if predicted_intent == "order_status_tracking":
            if "track" in cust_lower or "where" in cust_lower:
                reply = f"We'd be glad to help check your shipment! You can track real-time dispatch and courier updates under 'Your Orders' at {primary_link}. If tracking hasn't updated past the delivery window, please DM us your order ID."
            else:
                reply = f"We apologize for the tracking confusion! Please visit 'Your Orders' at {primary_link} for the latest carrier status, or reach out via DM with your order number so we can investigate."
            confidence = 0.90
            contains_hedge = False

        elif predicted_intent == "shipping_delay_delivery_issue":
            if "delivered" in cust_lower and ("not" in cust_lower or "missing" in cust_lower):
                reply = f"We're sorry to hear your package is missing! Sometimes carriers prematurely mark items as delivered. Please check around your property and with neighbors; if it doesn't arrive within 24 hours, connect via DM at {primary_link} to start a replacement claim."
            else:
                reply = f"We sincerely apologize for the shipping delay. Please DM us your order details via {primary_link} so we can check on the courier transit delay and help make this right."
            confidence = 0.88
            contains_hedge = False

        elif predicted_intent == "refund_and_cancellation":
            if "cancel" in cust_lower:
                reply = f"You can cancel items anytime before they enter the dispatch process from 'Your Orders' at {primary_link}. If the order has already shipped, you can refuse the parcel or initiate a free return."
            else:
                reply = f"Once our fulfillment center receives and inspects a return, refunds typically process within 3-5 business days. You can track your return and refund status under 'Your Orders' at {primary_link}."
            confidence = 0.91
            contains_hedge = False

        elif predicted_intent == "damaged_or_wrong_item":
            if "broken" in cust_lower or "damaged" in cust_lower or "smashed" in cust_lower:
                reply = f"We're so sorry your item arrived damaged! You can arrange an immediate free replacement or return through 'Your Orders' > 'Return or Replace Items' at {primary_link}. DM us if you need extra support."
            else:
                reply = f"We apologize for the wrong item! Please head to 'Your Orders' > 'Wrong item received' at {primary_link} to generate a prepaid return label and have the correct item dispatched."
            confidence = 0.89
            contains_hedge = False

        elif predicted_intent == "account_access_security":
            reply = f"For your security, please never share sensitive account details publicly on Twitter. Visit our secure Account Recovery page at {primary_link} immediately or contact our 24/7 security team."
            confidence = 0.95
            contains_hedge = False

        elif predicted_intent == "billing_and_payment_dispute":
            reply = f"We apologize for the billing concern! Please review your recent transactions and active subscriptions at {primary_link}. For specific charge investigations, send us a private DM with the charge date and amount."
            confidence = 0.87
            contains_hedge = False

        elif predicted_intent == "subscription_and_digital_services":
            reply = f"You can manage your active digital memberships and subscriptions directly under Account Settings at {primary_link}. If you're experiencing app errors, try restarting the app or re-logging into your account."
            confidence = 0.88
            contains_hedge = False

        else:
            # Fallback grounded synthesis
            reply = f"Thank you for reaching out to @{self.brand_name}! For assistance with this request, please visit our Help Center at {primary_link} or DM us with additional details."
            confidence = 0.70
            contains_hedge = True

        grounding_quality = "high" if top_ex.similarity_score > 0.4 else "moderate"

        return DraftOutput(
            draft_reply=reply,
            evidence_ids=evidence_ids,
            draft_confidence=confidence,
            contains_hedge=contains_hedge,
            grounding_quality=grounding_quality
        )


if __name__ == "__main__":
    drafter = ReplyDrafter()
    sample_ex = RetrievalExemplar(
        evidence_id="amzn_1001",
        intent="order_status_tracking",
        historical_customer_issue="Where is my order?",
        historical_brand_resolution="Track your order at https://amzn.to/YourOrders or DM us.",
        similarity_score=0.85
    )
    res = drafter.draft("My package hasn't arrived yet! Order 102-8472910", "order_status_tracking", [sample_ex])
    print(f"Draft: {res.draft_reply}")
    print(f"Evidence: {res.evidence_ids}, Confidence: {res.draft_confidence}, Hedge: {res.contains_hedge}")
