import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from src.config import (
    PROCESSED_THREADS_PATH,
    INTENT_TAXONOMY,
    INTENT_DESCRIPTIONS
)

# Curated Few-Shot Exemplars for LLM Prompting
FEW_SHOT_INTENT_EXEMPLARS: Dict[str, List[Dict[str, str]]] = {
    "order_status_tracking": [
        {"tweet": "Where is my order #112-9018274? It was dispatched 3 days ago.", "reason": "Asking for tracking/location of a placed order."},
        {"tweet": "Can someone give me an update on package 402-9831920 tracking number?", "reason": "Inquiring about shipment tracking progress."},
        {"tweet": "Why does tracking say 'Package delayed in transit' with no date?", "reason": "Asking about order transit status."}
    ],
    "shipping_delay_delivery_issue": [
        {"tweet": "Prime delivery guaranteed by 1 PM today but still not here at 8 PM!", "reason": "Late delivery complaint on Prime guaranteed shipment."},
        {"tweet": "Courier driver marked delivered but nothing is at my door or porch.", "reason": "Missing delivered package / carrier delivery failure."},
        {"tweet": "Driver threw my box over the gate in the rain and ruined the contents.", "reason": "Carrier mishandling during delivery."}
    ],
    "refund_and_cancellation": [
        {"tweet": "I returned the item last week. When will my refund be credited to my bank?", "reason": "Inquiring about return refund timeline."},
        {"tweet": "Accidentally placed duplicate order, need to cancel immediately.", "reason": "Requesting order cancellation prior to dispatch."},
        {"tweet": "Where do I generate a return shipping label for an unwanted coat?", "reason": "Initiating product return process."}
    ],
    "damaged_or_wrong_item": [
        {"tweet": "Opened the box and the ceramic mug is shattered into pieces.", "reason": "Reporting broken/damaged item upon arrival."},
        {"tweet": "I ordered size 10 shoes and you sent size 7.", "reason": "Reporting incorrect product received."},
        {"tweet": "The blender won't turn on even when plugged in. It's completely dead.", "reason": "Reporting defective/non-functional product."}
    ],
    "account_access_security": [
        {"tweet": "Someone changed the password on my Amazon account and bought items!", "reason": "Compromised account / suspected fraud."},
        {"tweet": "I am locked out because my old 2FA phone number is disconnected.", "reason": "Two-factor authentication access lock."},
        {"tweet": "Received an OTP verification code on my phone that I did not request.", "reason": "Security alert / unauthorized login attempt."}
    ],
    "billing_and_payment_dispute": [
        {"tweet": "I see a $139 charge on my credit card that I never authorized!", "reason": "Unrecognized charge / payment dispute."},
        {"tweet": "You charged my card twice for the same order #113-9028192.", "reason": "Duplicate transaction charge dispute."},
        {"tweet": "Why was my gift card balance not applied to the order total?", "reason": "Payment method dispute / billing discrepancy."}
    ],
    "subscription_and_digital_services": [
        {"tweet": "How do I cancel my Kindle Unlimited subscription before it renews?", "reason": "Digital subscription management inquiry."},
        {"tweet": "Prime Video giving error code 5004 on my smart TV.", "reason": "Digital streaming service technical issue."},
        {"tweet": "I bought a digital PlayStation gift card code but received nothing in email.", "reason": "Digital goods / gift card delivery issue."}
    ],
    "general_feedback_other": [
        {"tweet": "Agent David on the phone was incredibly patient and solved my problem. Kudos!", "reason": "General praise / customer service feedback."},
        {"tweet": "Are your warehouse staff getting proper holiday breaks this year?", "reason": "General policy inquiry."},
        {"tweet": "The new mobile app interface is super laggy on iOS 17.", "reason": "General app feedback / bug report."}
    ]
}


def run_unsupervised_clustering(n_clusters: int = 8, sample_size: int = 300) -> Dict[str, Any]:
    """
    Performs bottom-up TF-IDF + K-Means clustering on processed inbound tweets
    to demonstrate data-driven derivation of the intent taxonomy.
    """
    tweets = []
    if PROCESSED_THREADS_PATH.exists():
        with open(PROCESSED_THREADS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                tweets.append(data["customer_tweet"])
                if len(tweets) >= sample_size:
                    break
    
    if not tweets:
        return {"error": "No processed tweets found. Run data_prep.py first."}
        
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(tweets)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    
    clusters_info = {}
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :8]]
        
        # Find exemplar closest to centroid
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        exemplars = [tweets[idx] for idx in cluster_indices[:3]] if len(cluster_indices) > 0 else []
        
        clusters_info[f"cluster_{i}"] = {
            "top_terms": top_terms,
            "sample_size": int(len(cluster_indices)),
            "sample_exemplars": exemplars
        }
        
    return {
        "n_clusters": n_clusters,
        "total_tweets_clustered": len(tweets),
        "clusters": clusters_info,
        "derived_taxonomy": INTENT_TAXONOMY
    }


def get_taxonomy_prompt_context() -> str:
    """Formats the intent taxonomy and descriptions into a system prompt snippet."""
    lines = ["Available Intent Categories:"]
    for intent in INTENT_TAXONOMY:
        desc = INTENT_DESCRIPTIONS.get(intent, "")
        lines.append(f"- {intent}: {desc}")
    return "\n".join(lines)


def get_few_shot_prompt_context() -> str:
    """Formats few-shot exemplars for classifier prompting."""
    lines = ["Few-Shot Classification Examples:"]
    for intent, examples in FEW_SHOT_INTENT_EXEMPLARS.items():
        for ex in examples:
            lines.append(f'Tweet: "{ex["tweet"]}" -> Intent: {intent} (Reason: {ex["reason"]})')
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_unsupervised_clustering()
    print(json.dumps(result, indent=2))
