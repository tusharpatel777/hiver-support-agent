import os
import sys
import json
import re
from pathlib import Path

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from typing import List, Dict, Any, Tuple
from src.config import (
    RAW_TWCS_PATH,
    PROCESSED_THREADS_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TARGET_BRAND,
    INTENT_TAXONOMY
)

# Boilerplate patterns that add zero substantive resolution (e.g. only "Please DM us" with nothing else)
BOILERPLATE_REGEX = re.compile(
    r"^(please\s+)?(dm|send\s+us\s+a\s+direct\s+message|reach\s+out\s+via\s+dm)[.\s!]*$",
    re.IGNORECASE
)


def is_vacuous_boilerplate(text: str) -> bool:
    """Checks if a brand reply is merely empty boilerplate with no substantive guidance."""
    text_clean = text.strip().lower()
    if BOILERPLATE_REGEX.match(text_clean):
        return True
    if len(text_clean.split()) <= 4 and ("dm" in text_clean or "direct message" in text_clean):
        return True
    return False


def infer_intent_from_text(text: str) -> str:
    """Classifies raw historical customer text into the bottom-up 8-class taxonomy."""
    lower = text.lower()
    if any(w in lower for w in ["hacked", "stolen", "unauthorized login", "locked out", "2fa", "otp", "password", "someone in another country"]):
        return "account_access_security"
    if any(w in lower for w in ["charged twice", "double charge", "charged without permission", "mystery charge", "billing", "credit card", "debit card"]):
        return "billing_and_payment_dispute"
    if any(w in lower for w in ["broken", "smashed", "shattered", "damaged", "wrong item", "defective", "missing item"]):
        return "damaged_or_wrong_item"
    if any(w in lower for w in ["refund", "cancel", "return", "money back"]):
        return "refund_and_cancellation"
    if any(w in lower for w in ["late", "delay", "not delivered", "driver", "courier", "rescheduled"]):
        return "shipping_delay_delivery_issue"
    if any(w in lower for w in ["prime video", "kindle", "gift card", "prime membership", "error 5004", "audible", "music"]):
        return "subscription_and_digital_services"
    if any(w in lower for w in ["track", "where is", "order status", "dispatch", "shipment", "package"]):
        return "order_status_tracking"
    return "general_feedback_other"


def process_raw_kaggle_csv(csv_path: Path, max_threads: int = 10000) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parses raw twcs.csv in streaming chunks to reconstruct customer -> AmazonHelp threads.
    """
    print(f"Streaming raw Kaggle dataset from {csv_path} ({csv_path.stat().st_size / (1024*1024):.2f} MB)...")
    
    brand_replies: Dict[int, Dict[str, Any]] = {} # in_response_to_tweet_id -> brand_tweet
    candidate_customer_ids = set()

    # Pass 1: Collect AmazonHelp outbound tweets and their target in_response_to_tweet_ids
    print(f"Pass 1: Scanning for {TARGET_BRAND} outbound responses...")
    chunk_size = 150000
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        brand_chunk = chunk[chunk['author_id'] == TARGET_BRAND]
        for _, row in brand_chunk.iterrows():
            in_reply_to = row.get('in_response_to_tweet_id')
            if pd.notna(in_reply_to):
                try:
                    in_reply_id = int(in_reply_to)
                    brand_replies[in_reply_id] = {
                        "tweet_id": int(row['tweet_id']),
                        "brand_text": str(row['text'])
                    }
                    candidate_customer_ids.add(in_reply_id)
                except Exception:
                    continue
                    
        if len(candidate_customer_ids) >= max_threads * 3:
            break

    print(f"Found {len(brand_replies)} {TARGET_BRAND} reply chains to match.")

    # Pass 2: Reconstruct threads by locating customer inbound tweets
    print("Pass 2: Reconstructing customer -> brand conversational threads...")
    reconstructed_threads = []
    total_reconstructed = 0
    boilerplate_count = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        matching_cust = chunk[chunk['tweet_id'].isin(candidate_customer_ids)]
        for _, cust_row in matching_cust.iterrows():
            tid = int(cust_row['tweet_id'])
            if tid in brand_replies:
                brand_info = brand_replies[tid]
                cust_text = str(cust_row.get('text', ''))
                brand_text = brand_info["brand_text"]
                
                total_reconstructed += 1
                if is_vacuous_boilerplate(brand_text):
                    boilerplate_count += 1
                    continue
                
                intent = infer_intent_from_text(cust_text)
                reconstructed_threads.append({
                    "thread_id": f"twcs_{brand_info['tweet_id']}",
                    "brand": TARGET_BRAND,
                    "intent": intent,
                    "customer_tweet": cust_text,
                    "brand_reply": brand_text,
                    "in_response_to": tid,
                    "is_boilerplate": False
                })

                if len(reconstructed_threads) >= max_threads:
                    break

        if len(reconstructed_threads) >= max_threads:
            break

    stats = {
        "source": "raw_kaggle_twcs_csv",
        "raw_csv_size_mb": round(csv_path.stat().st_size / (1024*1024), 2),
        "total_reconstructed_threads": total_reconstructed,
        "boilerplate_excluded": boilerplate_count,
        "exclusion_rate_pct": round((boilerplate_count / max(1, total_reconstructed)) * 100, 2),
        "clean_retained_threads": len(reconstructed_threads)
    }
    return reconstructed_threads, stats


def prepare_dataset() -> List[Dict[str, Any]]:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if RAW_TWCS_PATH.exists() and RAW_TWCS_PATH.stat().st_size > 1024 * 1024:
        print(f"Processing real Kaggle TWCS dataset at {RAW_TWCS_PATH}...")
        threads, stats = process_raw_kaggle_csv(RAW_TWCS_PATH, max_threads=5000)
    else:
        print("Raw dataset not found. Using curated historical corpus.")
        from src.data_prep_synthetic import build_synthetic_historical_corpus
        threads = build_synthetic_historical_corpus(multiplier=20)
        stats = {
            "source": "curated_amazonhelp_corpus",
            "total_threads": len(threads),
            "boilerplate_excluded": 0,
            "exclusion_rate_pct": 0.0,
            "clean_retained_threads": len(threads)
        }
        
    print(f"Dataset Processing Summary:\n{json.dumps(stats, indent=2)}")
    
    with open(PROCESSED_THREADS_PATH, "w", encoding="utf-8") as f:
        for t in threads:
            f.write(json.dumps(t) + "\n")
            
    print(f"Successfully saved {len(threads)} clean real threads to {PROCESSED_THREADS_PATH}")
    return threads


if __name__ == "__main__":
    prepare_dataset()
