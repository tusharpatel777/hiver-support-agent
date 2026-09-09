import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field

from src.config import (
    PROCESSED_THREADS_PATH,
    DEFAULT_RETRIEVAL_SIMILARITY_THRESHOLD,
    config
)


class RetrievalExemplar(BaseModel):
    evidence_id: str
    intent: str
    historical_customer_issue: str
    historical_brand_resolution: str
    similarity_score: float = Field(ge=0.0, le=1.0)


class GroundingRetriever:
    """
    Intent-filtered vector retriever for historical brand customer support resolutions.
    """
    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = index_path or PROCESSED_THREADS_PATH
        self.corpus: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )
        self.intent_indices: Dict[str, List[int]] = {}
        self.tfidf_matrix = None
        self._build_index()

    def _build_index(self):
        """Loads processed threads and builds per-intent index partitions."""
        if not self.index_path.exists():
            return

        with open(self.index_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                intent = data.get("intent", "general_feedback_other")
                item = {
                    "evidence_id": data.get("thread_id", f"ev_{idx}"),
                    "intent": intent,
                    "customer_tweet": data.get("customer_tweet", ""),
                    "brand_reply": data.get("brand_reply", "")
                }
                self.corpus.append(item)
                
                if intent not in self.intent_indices:
                    self.intent_indices[intent] = []
                self.intent_indices[intent].append(len(self.corpus) - 1)

        if len(self.corpus) > 0:
            corpus_texts = [item["customer_tweet"] for item in self.corpus]
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(
        self,
        query: str,
        predicted_intent: str,
        top_k: int = 3,
        filter_by_intent: bool = True
    ) -> List[RetrievalExemplar]:
        """
        Retrieves top-k historical resolutions.
        If filter_by_intent is True, restricts search candidates to the specified intent class.
        """
        if not self.corpus or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        
        # Determine candidate index pool
        if filter_by_intent and predicted_intent in self.intent_indices and len(self.intent_indices[predicted_intent]) > 0:
            candidate_indices = self.intent_indices[predicted_intent]
        else:
            # Fallback to entire corpus if intent partition is empty or unconstrained
            candidate_indices = list(range(len(self.corpus)))

        sub_matrix = self.tfidf_matrix[candidate_indices]
        similarities = cosine_similarity(query_vec, sub_matrix)[0]

        # Rank candidates
        ranked_order = np.argsort(similarities)[::-1][:top_k]
        
        results: List[RetrievalExemplar] = []
        for rank_pos in ranked_order:
            orig_idx = candidate_indices[rank_pos]
            item = self.corpus[orig_idx]
            sim_score = float(similarities[rank_pos])
            
            # Normalize slight floating precision
            sim_score = max(0.0, min(1.0, sim_score))
            
            results.append(
                RetrievalExemplar(
                    evidence_id=item["evidence_id"],
                    intent=item["intent"],
                    historical_customer_issue=item["customer_tweet"],
                    historical_brand_resolution=item["brand_reply"],
                    similarity_score=round(sim_score, 4)
                )
            )
            
        return results

    def get_max_similarity(self, exemplars: List[RetrievalExemplar]) -> float:
        """Returns the highest similarity score in the retrieved set."""
        if not exemplars:
            return 0.0
        return max(e.similarity_score for e in exemplars)


if __name__ == "__main__":
    retriever = GroundingRetriever()
    query = "Where is my package? Tracking number 112-9847120 says delayed."
    intent = "order_status_tracking"
    
    exemplars = retriever.retrieve(query, predicted_intent=intent, top_k=3)
    print(f"Query: {query}")
    print(f"Predicted Intent: {intent}")
    print(f"Top-{len(exemplars)} Grounding Evidence:")
    for ex in exemplars:
        print(f"  [{ex.evidence_id}] (Sim: {ex.similarity_score:.4f})")
        print(f"    Cust: {ex.historical_customer_issue}")
        print(f"    Brand: {ex.historical_brand_resolution}\n")
