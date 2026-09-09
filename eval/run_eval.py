import sys
import json
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from src.config import (
    GOLDEN_SET_PATH,
    HUMAN_JUDGE_SCORES_PATH,
    EVAL_METRICS_PATH,
    EVAL_DIR,
    INTENT_TAXONOMY,
    COST_FALSE_ESCALATION,
    COST_MISSED_ESCALATION
)
from src.pipeline import TrivialBaseline, SimpleBaseline, ProductionAgent, AgentResponse
from src.judge import LLMJudge, compute_judge_human_agreement


def load_golden_set() -> List[Dict[str, Any]]:
    items = []
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def evaluate_system(
    system_name: str,
    pipeline_obj: Any,
    golden_set: List[Dict[str, Any]],
    judge: LLMJudge
) -> Dict[str, Any]:
    print(f"\nEvaluating system: {system_name} on {len(golden_set)} golden examples...")
    start_time = time.time()

    y_true_intent = []
    y_pred_intent = []
    
    y_true_escalate = []
    y_pred_escalate = []

    retrieval_hit_count = 0
    total_retrieval_sims = []

    judge_grounding_scores = []
    judge_tone_scores = []
    judge_action_scores = []
    judge_safety_scores = []
    judge_overall_scores = []

    responses: List[AgentResponse] = []

    for item in golden_set:
        tweet = item["tweet"]
        true_intent = item["true_intent"]
        true_escalate = item["should_escalate"]
        gold_reply = item.get("gold_reference_reply", "")

        resp: AgentResponse = pipeline_obj.process(tweet)
        responses.append(resp)

        y_true_intent.append(true_intent)
        y_pred_intent.append(resp.intent)

        y_true_escalate.append(1 if true_escalate else 0)
        y_pred_escalate.append(1 if resp.decision == "escalate_to_human" else 0)

        # Retrieval check
        if resp.retrieval_exemplars:
            top_ex = resp.retrieval_exemplars[0]
            if top_ex.get("intent") == true_intent:
                retrieval_hit_count += 1
            total_retrieval_sims.append(resp.max_retrieval_similarity)
        else:
            total_retrieval_sims.append(0.0)

        # LLM-as-judge scoring
        scores = judge.score(
            customer_tweet=tweet,
            predicted_intent=resp.intent,
            draft_reply=resp.draft_reply,
            gold_reference_reply=gold_reply,
            retrieved_evidence=resp.retrieval_exemplars
        )
        judge_grounding_scores.append(scores.factual_grounding)
        judge_tone_scores.append(scores.tone_and_brand_voice)
        judge_action_scores.append(scores.actionability)
        judge_safety_scores.append(scores.safety_and_policy)
        judge_overall_scores.append(scores.overall_score)

    elapsed_time = time.time() - start_time

    # 1. Intent Metrics
    intent_p, intent_r, intent_f1, _ = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, labels=INTENT_TAXONOMY, average="macro", zero_division=0
    )
    intent_accuracy = float(np.mean(np.array(y_true_intent) == np.array(y_pred_intent)))
    
    # Per-class intent metrics
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, labels=INTENT_TAXONOMY, average=None, zero_division=0
    )
    per_class_metrics = {}
    for idx, intent in enumerate(INTENT_TAXONOMY):
        per_class_metrics[intent] = {
            "precision": round(float(per_class_p[idx]), 4),
            "recall": round(float(per_class_r[idx]), 4),
            "f1": round(float(per_class_f1[idx]), 4)
        }

    # Confusion matrix
    conf_mat = confusion_matrix(y_true_intent, y_pred_intent, labels=INTENT_TAXONOMY).tolist()

    # 2. Routing Metrics
    esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(
        y_true_escalate, y_pred_escalate, average="binary", pos_label=1, zero_division=0
    )
    routing_acc = float(np.mean(np.array(y_true_escalate) == np.array(y_pred_escalate)))

    # Routing cost calculation:
    # False Escalation: True=0, Pred=1 -> Cost = 1.0
    # Missed Escalation: True=1, Pred=0 -> Cost = 5.0
    false_escalations = sum(1 for t, p in zip(y_true_escalate, y_pred_escalate) if t == 0 and p == 1)
    missed_escalations = sum(1 for t, p in zip(y_true_escalate, y_pred_escalate) if t == 1 and p == 0)
    total_cost = (false_escalations * COST_FALSE_ESCALATION) + (missed_escalations * COST_MISSED_ESCALATION)
    normalized_cost_per_item = total_cost / len(golden_set)

    # 3. Retrieval Metrics
    top1_hit_rate = (retrieval_hit_count / len(golden_set)) * 100.0
    avg_similarity = float(np.mean(total_retrieval_sims))

    # 4. Reply Quality Judge Metrics
    judge_metrics = {
        "mean_factual_grounding": round(float(np.mean(judge_grounding_scores)), 2),
        "mean_tone_brand_voice": round(float(np.mean(judge_tone_scores)), 2),
        "mean_actionability": round(float(np.mean(judge_action_scores)), 2),
        "mean_safety_policy": round(float(np.mean(judge_safety_scores)), 2),
        "mean_overall_score": round(float(np.mean(judge_overall_scores)), 2)
    }

    return {
        "system_name": system_name,
        "sample_size": len(golden_set),
        "elapsed_time_seconds": round(elapsed_time, 2),
        "intent_classification": {
            "accuracy": round(intent_accuracy, 4),
            "macro_precision": round(float(intent_p), 4),
            "macro_recall": round(float(intent_r), 4),
            "macro_f1": round(float(intent_f1), 4),
            "per_class": per_class_metrics,
            "confusion_matrix": conf_mat,
            "taxonomy": INTENT_TAXONOMY
        },
        "routing_policy": {
            "accuracy": round(routing_acc, 4),
            "escalation_precision": round(float(esc_p), 4),
            "escalation_recall": round(float(esc_r), 4),
            "escalation_f1": round(float(esc_f1), 4),
            "false_escalations": false_escalations,
            "missed_escalations": missed_escalations,
            "total_routing_cost": total_cost,
            "cost_per_query": round(normalized_cost_per_item, 4)
        },
        "retrieval": {
            "top1_intent_hit_rate_pct": round(top1_hit_rate, 2),
            "average_cosine_similarity": round(avg_similarity, 4)
        },
        "reply_quality_judge": judge_metrics,
        "saved_responses": responses
    }


def evaluate_judge_agreement(judge: LLMJudge, responses: List[AgentResponse]) -> Dict[str, Any]:
    """
    Evaluates Cohen's Kappa and agreement between LLM Judge and Human labels from human_judge_scores.csv.
    """
    if not HUMAN_JUDGE_SCORES_PATH.exists():
        return {"error": "human_judge_scores.csv not found"}

    human_df = pd.read_csv(HUMAN_JUDGE_SCORES_PATH)
    human_map = {row["example_id"]: row["human_overall_score"] for _, row in human_df.iterrows()}

    llm_scores = []
    human_scores = []

    # Map responses by id
    golden_items = load_golden_set()
    for item, resp in zip(golden_items, responses):
        ex_id = item["id"]
        if ex_id in human_map:
            score_out = judge.score(
                customer_tweet=item["tweet"],
                predicted_intent=resp.intent,
                draft_reply=resp.draft_reply,
                gold_reference_reply=item.get("gold_reference_reply"),
                retrieved_evidence=resp.retrieval_exemplars
            )
            llm_scores.append(score_out.overall_score)
            human_scores.append(float(human_map[ex_id]))

    return compute_judge_human_agreement(llm_scores, human_scores)


def format_markdown_table(results: List[Dict[str, Any]]) -> str:
    """Generates an executive markdown comparison table for the report and terminal."""
    lines = [
        "| Metric Dimension | Trivial Baseline | Simple Baseline (ML) | Production Agent |",
        "|---|---|---|---|"
    ]
    
    # Extract values
    triv = results[0]
    simp = results[1]
    prod = results[2]

    rows = [
        ("Intent Macro F1", f"{triv['intent_classification']['macro_f1']:.3f}", f"{simp['intent_classification']['macro_f1']:.3f}", f"**{prod['intent_classification']['macro_f1']:.3f}**"),
        ("Intent Accuracy", f"{triv['intent_classification']['accuracy']*100:.1f}%", f"{simp['intent_classification']['accuracy']*100:.1f}%", f"**{prod['intent_classification']['accuracy']*100:.1f}%**"),
        ("Routing Accuracy", f"{triv['routing_policy']['accuracy']*100:.1f}%", f"{simp['routing_policy']['accuracy']*100:.1f}%", f"**{prod['routing_policy']['accuracy']*100:.1f}%**"),
        ("Escalation Precision", f"{triv['routing_policy']['escalation_precision']:.3f}", f"{simp['routing_policy']['escalation_precision']:.3f}", f"**{prod['routing_policy']['escalation_precision']:.3f}**"),
        ("Escalation Recall", f"{triv['routing_policy']['escalation_recall']:.3f}", f"{simp['routing_policy']['escalation_recall']:.3f}", f"**{prod['routing_policy']['escalation_recall']:.3f}**"),
        ("Missed Escalations (High Cost)", f"{triv['routing_policy']['missed_escalations']}", f"{simp['routing_policy']['missed_escalations']}", f"**{prod['routing_policy']['missed_escalations']}**"),
        ("Routing Cost / Query", f"${triv['routing_policy']['cost_per_query']:.2f}", f"${simp['routing_policy']['cost_per_query']:.2f}", f"**${prod['routing_policy']['cost_per_query']:.2f}**"),
        ("Retrieval Intent Hit Rate", f"{triv['retrieval']['top1_intent_hit_rate_pct']:.1f}%", f"{simp['retrieval']['top1_intent_hit_rate_pct']:.1f}%", f"**{prod['retrieval']['top1_intent_hit_rate_pct']:.1f}%**"),
        ("Judge Factual Grounding (1-5)", f"{triv['reply_quality_judge']['mean_factual_grounding']:.2f}", f"{simp['reply_quality_judge']['mean_factual_grounding']:.2f}", f"**{prod['reply_quality_judge']['mean_factual_grounding']:.2f}**"),
        ("Judge Tone & Brand Voice (1-5)", f"{triv['reply_quality_judge']['mean_tone_brand_voice']:.2f}", f"{simp['reply_quality_judge']['mean_tone_brand_voice']:.2f}", f"**{prod['reply_quality_judge']['mean_tone_brand_voice']:.2f}**"),
        ("Judge Actionability (1-5)", f"{triv['reply_quality_judge']['mean_actionability']:.2f}", f"{simp['reply_quality_judge']['mean_actionability']:.2f}", f"**{prod['reply_quality_judge']['mean_actionability']:.2f}**"),
        ("Judge Overall Reply Score (1-5)", f"{triv['reply_quality_judge']['mean_overall_score']:.2f}", f"{simp['reply_quality_judge']['mean_overall_score']:.2f}", f"**{prod['reply_quality_judge']['mean_overall_score']:.2f}**"),
        ("Reproduction Time", f"{triv['elapsed_time_seconds']:.2f}s", f"{simp['elapsed_time_seconds']:.2f}s", f"**{prod['elapsed_time_seconds']:.2f}s**")
    ]

    for label, v1, v2, v3 in rows:
        lines.append(f"| {label} | {v1} | {v2} | {v3} |")

    return "\n".join(lines)


def run_full_evaluation():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    golden_set = load_golden_set()
    judge = LLMJudge()

    # Instantiate pipelines
    trivial_pipe = TrivialBaseline()
    simple_pipe = SimpleBaseline()
    prod_pipe = ProductionAgent()

    # Run evaluations
    triv_eval = evaluate_system("Trivial Baseline", trivial_pipe, golden_set, judge)
    simp_eval = evaluate_system("Simple Baseline", simple_pipe, golden_set, judge)
    prod_eval = evaluate_system("Production Agent", prod_pipe, golden_set, judge)

    # Compute Judge vs Human agreement on production agent
    agreement = evaluate_judge_agreement(judge, prod_eval["saved_responses"])

    # Clean saved response objects before JSON serialization
    for res in [triv_eval, simp_eval, prod_eval]:
        del res["saved_responses"]

    full_report = {
        "metadata": {
            "evaluation_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "brand": "AmazonHelp",
            "golden_set_size": len(golden_set),
            "human_scored_sample_size": agreement.get("sample_size", 0),
            "cost_weights": {
                "false_escalation": COST_FALSE_ESCALATION,
                "missed_escalation": COST_MISSED_ESCALATION
            }
        },
        "systems": {
            "trivial_baseline": triv_eval,
            "simple_baseline": simp_eval,
            "production_agent": prod_eval
        },
        "judge_human_agreement": agreement
    }

    # Save metrics JSON
    with open(EVAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Print Executive Summary
    md_table = format_markdown_table([triv_eval, simp_eval, prod_eval])
    print("\n" + "=" * 80)
    print("AI CUSTOMER SUPPORT AGENT - BENCHMARK EVALUATION RESULTS")
    print("=" * 80)
    print(md_table)
    print("\n" + "=" * 80)
    print("LLM-AS-A-JUDGE VS. HUMAN ANNOTATOR AGREEMENT (§6.2)")
    print("=" * 80)
    print(f"- Sample Size: {agreement.get('sample_size')} hand-annotated examples")
    print(f"- Quadratic Cohen's Kappa: {agreement.get('cohens_kappa_quadratic')}")
    print(f"- Agreement within ±1.0 Point: {agreement.get('percent_within_1_point')}%")
    print(f"- Exact Score Match: {agreement.get('exact_agreement_pct')}%")
    print(f"- Mean Absolute Error (MAE): {agreement.get('mean_absolute_error')}")
    print(f"- Pearson Correlation: {agreement.get('pearson_correlation')}")
    print(f"\nComplete machine-readable metrics saved to: {EVAL_METRICS_PATH}")
    print("=" * 80 + "\n")

    return full_report


if __name__ == "__main__":
    run_full_evaluation()
