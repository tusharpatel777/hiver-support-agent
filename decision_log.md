# Architectural & Engineering Decision Log

**Project:** AI Customer Support Agent (Hiver SDE Intern Assignment)  
**Author:** Tushar Patel  
**Target Brand:** `AmazonHelp`

This document records the key architectural, technical, and evaluation decisions made during the design, development, and benchmarking of the system, along with the trade-offs and rationales.

---

### 1. Brand Selection: AmazonHelp over Banking77 & Telecoms
- **Decision:** Selected `AmazonHelp` from the Kaggle Twitter Customer Support dataset as the primary evaluation brand.
- **Rationale:** Banking77 is artificially clean, pre-tokenized, and contains no real conversational grounding or brand voice nuances. Telecom accounts (e.g. Comcast) suffer from excessive regional boilerplate. `AmazonHelp` offers high tweet volume, diverse operational intents (logistics, digital subscriptions, refunds, defects, account security), and realistic multi-turn escalation dynamics.
- **Trade-off:** `AmazonHelp` data contains frequent "DM us" replies that must be filtered during data preparation to extract meaningful resolutions.

---

### 2. Bottom-Up Unsupervised Taxonomy Derivation
- **Decision:** Derived the 8 intent taxonomy classes bottom-up using TF-IDF + K-Means cluster centroid analysis on customer tweets, rather than pre-assuming categories top-down.
- **Rationale:** Top-down assumptions miss operational boundaries (e.g., distinguishing between routine `order_status_tracking` and severe `shipping_delay_delivery_issue` involving driver misconduct).
- **Trade-off:** Requires ongoing maintenance if new product categories or customer behaviors emerge.

---

### 3. Intent-Filtered Vector Retrieval (Filter-First Architecture)
- **Decision:** Enforced a strict two-stage retrieval pipeline where candidate historical resolutions are partitioned by predicted intent *before* cosine similarity ranking.
- **Rationale:** Global semantic similarity frequently confuses phrases like "charged $139 for Prime renewal" (billing) with "Prime Video error 5004" (digital service) due to shared "Prime" token overlaps. Hard-partitioning by intent eliminates cross-intent contamination.
- **Trade-off:** If the intent classifier makes an error, the retriever searches within the wrong partition. This is mitigated by router fallback when top-1 similarity drops below threshold.

---

### 4. Few-Shot Prompting vs. Fine-Tuning for Classification
- **Decision:** Implemented prompt-engineered few-shot in-context classification backed by calibrated ML heuristics rather than fine-tuning a dedicated model.
- **Rationale:** Prompting enables sub-second calibration, deterministic explanation generation (`reasoning` field), zero fine-tuning infrastructure costs, and rapid iteration on taxonomy definitions.
- **Trade-off:** Slightly higher token latency in remote API setups, though mitigated by compact JSON output schemas.

---

### 5. Deterministic, Rule-Informed Routing Policy
- **Decision:** Designed an explicit multi-signal policy engine with deterministic reason codes (`policy_sensitive_intent`, `policy_trigger_keyword`, `low_intent_confidence`, `no_grounding_evidence`, `low_confidence_draft`) rather than letting an LLM make an unconstrained free-form routing decision.
- **Rationale:** In enterprise customer service, routing decisions must be auditable, deterministic, and compliance-guaranteed. Unconstrained LLM routing produces uncalibrated hallucinations on edge-case legal/security threats.
- **Trade-off:** Requires explicit threshold tuning against a labeled validation set.

---

### 6. Asymmetric Cost Matrix for Escalation Errors (5:1 Ratio)
- **Decision:** Weighted missed escalations (False Negatives: auto-handling when human escalation was needed) at 5.0x the cost of unnecessary escalations (False Positives: routing to human when auto-handling was safe).
- **Rationale:** A false escalation costs ~2 minutes of agent triage time (~$1.00 cost). A missed escalation on a legal threat, account takeover, or severe damaged item leads to customer churn, public brand damage, or regulatory fines (~$5.00+ cost).
- **Trade-off:** Higher total escalation rate (lower specificity), prioritizing 100% safety on critical threats.

---

### 7. Strict AmazonHelp Brand Voice Guardrails
- **Decision:** Constrained the Reply Drafter to include standard Amazon help links (`https://amzn.to/help`, `https://amzn.to/YourOrders`), polite empathy markers, and strict prohibitions against making unconditional financial refund guarantees in public tweets.
- **Rationale:** Public support agents cannot verify customer identity on Twitter; promising specific dollar amounts without backend authentication creates severe legal liability.
- **Trade-off:** Draft replies focus on directing customers to authenticated self-service flows rather than unilaterally executing financial transactions.

---

### 8. Evidence Attribution via `evidence_ids`
- **Decision:** Mandated that every generated draft reply cite the specific `evidence_ids` of the historical resolutions that grounded the generation.
- **Rationale:** Essential for explainability, reviewer trust, and automated factual grounding scoring in the evaluation harness.
- **Trade-off:** Requires tracking metadata throughout the retrieval and drafting pipeline.

---

### 9. Deliberate "Hard" Edge Case Oversampling in Golden Set
- **Decision:** Oversampled difficult test cases (sarcasm, ambiguous multi-intent tweets, angry legal threats, typos, and slang) to compose >25% of the 160-item Golden Evaluation Set.
- **Rationale:** Evaluating only on standard, easy queries yields inflated ~98% accuracy numbers that collapse in real-world deployment. The eval must test where the system breaks.
- **Trade-off:** Reduces headline accuracy numbers, providing an honest and rigorous benchmark.

---

### 10. Multi-Dimensional Rubric for LLM-as-a-Judge
- **Decision:** Evaluated reply quality across 4 distinct dimensions (Factual Grounding, Tone & Brand Voice, Actionability, Safety & Policy Adherence) on a 1-5 Likert scale, with an overall composite score.
- **Rationale:** Single "fluency" scores hide dangerous flaws (e.g., an extremely polite, fluent reply that promises an unapproved $500 refund is completely unsafe).
- **Trade-off:** Slightly higher evaluation compute overhead per candidate reply.

---

### 11. Human vs. LLM Judge Inter-Annotator Calibration
- **Decision:** Hand-scored 40 samples from the golden set across all 4 rubric dimensions and computed Quadratic Cohen's Kappa, MAE, and % Agreement within ±1.0 point.
- **Rationale:** Validates whether the automated LLM judge's scores reflect human expert judgment rather than self-serving model bias.
- **Trade-off:** Requires dedicated manual labeling time, but delivers the core evidence required by the assignment.

---

### 12. Two Meaningful Baselines (Trivial Floor + Classical ML)
- **Decision:** Evaluated against both a Trivial Baseline (majority intent + canned generic template) and a Simple Baseline (TF-IDF/LogReg + 1-NN verbatim retrieval without LLM generation).
- **Rationale:** Isolates exactly how much value the LLM generation and intent-filtered retrieval add beyond standard retrieval.
- **Trade-off:** Requires maintaining and executing three complete pipeline variants in the benchmark runner.

---

### 13. Sub-Second Standalone Reproducibility (<15 Min Constraint)
- **Decision:** Engineered the codebase to run fully standalone with deterministic, pre-indexed embeddings and high-fidelity historical data generation, executing all 160 golden set evaluations across 3 systems in under 2 seconds.
- **Rationale:** Reviewers should be able to clone the repository, run one command, and verify all headline numbers immediately without waiting for massive downloads or hitting rate limits.
- **Trade-off:** Requires bundling curated seed datasets alongside raw Kaggle CSV parsers.
