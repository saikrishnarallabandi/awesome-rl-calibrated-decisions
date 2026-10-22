# Awesome RL for Calibrated Decisions

A curated, annotated reading list on training AI systems whose probabilities can be acted on: reinforcement learning with calibration-aware rewards, abstention, uncertainty-aware reward models, and decision-making under uncertainty.

Maintained by **Sai Krishna Rallabandi**. Independent project, not affiliated with TypeSafe AI or any other organization mentioned here.

## Motivation

Most post-training methods optimize for what people prefer (RLHF) or what can be checked (RLVR). Both reward a confident answer over an honest "70% sure", and a large body of evidence shows they leave models overconfident. If software is going to act on a model's output, by routing, approving, escalating, or abstaining, the probability attached to that output matters as much as the answer. This list collects the research on making those probabilities trustworthy: proper-scoring-rule rewards, calibrated abstention, and models built to return decisions with probabilities, such as TypeSafe AI's Jev, trained with what its makers call Reinforcement Learning for Calibrated Decisions (RLCD).

## Primer

- **Calibration.** A model is calibrated if, among all the answers it gives with 70% confidence, about 70% are correct. Common metrics: Expected Calibration Error (ECE), Brier score, negative log-likelihood, and risk-coverage curves for selective prediction.
- **Proper scoring rules.** A scoring rule is *strictly proper* if the only way to maximize your expected score is to report your true belief (Gneiting & Raftery, 2007). The Brier score and log score are the standard examples. Using one as an RL reward on a model's stated confidence removes the incentive to bluff, which is the idea behind RLCR and Rewarding Doubt.
- **Abstention.** Answering only when confidence clears a threshold. With a reward of +1 for correct, 0 for abstaining, and −λ for wrong, the optimal threshold follows from the cost of an error (λ).
- **RLCD and Jev.** TypeSafe AI describes RLCD as training for "calibrated decisions: answers with epistemically honest probabilities". Jev is its first model: it takes state plus typed questions and returns a choice, score, or probability instead of text. See [Part 10](#part-10) for claims versus independent evidence.

## Start here

1. [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (ICML 2017). The problem and the simplest fix.
2. [Just Ask for Calibration](https://arxiv.org/abs/2305.14975) (EMNLP 2023). How RLHF models' confidence behaves and how to elicit it.
3. [Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty](https://arxiv.org/abs/2507.16806) (ICLR 2026). Brier-score rewards for reasoning models.
4. [Honesty over Accuracy: Trustworthy Language Models through Reinforced Hesitation](https://arxiv.org/abs/2511.11500) (preprint). Abstention as a ternary reward and a cost-dependent frontier.
5. [Evaluating and Benchmarking the System One Model Jev](https://arxiv.org/abs/2609.37647) (preprint). The first independent benchmark of an RLCD model.

## Contents

1. [Foundations: calibration & verbal confidence](#part-1)
2. [How alignment/RLHF affects calibration](#part-2)
3. [RL with calibration-aware (scoring-rule) rewards](#part-3)
4. [Calibration in RLVR / reasoning models](#part-4)
5. [Abstention & hallucination-aware RL](#part-5)
6. [Uncertainty-aware reward models](#part-6)
7. [Decisions & agents under uncertainty](#part-7)
8. [Theory / position](#part-8)
9. [Surveys](#part-9)
10. [Jev / System One models](#part-10)

Entries are newest first within each part. Venue is shown when verified against proceedings, ACL Anthology, PMLR, or publisher pages; otherwise entries are marked *preprint* or *under review / unconfirmed*. Full metadata: [`papers.csv`](papers.csv), [`papers.bib`](papers.bib). Synthesis and open problems: [`lit_review.md`](lit_review.md). Training and evaluation data: [`datasets.md`](datasets.md).

<a id="part-1"></a>

## 1. Foundations: calibration & verbal confidence

What calibration is, how it is measured, and early evidence that models can state confidence.

- [On Verbalized Confidence Scores for LLMs](https://arxiv.org/abs/2412.14737) (2026 · *under review / unconfirmed*). Verbalized scores reliability depends heavily on prompting and model.
- [Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation](https://arxiv.org/abs/2302.09664) (ICLR 2023 (notable top 25%)). Clustering by meaning gives better uncertainty (AUROC) for free-form QA.
- [Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs](https://arxiv.org/abs/2306.13063) (ICLR 2024). LLMs are overconfident verbally; sampling-consistency aggregation helps.
- [Teaching Models to Express Their Uncertainty in Words](https://arxiv.org/abs/2205.14334) (TMLR 2022 (OpenReview forum)). GPT-3 can learn to state calibrated confidence in words, moderately robust to shift.
- [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221) (2022 · *preprint*). Larger LMs are well-calibrated on MC/TF in the right format; P(True) self-evaluation scales and improves with multiple samples; P(IK) partially generalizes; RLHF policies look miscalibrated but a temperature adjustment largely fixes it.
- [Reducing conversational agents’ overconfidencethrough linguistic calibration](https://arxiv.org/abs/2012.14983) (TACL 2022 · *venue unverified*). Dialogue models' verbal doubt is poorly calibrated; a probe + control tokens improve it.
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (ICML 2017 (PMLR v70)). Modern nets' miscalibration worsens as accuracy improves; depth, width, weight decay, BatchNorm matter; temperature scaling is simplest and often most effective.
- [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500) (NeurIPS 2017). Reject option lets user pick target risk with high-probability guarantee.

<a id="part-2"></a>

## 2. How alignment/RLHF affects calibration

Post-training (RLHF, preference RL) tends to inflate confidence.

- [Calibration Collapse Under Sycophancy Fine-Tuning: How Reward Hacking Breaks Uncertainty Quantification](https://arxiv.org/abs/2604.10585) (2026 · *under review / unconfirmed*). Sycophantic reward optimization degrades calibration (reward hacking breaks UQ).
- [Does Alignment Tuning Really Break LLMs' Internal Confidence?](https://arxiv.org/abs/2409.00352) (2024 · *preprint*). Alignment–calibration relation is metric-/method-dependent, not uniformly harmful.
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) (2023 · *preprint*). Pre-trained GPT-4 is well calibrated on MMLU; post-training RLHF hurts calibration.
- [Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback](https://arxiv.org/abs/2305.14975) (EMNLP 2023 (main)). For RLHF-LMs (ChatGPT, GPT-4, Claude), verbalized confidences are typically better calibrated than token probabilities on TriviaQA/SciQ/TruthfulQA, often ~50% relative ECE reduction.
- [Investigating Uncertainty Calibration of Aligned Language Models under the Multiple-Choice Setting](https://arxiv.org/abs/2310.11732) (2023 · *preprint*). Alignment conflates answer and format uncertainty, causing overconfidence; proposes calibration-aware tuning.

<a id="part-3"></a>

## 3. RL with calibration-aware (scoring-rule) rewards

Reward = correctness + a proper scoring rule on stated confidence.

- [Beyond Binary Rewards: Training LMs to Reason about Their Uncertainty](https://arxiv.org/abs/2507.16806) (ICLR 2026). Proves correctness+bounded proper scoring rule (Brier) reward yields accurate AND calibrated models; empirically RLVR worsens OOD calibration while RLCR improves it with no accuracy loss, beating post-hoc confidence classifiers; verbal confidence improves confidence-weighted best-of-N.
- [On the effectiveness of reward functions in reinforcement learning for confidence calibration of large language models](https://arxiv.org/abs/2607.04332) (2026 · *preprint*). Identifies 'confidence reward hacking' (answer wrong to be confidently wrong); defines non-hackable reward schemes and an over-/under-confidence spectrum; best scheme is dataset-dependent → treat as hyperparameter.
- [Verifiable Rewards for Calibrated Probabilistic Forecasting](https://arxiv.org/abs/2607.00164) (2026 · *preprint*). Shows RLVR with proper scoring still degrades calibration in practice for forecasting; proposes remedies for aleatoric targets.
- [Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models](https://arxiv.org/abs/2503.02623) (ICLR 2026). Log-scoring-rule reward penalizes both over- and under-confidence; optimal policy is perfectly calibrated; trained models generalize calibration to unseen tasks without further fine-tuning.
- [Reinforcement Learning for Better Verbalized Confidence in Long-Form Generation](https://arxiv.org/abs/2505.23912) (ACL 2026 (long) [as LoVeC]). Online/offline RL makes per-claim confidence calibrated in long-form generation.
- [Confidence as a Reward: Transforming LLMs into Reward Models](https://arxiv.org/abs/2510.13501) (2025 · *preprint*). Model confidence in final answer works as training-free reward and for preference data.
- [Beyond Correctness: Confidence-Aware Reward Modeling for Enhancing Large Language Model Reasoning](https://arxiv.org/abs/2511.07483) (EMNLP 2025 (main)). Rewards correct answers only when confidence is warranted, reducing lucky-guess reasoning in small models.
- [When to Trust LLMs: Aligning Confidence with Response Quality](https://arxiv.org/abs/2404.17287) (Findings of ACL 2024). Aligning confidence order with quality improves calibration and trust decisions.
- [Taming Overconfidence in LLMs: Reward Calibration in RLHF](https://arxiv.org/abs/2410.09724) (ICLR 2025). RLHF reward models are biased toward high verbalized confidence regardless of quality. PPO-M (calibrated RM training) and PPO-C (reward adjustment) cut ECE (e.g., −6.44 ECE pts, +2.73 acc on GSM8K for Llama3-8B) and preserve MT-Bench/Arena-Hard; extends to DPO (CDPO).

<a id="part-4"></a>

## 4. Calibration in RLVR / reasoning models

Binary-reward RL for reasoning degrades calibration; fixes from 2025–2026.

- [Know What You Know: Metacognitive Entropy Calibration for Verifiable RL Reasoning](https://arxiv.org/abs/2602.22751) (2026 · *preprint*). Aligns entropy with correctness during RLVR, improving calibration.
- [Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards](https://arxiv.org/abs/2603.09117) (ICML 2026 (poster)). Shows gradient conflict between accuracy and calibration; decoupling restores calibration.
- [Process Supervision of Confidence Margin for Calibrated LLM Reasoning](https://arxiv.org/abs/2604.23333) (2026 · *preprint*). Joint correctness/confidence reward via process supervision improves calibration and compute allocation.
- [Calibrating LLMs with Semantic-level Reward](https://arxiv.org/abs/2605.15588) (2026 · *preprint*). Semantic-level confidence reward improves calibration over token-level.
- [Post-Training Shifts Confidence: A Three-Stage Analysis of How SFT, RL, and OPD Shape Pre-, Intra-, and Post-CoT Calibration](https://arxiv.org/abs/2607.13753) (2026 · *preprint*). Post-training methods reshape pre/intra/post-reasoning confidence differently.
- [Scaling with Confidence: Calibrating Confidence of LLMs for Adaptive Test Time Scaling](https://arxiv.org/abs/2607.01612) (2026 · *under review / unconfirmed*). Calibrated confidence enables early stopping / adaptive sampling savings.
- [On the Pitfalls of Verbalized Confidence Priors for Calibrating Large Reasoning Models](https://arxiv.org/abs/2609.32470) (2026 · *preprint*). Concentrated pre-RL confidence priors suppress calibration gradients; proposes remedy.
- [Beyond Verbalized Confidence: Calibrating Reasoners with Differentiable Readouts](https://arxiv.org/abs/2609.34857) (2026 · *preprint*). Readout head avoids variance/collapse of verbal confidence in RLVR.
- [RL-ARC: Calibrating Large Reasoning Models via Reasoning-guided Uncertainty](https://arxiv.org/abs/2610.11352) (2026 · *preprint*). Targets overconfidence under distribution shift without sacrificing reasoning.
- [Calibration-Aware Policy Optimization for Reasoning LLMs](https://arxiv.org/abs/2604.12632) (ACL 2026 (long)). Proves GRPO's uncertainty-agnostic advantage degrades relative calibration (AUC); fix improves calibration without hurting accuracy.

<a id="part-5"></a>

## 5. Abstention & hallucination-aware RL

Training models to say "I don't know" via ternary or cost-aware rewards.

- [Rewarding Intellectual Humility Learning When Not To Answer In Large Language Models](https://arxiv.org/abs/2601.20126) (2026 · *preprint*). Ternary-reward RL trades small accuracy for large hallucination drop.
- [When Silence Is Golden: Can LLMs Learn to Abstain in Temporal QA and Beyond?](https://arxiv.org/abs/2602.04755) (ICLR 2026). Abstention-aware RL transfers beyond temporal QA.
- [Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL](https://arxiv.org/abs/2604.17073) (Findings of ACL 2026). Rewards abstaining AND identifying missing info on unanswerable queries.
- [I-CALM: Incentivizing Confidence-Aware Abstention for LLM Hallucination Mitigation](https://arxiv.org/abs/2604.03904) (2026 · *preprint*). Announcing answer/abstain rewards in prompt reduces hallucination without training.
- [KARL: Mitigating Hallucinations in LLMs via Knowledge-Boundary-Aware Reinforcement Learning](https://arxiv.org/abs/2604.22779) (2026 · *preprint*). Adapting abstention reward to evolving knowledge avoids over-caution.
- [Can Language Models Learn When to Abstain? Policy Learning and Calibrated Answer Selection](https://arxiv.org/abs/2610.13821) (2026 · *preprint*). Separate learned selector beats calibrated-threshold filter on utility.
- [Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning](https://arxiv.org/abs/2512.19920) (ICLR 2026 Workshop: Agentic AI in the Wild). Rewards behavior consistent with calibrated confidence to reduce hallucination.
- [Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty](https://arxiv.org/abs/2601.12471) (EACL 2026 (long)). Abstention ability, not just accuracy, should be evaluated for safety-critical choice tasks.
- [Are LLM Decisions Faithful to Verbal Confidence?](https://arxiv.org/abs/2601.07767) (2026 · *under review / unconfirmed*). Frontier models' stated confidence is dissociated from cost-aware abstention behavior.
- [Learning Conformal Abstention Policies for Adaptive Risk Management in Large Language and Vision-Language Models](https://arxiv.org/abs/2502.06884) (2025 · *preprint*). Learning conformal thresholds via RL adapts risk control and improves coverage/accuracy.
- [TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning](https://arxiv.org/abs/2509.25760) (ICML 2026 (PMLR v306)). Ternary reward (correct/abstain/hallucinate) cuts hallucination by up to 28.9% and raises truthfulness by up to 21.1% vs vanilla RL across 4 knowledge-intensive benchmarks, with and without retrieval.
- [Pay-Per-Search Models are Abstention Models](https://arxiv.org/abs/2510.01152) (2025 · *under review / unconfirmed*). Penalizing search calls teaches selective help-seeking and abstention.
- [Train for Truth, Keep the Skills: Binary Retrieval-Augmented Reward Mitigates Hallucinations](https://arxiv.org/abs/2510.17733) (ICML 2026 (PMLR v306)). Penalizing any factual error (binary) cuts hallucination while keeping skills; model learns to abstain/hedge.
- [Teaching LLMs to Abstain via Fine-Grained Semantic Confidence Reward](https://arxiv.org/abs/2510.24020) (2025 · *under review / unconfirmed*). Fine-grained semantic confidence reward teaches abstention better than coarse rewards.
- [Honesty over Accuracy: Trustworthy Language Models through Reinforced Hesitation](https://arxiv.org/abs/2511.11500) (2025 · *preprint*). Frontier models almost never abstain even with explicit penalties; ternary reward (+1/0/−λ) traces a Pareto frontier where each λ is optimal for its risk regime; trained abstention enables cascades/self-cascades.
- [Uncertainty-Based Abstention in LLMs Improves Safety and Reduces Hallucinations](https://arxiv.org/abs/2404.10960) (2024 · *under review / unconfirmed*). Abstaining on uncertainty improves correctness, reduces hallucination and unsafe outputs.
- [Mitigating LLM Hallucinations via Conformal Abstention](https://arxiv.org/abs/2405.01563) (2024 · *preprint*). Gives risk-controlled abstention with fewer unneeded abstentions.
- [LLMs can learn self-restraint through iterative self-reflection](https://arxiv.org/abs/2405.13022) (TMLR (OpenReview)). Teaches self-restraint/abstention without external references; fewer hallucinations.
- [Fine-Tuning Large Language Models to Appropriately Abstain with Semantic Entropy](https://arxiv.org/abs/2410.17234) (NeurIPS 2024 Workshop: Safe Generative AI). Label-free abstention tuning matches or beats accuracy-based labels.
- [Don't Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration](https://arxiv.org/abs/2402.00367) (ACL 2024 (long)). Collaboration-based abstention outperforms calibration/prompting baselines at identifying knowledge gaps.
- [Selective Generation for Controllable Language Models](https://arxiv.org/abs/2307.09254) (NeurIPS 2024 (spotlight)). Certified risk control for generative LMs via textual entailment.

<a id="part-6"></a>

## 6. Uncertainty-aware reward models

Calibrating the critic: reward models that know what they don't know.

- [Know What You Don't Know: Uncertainty Calibration of Process Reward Models](https://arxiv.org/abs/2506.09338) (NeurIPS 2025). SOTA PRMs overestimate success probability; calibrated PRMs enable instance-adaptive inference-time scaling.
- [Towards Reliable Alignment: Uncertainty-aware RLHF](https://arxiv.org/abs/2410.23726) (2024 · *preprint*). Variance-aware policy optimization gives more reliable alignment.
- [Uncertainty-aware Reward Model: Teaching Reward Models to Know What is Unknown](https://arxiv.org/abs/2410.00847) (2024 · *under review / unconfirmed*). Reward model that models aleatoric & epistemic uncertainty filters unreliable rewards.
- [Bayesian Reward Models for LLM Alignment](https://arxiv.org/abs/2402.13210) (ICLR 2024 Workshop (SeT LLM) / ICML 2024 Workshop). Bayesian RMs mitigate reward overoptimization in best-of-n.
- [Overcoming Reward Overoptimization via Adversarial Policy Optimization with Lightweight Uncertainty Estimation](https://arxiv.org/abs/2403.05171) (2024 · *under review / unconfirmed*). Uncertainty-penalized policy optimization reduces overoptimization.
- [Post-hoc Reward Calibration: A Case Study on Length Bias](https://arxiv.org/abs/2409.17407) (ICLR 2025). Estimating & removing bias from RM scores improves alignment.

<a id="part-7"></a>

## 7. Decisions & agents under uncertainty

From QA to control: bandits, tools, and agents acting on confidence.

- [Balancing Classification and Calibration Performance in Decision-Making LLMs via Calibration Aware Reinforcement Learning](https://arxiv.org/abs/2601.13284) (Findings of ACL 2026). Balances classification accuracy and ECE in decision tasks.
- [Can an Agent Learn to Predict Its Own Failure? Outcome-Grounded Confidence Calibration in LLM Agents](https://arxiv.org/abs/2610.18614) (2026 · *preprint*). Calibrates agent confidence to eventual episode success for trust/abstain/handoff.
- [Closing the Reflection Gap: A Free Calibration Bonus for Agentic RL](https://arxiv.org/abs/2606.14211) (2026 · *preprint*). Agents mis-assess own outputs after environment feedback; a free calibration bonus fixes credit-assignment mismatch.
- [Linguistic Calibration of Long-Form Generations](https://arxiv.org/abs/2404.00474) (ICML 2024 (PMLR v235)). Defines linguistic calibration via downstream user decisions; RL improves it for long-form.
- [On the Importance of Uncertainty in Decision-Making with Large Language Models](https://arxiv.org/abs/2404.02649) (TMLR (OpenReview)). Uncertainty-aware LLM agents beat greedy ones in text-context bandits.
- [Towards Uncertainty-Aware Language Agent](https://arxiv.org/abs/2401.14016) (Findings of ACL 2024). Uncertainty-gated tool calls reduce cost and improve accuracy vs ReAct.
- [DeLLMa: Decision Making Under Uncertainty with Large Language Models](https://arxiv.org/abs/2402.02392) (ICLR 2025 (spotlight)). Direct prompting fails on decision-under-uncertainty tasks as complexity grows; decision-theoretic scaffolding improves outcomes.

<a id="part-8"></a>

## 8. Theory / position

Scoring-rule theory and arguments about how evaluation drives (mis)calibration.

- [BAS: A Decision-Theoretic Approach to Evaluating Large Language Model Confidence](https://arxiv.org/abs/2604.03216) (2026 · *preprint*). Proposes a metric for how well confidence supports abstention decisions across risk preferences.
- [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) (2025 · *preprint*). Benchmarks with binary grading reward guessing; argues for confidence-target scoring.
- [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437) (Journal of the American Statistical Association 102(477):359–378). Foundational reference: a scoring rule is strictly proper iff expected score is uniquely optimized by the true belief, which is the justification for Brier/log rewards in RLCR and Rewarding Doubt.

<a id="part-9"></a>

## 9. Surveys

Broad surveys to orient a newcomer.

- [Know Your Limits: A Survey of Abstention in Large Language Models](https://aclanthology.org/2025.tacl-1.26/) (TACL 2025 (vol. 13)). Frames abstention as a meta-capability spanning query answerability, model knowledge and human values; catalogs methods and evaluation.
- [Uncertainty Quantification and Confidence Calibration in Large Language Models: A Survey](https://doi.org/10.1145/3711896.3736569) (KDD 2025). Organizes LLM UQ by uncertainty source (input, reasoning, parameter, prediction) and reviews calibration methods and applications.
- [A Survey of Confidence Estimation and Calibration in Large Language Models](https://arxiv.org/abs/2311.08298) (NAACL 2024 (long)). Organizes confidence estimation & calibration methods for LLMs.

<a id="part-10"></a>

## 10. Jev / System One models

Typed, non-generative decision models trained with RLCD.

**Vendor claims (TypeSafe AI, not independently verified):**
- Launch post: [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (Sep 15, 2026). Introduces RLCD, contrasts it with RLHF/RLVR, and claims speed and cost about two orders of magnitude better than LLMs on "System One" tasks, and that Jev "can't hallucinate" because it does not generate strings.
- No technical paper describing RLCD had been published as of Oct 21, 2026, as far as this list's searches found.

**Independent evidence and analysis:**
- [Evaluating and Benchmarking the System One Model Jev](https://arxiv.org/abs/2609.37647) (2026 · *preprint*). Zero-shot Jev (jev-1.13.0) on 37 datasets/346,009 requests for <US$10: 95–99% on IMDB/SST-2/HellaSwag/ARC, beats Qwen3.8-27B on 27/37; choice probabilities well calibrated & support selective prediction, but binary probabilities poorly placed vs a fixed 0.5 threshold (tuned thresholds lift UNFAIR-ToS F1 0.50→0.75).
- [JevAdvBench: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models](https://arxiv.org/abs/2609.31142) (2026 · *preprint*). First robustness measurement for RLCD models; typed models return well-formed answers even when manipulated.
- [Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures](https://arxiv.org/abs/2609.29429) (2026 · *preprint*). Tests whether one-call multi-question calibrated probabilities can replace generative judges for alignment screening.
- [Beyond Calibration: Do a Typed-Decision Model's Probabilities Obey the Probability Axioms?](https://arxiv.org/abs/2609.33209) (2026 · *preprint*). Calibration ≠ coherence: probabilities on logically linked questions may not fit together, which matters for systems that act on them.
- [TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models](https://arxiv.org/abs/2610.11392) (2026 · *preprint*). Evaluates wording sensitivity and cost beyond accuracy/calibration for typed decision models.
- [Calibration and Risk-Budgeted Decisions with System One Models](https://arxiv.org/abs/2610.19811) (2026 · *preprint*). Studies controlling false-omission risk with System One scores when human deferral is unavailable (moderation, safety, reranking).
- [Narrow Decision Models:A System-1 Architecture for Calibrated, Typed Decisions](https://arxiv.org/abs/2610.25553) (2026 · *preprint*). Explores the narrow end of the design space vs general-purpose Jev with the same Choice/Score/Noul interface.
- [How Jev works: calibrated decision models](https://victordibia.com/papers/jev.pdf) (2026, independent explainer by V. Dibia).
- [Jev and Typed Decision Models: An Empirical Survey](https://eurekaleo.github.io/awesome-jev-survey/paper/main.pdf) (2026, draft, not peer-reviewed).
- Press and practitioner tests (not peer-reviewed): [InfoQ](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/), [TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/), [Arize](https://arize.com/blog/typesafe-jev-llm-judge/), [TrueStandard](https://truestandard.ai/blog/jev-accuracy-tested), [tsukumo](https://tsukumo.ch/blog/jev-bounded-probability-benchmark). InfoQ cites an analysis of launch tweets reporting a median user-reported speedup of about 7× against the 193.6× headline.

## Contributing

Suggestions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and open a [paper suggestion](.github/ISSUE_TEMPLATE/paper-suggestion.yml).

## Citation

If this list is useful in your work, please cite it:

```bibtex
@misc{rallabandi2026awesomerlcd,
  author       = {Rallabandi, Sai Krishna},
  title        = {Awesome RL for Calibrated Decisions: A Curated Reading List},
  year         = {2026},
  howpublished = {\url{https://github.com/saikrishnarallabandi/awesome-rl-calibrated-decisions}},
  note         = {Version 0.1}
}
```

See also [CITATION.cff](CITATION.cff).

## License

Content (text, annotations, `papers.csv`, `papers.bib`) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Any code in this repository is licensed under the MIT License. See [LICENSE](LICENSE).
