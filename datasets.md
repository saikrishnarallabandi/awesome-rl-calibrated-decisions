# Datasets for Training and Evaluating RLCD-Style Models

This page covers datasets for fine-tuning a model with reinforcement learning for calibrated decisions (RLCD-style): rewards that combine correctness with a proper scoring rule on the model's stated probability. For background, see the [README](README.md) and [lit_review.md](lit_review.md).

**How to read this page**
- **License** is the tag on the linked Hugging Face card, or the GitHub repository license, as checked on 2026-10-21. Code licenses and data licenses can differ. Always read the dataset card and the original paper's terms before use. ⚠️ marks non-commercial, share-alike, gated, or unclear terms.
- **Size** is the Hugging Face size category (a range), not an exact count, unless stated otherwise.
- **Used by** lists papers in this list whose full text we checked. "Train" or "eval" is given only where the paper states the role. "Mentioned" means the dataset name appears in the paper but its role was not confirmed.
- **Unverified** means we could not confirm the item. Nothing here is guessed.

## What makes a dataset usable for a scoring-rule reward

A Brier or log-score reward needs a binary correctness signal y ∈ {0,1} for each answer, so the dataset has to support **automatic, reliable answer checking**:

| Answer type | Verification for the reward |
|---|---|
| Multiple choice / binary | Exact match on the option letter or label. Most reliable. |
| Numeric | Parse the final answer and compare after normalization (tolerance, fractions, units). Math-verify-style checkers for LaTeX. |
| Short free-form | Normalized exact match or F1 against alias lists; an LLM judge for paraphrases (adds noise, so audit it). |
| Unanswerable / abstain | A label marks no-answer items. Reward abstention there; on answerable items, treat abstention per your ternary scheme. |
| Forecast (resolved event) | The outcome resolves later. The reward is the Brier or log score of the stated probability against the resolution. |

## 1. Factual QA (short free-form)

| Dataset | Link | License | Size | Answer type | Verification | Used by |
|---|---|---|---|---|---|---|
| TriviaQA | [HF](https://huggingface.co/datasets/mandarjoshi/trivia_qa) | ⚠️ `unknown` on HF card (unverified) | 100K–1M | Free-form short | Normalized EM against alias lists | Rewarding Doubt (train); RLCR, Kadavath et al., Tian et al. (mentioned) |
| Natural Questions | [HF](https://huggingface.co/datasets/google-research-datasets/natural_questions) · [GitHub](https://github.com/google-research-datasets/natural-questions) | ⚠️ HF card: CC BY-SA 3.0 (share-alike); GitHub repo: Apache-2.0 (likely code) | 10K–100K (HF category) | Free-form short / long / yes-no | EM on short answers; some items have no short answer | TruthRL (mentioned) |
| HotpotQA | [HF](https://huggingface.co/datasets/hotpotqa/hotpot_qa) | ⚠️ CC BY-SA 4.0 (share-alike) | 100K–1M | Free-form short (multi-hop) | Normalized EM/F1 | RLCR (train; in-domain ECE 0.37→0.03); 2607.04332 (train, including a modified variant); TruthRL (mentioned) |
| SimpleQA | [HF mirror](https://huggingface.co/datasets/basicv8vc/SimpleQA) · [official simple-evals](https://github.com/openai/simple-evals) | MIT (HF mirror tag; official repo licensing not separately verified for data) | 1K–10K | Free-form short | Designed for a single indisputable answer; usually graded with an LLM judge | RLCR (mentioned, OOD eval) |

## 2. Math and reasoning (numeric)

| Dataset | Link | License | Size | Answer type | Verification | Used by |
|---|---|---|---|---|---|---|
| GSM8K | [HF](https://huggingface.co/datasets/openai/gsm8k) | MIT | 10K–100K | Numeric | Parse the number after `####`; exact compare | Taming Overconfidence (eval); RLCR, Reinforced Hesitation, Kadavath et al. (mentioned) |
| MATH | [HF](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | MIT (HF tag) ⚠️ the original `hendrycks/competition_math` page also tags MIT, but availability history should be checked | 10K–100K | Numeric / LaTeX expression | Extract the `\boxed{}` answer and check symbolic equivalence | RLCR (mentioned) |
| Big-Math (RL-Verified) | [HF](https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified) | Apache-2.0 ⚠️ gated (accept terms on HF) | 100K–1M | Numeric / closed-form | Filtered for verifiable answers; rule-based check | RLCR (math training, mentioned); 2607.04332 (train) |

## 3. Multiple choice and binary

| Dataset | Link | License | Size | Answer type | Verification | Used by |
|---|---|---|---|---|---|---|
| MMLU | [HF](https://huggingface.co/datasets/cais/mmlu) | MIT | 100K–1M (HF category, includes auxiliary train) | MC (4) | Option letter EM | Kadavath et al., Taming Overconfidence, Jev benchmark (mentioned) |
| MMLU-Pro | [HF](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | MIT | 10K–100K | MC (up to 10) | Option letter EM | none confirmed in this list |
| ARC (Easy/Challenge) | [HF](https://huggingface.co/datasets/allenai/ai2_arc) | ⚠️ CC BY-SA 4.0 | 1K–10K | MC | Option letter EM | Jev benchmark (eval) |
| CommonsenseQA | [HF](https://huggingface.co/datasets/tau/commonsense_qa) | MIT | 10K–100K | MC (5) | Option letter EM | Rewarding Doubt (OOD eval); RLCR, Taming Overconfidence (mentioned) |
| BoolQ | [HF](https://huggingface.co/datasets/google/boolq) | ⚠️ CC BY-SA 3.0 | 10K–100K | Binary (yes/no) | Label EM | none confirmed in this list |

## 4. Abstention and unanswerable

| Dataset | Link | License | Size | Answer type | Verification | Used by |
|---|---|---|---|---|---|---|
| SQuAD 2.0 | [HF](https://huggingface.co/datasets/rajpurkar/squad_v2) | ⚠️ CC BY-SA 4.0 | 100K–1M | Extractive span or no-answer | EM/F1; empty answer list = unanswerable | none confirmed in this list |
| FEVER (NOT ENOUGH INFO label) | [HF](https://huggingface.co/datasets/fever/fever) | ⚠️ HF card lists both CC BY-SA 3.0 and GPL-3.0; check per component | 100K–1M | 3-way label (SUPPORTS / REFUTES / NEI) | Label EM; NEI ≈ the abstain target | none confirmed in this list |
| AbstentionBench | [HF](https://huggingface.co/datasets/facebook/AbstentionBench) · [GitHub](https://github.com/facebookresearch/AbstentionBench) · [paper](https://papers.neurips.cc/paper_files/paper/2025/file/fb122bfc3f0127a94ded048b5b03496f-Paper-Datasets_and_Benchmarks_Track.pdf) | ⚠️ **CC BY-NC 4.0 (non-commercial)** on HF; GitHub license not asserted | 10K–100K | Mixed; label says whether to abstain | Abstention detection (judge) on unanswerable items | RLCR (mentioned) |
| CRAG | not verified | ⚠️ unverified | unverified | Free-form with "I don't know" credit | Ternary scoring (correct / missing / hallucinated) | TruthRL (preliminary experiments) |

## 5. Forecasting (resolved real-world events)

| Dataset | Link | License | Size | Answer type | Verification | Used by |
|---|---|---|---|---|---|---|
| ForecastBench | [HF](https://huggingface.co/datasets/forecastingresearch/forecastbench-datasets) · [GitHub](https://github.com/forecastingresearch/forecastbench) | ⚠️ HF card: CC BY-SA 4.0; GitHub code: MIT | not stated on card (unverified) | Probability on binary events | Brier score after resolution | none confirmed in this list (related: 2607.00164) |
| Metaculus resolved questions | [metaculus.com](https://www.metaculus.com/) | ⚠️ unverified; check Metaculus API terms before use | unverified | Probability / numeric | Brier or log score against resolution | none confirmed |
| Good Judgment resolved questions | not verified | ⚠️ unverified | unverified | Probability | Brier score | none confirmed |

Forecasting data is the only group where the **label is itself uncertain at answer time** (aleatoric). 2607.00164 reports that Brier-score RLVR still degrades calibration in this setting, so treat it as a separate domain.

## 6. Finance (numeric reasoning over reports)

| Dataset | Link | License | Size | Answer type | Verification | Used by |
|---|---|---|---|---|---|---|
| FinQA | [GitHub](https://github.com/czyssrs/FinQA) | MIT (GitHub repo license) | unverified | Numeric (program-derived) | Execute or compare the numeric answer with tolerance | none confirmed in this list |
| ConvFinQA | [GitHub](https://github.com/czyssrs/ConvFinQA) | MIT (GitHub repo license) | unverified | Numeric, conversational | Per-turn numeric compare | none confirmed in this list |
| TAT-QA | [HF](https://huggingface.co/datasets/next-tat/TAT-QA) · [GitHub](https://github.com/NExTplusplus/TAT-QA) | HF card: CC BY 4.0; GitHub repo: MIT | 10K–100K | Numeric / span / count | Numeric compare with scale and unit handling | none confirmed in this list |

## 7. Other datasets used in the reviewed papers

Included only because a reviewed paper uses them.

| Dataset | Link | License | Size | Answer type | Used by |
|---|---|---|---|---|---|
| SciQ | [HF](https://huggingface.co/datasets/allenai/sciq) | ⚠️ **CC BY-NC 3.0 (non-commercial)** | 10K–100K | MC | Tian et al. (eval); Taming Overconfidence (mentioned) |
| TruthfulQA | [HF](https://huggingface.co/datasets/truthfulqa/truthful_qa) | Apache-2.0 | 1K–10K | MC / free-form | Tian et al. (eval); Kadavath et al., Taming Overconfidence (mentioned) |
| GPQA | [HF](https://huggingface.co/datasets/Idavidrein/gpqa) | CC BY 4.0 ⚠️ gated; the authors ask that examples not be posted online, to prevent contamination | 1K–10K | MC | Reinforced Hesitation (eval); RLCR (mentioned) |
| MedQA (USMLE, 4-option) | [HF](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | CC BY 4.0 (community mirror tag; original license unverified) | 10K–100K | MC | Rewarding Doubt (OOD eval); Reinforced Hesitation (eval) |
| MuSiQue | [HF](https://huggingface.co/datasets/dgslibisey/MuSiQue) | ⚠️ no license tag on this mirror (unverified) | 10K–100K | Free-form multi-hop | TruthRL (mentioned) |
| HellaSwag | [HF](https://huggingface.co/datasets/Rowan/hellaswag) | ⚠️ no license tag on card (unverified) | 10K–100K | MC | Jev benchmark (eval) |
| QAMPARI | unverified | unverified | unverified | Multi-answer list | Rewarding Doubt (multi-answer training) |
| DeepMath-103K | unverified | unverified | about 103K (from name) | Numeric | 2607.04332 (train) |

## Design guidance

1. **Hold out whole domains for OOD calibration.** Calibration gained in-domain often does not transfer. RLCR trains on HotpotQA or math and reports separate out-of-domain results; RLVR worsens OOD calibration while RLCR improves it ([2507.16806](https://arxiv.org/abs/2507.16806)). Rewarding Doubt trains on TriviaQA and tests on CommonsenseQA and MedQA ([2503.02623](https://arxiv.org/abs/2503.02623)). Keep at least one format and one subject area completely unseen.
2. **Spread difficulty.** If base accuracy is near 0% or 100%, the trivially optimal confidence is constant and the scoring rule teaches nothing. Pre-filter with base-model pass@k so you get a range of difficulty. Watch for confidence priors concentrated on a few values, which stall learning ([2609.32470](https://arxiv.org/abs/2609.32470)).
3. **Mix formats on purpose.** Combine MC/binary (clean labels), numeric (rule-checkable) and short free-form (noisier). Report results per format, because aggregate ECE can hide format-specific miscalibration.
4. **Guard against reward hacking.** If correct and incorrect answers get separate confidence rewards, the model can profit from answering wrong with high confidence that it is wrong. Use a non-hackable, strictly proper scheme ([2607.04332](https://arxiv.org/abs/2607.04332)) and tune where it sits on the over/under-confidence spectrum as a hyperparameter. Also:
   - Audit checker noise. A noisy correctness label biases the learned confidence (a limitation the authors note).
   - Do not let the model choose its own answer format in ways that dodge the checker.
   - For ternary abstention rewards, derive λ from the real cost of an error ([2511.11500](https://arxiv.org/abs/2511.11500)).
   - Check that abstention does not collapse to always or never.
5. **Prevent contamination.** Deduplicate train and eval sets across the sources above (HotpotQA and NQ share Wikipedia). Respect gating requests such as GPQA's.
6. **Licensing.** For commercial work, exclude or separately clear the non-commercial sets (AbstentionBench, SciQ) and anything marked unverified. Share-alike sets (CC BY-SA) carry obligations for derivative datasets.

## Evaluation metrics

- **ECE** (expected calibration error): report the bin count and binning scheme, and preferably an adaptive or equal-mass variant too.
- **Brier score**: proper and bin-free; decompose it into reliability and resolution where possible.
- **AUROC** of confidence for separating correct from incorrect answers: measures discrimination, which ECE misses.
- **Selective accuracy / risk-coverage curve** (and AURC): accuracy at fixed coverage levels. This is closest to how a deployed system acts on thresholds.
- **Accuracy and abstention rate**: report them alongside the others to show calibration was not bought by giving up accuracy or by always abstaining.
- For typed decision models, also check **coherence** across logically linked questions ([2609.33209](https://arxiv.org/abs/2609.33209)).
