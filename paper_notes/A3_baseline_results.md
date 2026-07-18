# Stage A3 — Baseline results

Status: complete. Numbers below are final unless the pipeline changes.
Register: drafted in the register intended for the manuscript; tighten at
writing time rather than rewriting from scratch.

Provenance: `results/loso/centralised/multiclass/`,
`results/within\\\_subject/centralised/multiclass/`,
`results/within\\\_subject/local\\\_only/multiclass/`.
Protocol fixed by stage A2 (`data/processed/splits/manifest.json`).

\---

## 1\. Configuration

|Item|Value|
|-|-|
|Dataset|WESAD, 15 subjects, 4,419 windows|
|Window|10 s, non-overlapping|
|Features|144 (81 chest, 63 wrist), whitelist-selected|
|Task|4-class: baseline / stress / amusement / meditation|
|Model|MLP 144-128-64-4, dropout 0.2, 27,076 parameters|
|Loss|Class-weighted cross-entropy (weights from training partition only)|
|Normalisation|Per-subject standardisation (federation-compatible)|
|Model selection|Validation macro-F1; test evaluated once, after training|
|Seeds|5 per condition|
|Headline metric|Macro-F1|

\---

## 2\. Centralised baseline, LOSO protocol

The utility ceiling for an unseen subject: all training data pooled at one
location, no privacy preserved. 15 folds x 5 seeds = 75 runs.

|Metric|Mean|SD|
|-|-|-|
|Macro-F1|0.7812|0.0983|
|Accuracy|0.8485|0.0734|
|Balanced accuracy|0.7917|0.0929|
|Weighted F1|0.8413|0.0741|

Per class:

|Class|F1|SD|
|-|-|-|
|Stress|0.9183|0.1051|
|Baseline|0.9043|0.0873|
|Meditation|0.8702|0.1079|
|**Amusement**|**0.4320**|**0.3088**|

Accuracy exceeds macro-F1 by 0.067, reflecting the cohort imbalance
(baseline 39.5%, amusement 12.2%). Accuracy is therefore reported but does
not lead.

**Caveat on the SD column.** These deviations pool all 75 runs and so
conflate between-subject variance with between-seed variance. Stage G must
average seeds within fold before reporting a spread. Do not quote these
deviations in the manuscript as they stand.

\---

## 3\. Amusement: the dominant source of degradation

### 3.1 Per-subject fidelity (mean over 5 seeds)

|Subject|F1|Subject|F1|Subject|F1|
|-|-|-|-|-|-|
|S6|0.862|S13|0.543|S11|0.126|
|S9|0.860|S8|0.429|S5|0.071|
|S3|0.769|S7|0.325|S4|0.031|
|S17|0.677|S15|0.316|S10|0.000|
|S2|0.612|S16|0.277|||
|S14|0.582|||||

Range 0.000-0.862 across subjects.

### 3.2 Variance decomposition

|Source|SD|
|-|-|
|Between subjects (15 fold means)|0.296|
|Within subject (5 seeds, averaged)|0.104|

Between-subject variance exceeds within-subject variance by approximately
eight-fold: roughly 89% of the observed spread is attributable to
inter-subject physiological difference rather than training stochasticity.

### 3.3 Stability structure

|Group|Subjects|F1|Seed SD|
|-|-|-|-|
|Reliably learnable|S3, S6, S9|0.77-0.86|0.027-0.073|
|Reliably unlearnable|S4, S5, S10|0.00-0.07|0.000-0.082|
|Marginal|S7, S8, S13, S14, S15, S16|0.28-0.58|0.118-0.235|

S10 records a seed SD of exactly 0.000: across five independent
initialisations, no amusement window was ever correctly classified.

The marginal group is the volatile one, with initialisation alone shifting
F1 by up to 0.235. This is the group expected to be most sensitive to
DP noise at stage C.

### 3.4 Mechanism

Confusion pooled over 75 runs, 2,700 true amusement windows:

|Predicted as|Count|Proportion|
|-|-|-|
|Amusement (correct)|1,234|45.7%|
|Meditation|670|24.8%|
|Baseline|615|22.8%|
|Stress|181|6.7%|

Only 6.7% of amusement windows are misclassified as stress. The model
therefore places amusement correctly on the arousal dimension and fails only
to separate it from the other low-arousal states.

This is consistent with an established limitation of peripheral
physiological sensing. Under the circumplex model of affect (Russell, 1980),
affective states are organised by arousal and valence. Electrodermal,
cardiac and respiratory signals index arousal robustly; valence is far less
accessible to them. Stress is the sole high-arousal condition in the WESAD
protocol and separates cleanly. Amusement, baseline and meditation are all
low-arousal and are distinguished principally by valence.

The degradation is therefore attributable to a sensing limitation rather
than a modelling deficiency.

*Reference to verify before citing: Russell, J.A. (1980), "A circumplex
model of affect", Journal of Personality and Social Psychology.*

\---

## 4\. Local-only baseline, within-subject protocol

The no-collaboration floor: each subject trains on their own chronologically
earlier data and is evaluated on their own later data. Nothing is shared, so
privacy is perfect by construction. 15 subjects x 5 seeds = 75 runs.

|Metric|Mean|SD|
|-|-|-|
|Macro-F1|0.9088|0.1003|
|Accuracy|0.9182|0.0939|
|Balanced accuracy|0.9226|0.0809|
|Weighted F1|0.9139|0.1027|

LOSO is not defined for this condition: a held-out subject contributes no
training data and therefore has no local model. Local-only results are
comparable only with the within-subject centralised run below.

**Comparison with the prior pipeline.** An earlier implementation reported
local-only accuracy of 0.9946 under a random within-subject split. Adjacent
10 s windows are strongly autocorrelated, so random assignment placed
near-duplicate windows on both sides of the partition boundary. Under the
chronological split with a discard buffer introduced at stage A2, accuracy
falls to 0.9182 — a reduction of 7.6 percentage points, which is the
correction rather than a regression.

\---

## 5\. Local-only versus centralised, within-subject

Both conditions evaluated on identical test windows, aggregated identically
(one score per subject per seed; each subject weighted equally).

|Metric|Local-only|Centralised|Difference|
|-|-|-|-|
|Macro-F1|0.9088 ± 0.0953|0.8540 ± 0.0953|+0.0548|
|Accuracy|0.9182 ± 0.0939|0.8773 ± 0.0947|+0.0409|
|Balanced accuracy|0.9226 ± 0.0809|0.8866 ± 0.0694|+0.0360|

Per class:

|Class|Local-only|Centralised|Difference|
|-|-|-|-|
|Amusement|0.8722|0.7060|+0.1662|
|Baseline|0.9102|0.8306|+0.0796|
|Stress|0.9263|0.9304|-0.0041|
|Meditation|0.9264|0.9489|-0.0225|

### 5.1 Statistical testing

Wilcoxon signed-rank test, paired across 15 subjects. Seeds averaged within
subject before testing: the five runs per subject are not independent
observations and treating them as such would inflate significance.

|Comparison|Wins|Mean paired difference|Statistic|p|
|-|-|-|-|-|
|Macro-F1|10/15|+0.0548 ± 0.1304|36.0|0.188|
|**Amusement F1**|**11/15**|**+0.1661 ± 0.2134**|**15.0**|**0.008**|

**Macro-F1: no significant difference detected.** The comparison is
underpowered — at n = 15 and an effect size of approximately d = 0.42, power
is roughly 0.30, and detection would require some 45-50 subjects. The correct
statement is that no difference was detected, not that no difference exists.

**Amusement F1: significant** (p = 0.008). The class was nominated in advance
on mechanistic grounds (section 3.4) rather than selected post hoc from among
the four, so this is a directed test rather than multiple comparison.

### 5.2 Interpretation

Training on a subject's own data yields significantly higher
amusement-detection fidelity than pooling data across subjects. The effect
is confined to amusement — the state least separable by arousal alone — and
does not extend to the task as a whole. Stress shows no advantage in either
direction (-0.0041), indicating that the high-arousal signature is shared
across individuals; meditation marginally favours pooling.

The pattern suggests that physiological signatures of low-arousal positive
affect are individual rather than shared, and that pooling introduces
negative transfer for such states.

### 5.3 Corroboration from LOSO

Amusement fidelity as a function of how personal the training data are:

|Training data|Protocol|Amusement F1|
|-|-|-|
|Own data only|Within-subject, local-only|0.8722|
|Own plus others'|Within-subject, centralised|0.7060|
|Others' data only|LOSO, centralised|0.4320|

The monotonic gradient across two independent protocols is the stronger
evidence and should lead in the manuscript; the signed-rank result supports
it. Presenting the small-sample test as the primary evidence would invert
the strength of the argument.

\---

## 6\. Limitations to state explicitly

1. **Per-subject test partitions are small.** Amusement support is 5-6
windows per subject. Four subjects recorded F1 = 1.0000 exactly. Section 5
findings are indicative and must be framed as such; the LOSO gradient in
5.3 carries the argument.
2. **Cohort size.** N = 15 renders the macro-F1 comparison underpowered.
This is a property of WESAD, not of the analysis, and recurs throughout
the study.
3. **Transductive normalisation.** Per-subject standardisation of the LOSO
test subject uses that subject's own feature statistics. No labels are
used, so this is not label leakage; it corresponds to on-device
calibration. The stricter `baseline\\\_only` variant is implemented and
should be run as an ablation at stage G.
4. **Pooled standard deviations.** Section 2 deviations conflate
between-subject and between-seed variance and require recomputation at
stage G.

\---

## 7\. Consequences for the study

**The comparison bracket is established.**

|Condition|Protocol|Macro-F1|
|-|-|-|
|Centralised (no privacy, unseen subject)|LOSO|0.7812|
|Local-only (perfect privacy, known subject)|Within-subject|0.9088|
|Centralised (no privacy, known subject)|Within-subject|0.8540|

**FedAvg must be justified against the correct floor in each protocol.**
Under LOSO, local-only is undefined and federation is the only option for a
new user; the centralised figure is the ceiling. Under within-subject,
local-only is available, private and communication-free, and federation must
demonstrate an advantage over it to warrant its cost and its disclosure risk.
That local-only does not significantly exceed centralised at macro level
leaves that question open rather than settled.

**Hypothesis carried to stage C.** The marginal subjects identified in 3.3
are expected to be the first to degrade under DP noise, since the reliably
learnable group retains margin and the reliably unlearnable group has none
to lose. If observed, the resulting claim is that privacy does not degrade
twin fidelity uniformly but stratifies the cohort, silently separating
subjects whose twin remains faithful from those whose does not.

**Personalised federated learning is indicated as a discussion point.** If
pooling introduces negative transfer for individual physiological
signatures, approaches retaining subject-specific components warrant
engagement in the discussion, and possibly a follow-up study.

\---

## 8\. Open items

* Recompute stage 2 deviations with seeds averaged within fold (stage G).
* Run the `baseline\\\_only` normalisation ablation (stage G).
* Verify the Russell (1980) citation and locate published WESAD LOSO
baselines for comparison at related-work time.
* Binary task not yet run; secondary throughout.

