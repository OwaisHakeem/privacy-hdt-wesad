# Stage B — Federated averaging (non-private)

Status: complete, including sensitivity analysis.
Register: drafted for the manuscript; tighten at writing time.

Provenance: `results/loso/fedavg/multiclass/`,
`results/within_subject/fedavg/multiclass/` (default E=5),
`.../L1/`, `.../L10/` (sweep). Aggregation verified by
`05_fedavg_baseline.py --self-test` before any result was accepted.

Read alongside `A3_baseline_results.md`, which establishes the bracket this
stage is measured against.

---

## 1. Configuration

| Item | Value |
|---|---|
| Clients | One subject = one client = one Human Digital Twin |
| Clients per fold (LOSO) | 12 |
| Clients (within-subject) | 15 |
| Local epochs (E) | 5 (primary); 1 and 10 swept |
| Client fraction (C) | 1.0 |
| Local optimiser | Adam, lr 1e-3, weight decay 1e-4 |
| Aggregation | FedAvg, weighted by client sample count |
| Class weights | Per client, from that client's own labels |
| Model selection | Validation macro-F1; test evaluated once, after training |
| Seeds | 5 |

Two configuration choices deviate from canonical FedAvg and are stated rather
than assumed.

**Adam rather than SGD.** McMahan et al. (2017) specify SGD locally. Adam's
optimiser state does not survive aggregation and is reset each round. Adam is
nevertheless used here because the centralised baseline uses it: had the
federated condition used SGD, any observed gap would confound federation with
the optimiser, and the cost of federation — the quantity this stage exists to
isolate — would not be measurable. An SGD ablation is available via
`--local-optimizer sgd`.

**Server-side validation.** Under LOSO the server evaluates the global model
on two held-out subjects to determine stopping. This corresponds to an
institution retaining a small labelled validation cohort, which is realistic,
but it is a server-side capability and not pure cross-device federation.
Federated evaluation, in which clients score locally and return only metrics,
would leak less and introduces its own accounting; it is noted as future work.

---

## 2. LOSO protocol — the cost of federation

15 folds x 5 seeds = 75 runs.

| Metric | FedAvg | Centralised (A3) | Difference |
|---|---|---|---|
| Macro-F1 | 0.7649 ± 0.0994 | 0.7812 ± 0.0983 | −0.0163 |
| Accuracy | 0.8594 ± 0.0579 | 0.8485 ± 0.0734 | +0.0109 |
| Balanced accuracy | 0.7779 ± 0.0899 | 0.7917 ± 0.0929 | −0.0138 |
| Weighted F1 | 0.8380 ± 0.0656 | 0.8413 ± 0.0741 | −0.0033 |

Per class:

| Class | FedAvg | Centralised | Difference |
|---|---|---|---|
| Baseline | 0.9089 | 0.9043 | +0.005 |
| Stress | 0.9213 | 0.9183 | +0.003 |
| Meditation | 0.8918 | 0.8702 | +0.022 |
| **Amusement** | **0.3375 ± 0.3474** | **0.4320 ± 0.3088** | **−0.095** |

Three of four classes are unchanged or marginally improved under federation.
The entire penalty falls on amusement, whose standard deviation (0.3474) now
exceeds its mean. This reproduces, and sharpens, the pattern established at
A3: the robust arousal-separable classes are indifferent to how training is
distributed, while the valence-dependent class is not.

### 2.1 Equivalence testing

Paired across 15 folds, seeds averaged within fold before testing.

| Test | Result |
|---|---|
| FedAvg wins | 5/15 folds |
| Mean paired difference | −0.0163 ± 0.0727 |
| Wilcoxon signed-rank | statistic = 41.0, p = 0.3028 |
| 90% CI on the paired difference | [−0.0494, +0.0167] |

A non-significant Wilcoxon does not establish equivalence. Two one-sided
tests (TOST) were therefore applied:

| Equivalence margin | p | Conclusion |
|---|---|---|
| ±0.03 | 0.2390 | not established |
| **±0.05** | **0.0472** | **equivalent** |
| ±0.10 | 0.0003 | equivalent |

**Reportable claim.** Federated averaging is statistically equivalent to
centralised training within a margin of 0.05 macro-F1 (TOST, α = 0.05).

**Disclosure required.** The ±0.05 margin was selected after the confidence
interval had been inspected, and p = 0.0472 is marginal. The margin is
defensible on application grounds — between-subject standard deviation in this
cohort is approximately 0.10 macro-F1, so 0.05 represents half of one
subject-level standard deviation and is arguably immaterial beside individual
variation — but that justification was constructed post hoc and must be
presented as such. The margin is fixed at ±0.05 for every equivalence claim in
the study from this point, and the failure at ±0.03 is reported alongside it.

---

## 3. Within-subject protocol — the case against federation

Subject-weighted throughout (one score per subject per seed, each person
weighted equally), which is the only aggregation comparable with local-only.

| Condition | Macro-F1 | Communication |
|---|---|---|
| Local-only | **0.9088 ± 0.1003** | **0 MB** |
| Centralised | 0.8540 ± 0.0953 | not applicable |
| FedAvg (E=5) | 0.8024 ± 0.1010 | 122.82 MB |

Federated averaging is the weakest of the three conditions while incurring the
only communication cost and the only disclosure channel.

| Comparison | Mean paired difference | Wins | Statistic | p |
|---|---|---|---|---|
| Local-only vs FedAvg | +0.1063 ± 0.1311 | 9/15 | 24.0 | 0.041 |
| Centralised vs FedAvg | +0.0516 ± 0.1089 | 7/15 | 43.0 | 0.359 |

### 3.1 The argument does not rest on significance

Federated averaging offers no utility advantage over local-only training at
any configuration tested, while requiring on the order of 100 MB of transfer
and exposing model updates to a server. The burden of demonstrating benefit
rests with federation; it has not been met. This is a dominance argument and
holds irrespective of the p-value, which is material given that p = 0.041 does
not survive Bonferroni correction (section 6.2) and that the best-performing
configuration returns p = 0.073.

### 3.2 Per-subject results

| Subject | Local-only | Centralised | FedAvg |
|---|---|---|---|
| S2 | 0.8759 | 0.7626 | 0.7951 |
| S3 | 0.9557 | 0.7386 | 0.7443 |
| S4 | 0.9503 | 1.0000 | 0.7262 |
| S5 | 0.6786 | 0.8468 | 0.6902 |
| S6 | 0.8745 | 0.8992 | 0.9183 |
| S7 | 0.9873 | 0.8675 | 0.7093 |
| S8 | 1.0000 | 0.7109 | 0.7603 |
| S9 | 0.9689 | 0.9668 | 0.9869 |
| S10 | 0.9283 | 0.8831 | 0.7128 |
| S11 | 0.9557 | 0.6907 | 0.7574 |
| S13 | 0.8356 | 0.9200 | 0.8672 |
| S14 | 0.7624 | 0.8223 | 0.8138 |
| S15 | 1.0000 | 0.9103 | 0.7141 |
| S16 | 0.9658 | 0.9189 | 0.9676 |
| S17 | 0.8924 | 0.8725 | 0.8731 |

---

## 4. Sensitivity to the local-epoch parameter

A single untuned configuration cannot support a claim about federation itself.
E was therefore swept before the dominance claim was accepted.

| Condition | Macro-F1 | Amusement F1 | vs local-only | FedAvg wins | p (uncorrected) |
|---|---|---|---|---|---|
| Local-only | 0.9088 | 0.8722 | — | — | — |
| FedAvg, E=1 | 0.7949 | 0.4947 | −0.1139 | 2/15 | 0.0054 |
| FedAvg, E=5 | 0.8024 | 0.5170 | −0.1063 | 6/15 | 0.0413 |
| FedAvg, E=10 | 0.8074 | 0.5213 | −0.1013 | 7/15 | 0.0730 |

Dominance holds at every value tested, and the deficit is stable to within
0.013 macro-F1 across a tenfold variation in E. The finding therefore concerns
federation rather than a hyper-parameter, and the claim may be stated as
holding across E ∈ {1, 5, 10} rather than at a single setting.

### 4.1 The mechanism is not client drift

An initial explanation attributed the deficit to client drift: with roughly
170 training windows per client, E = 5 local epochs drive the local loss to
0.04–0.10 while validation accuracy plateaus, which is consistent with clients
overfitting before aggregation.

**The sweep refutes this.** If premature local convergence were responsible,
E = 1 would recover the deficit. It does not; E = 1 is the worst configuration
tested, and performance increases monotonically with E. At E = 1 the local loss
falls only from 0.68 to 0.20 — clients are not overfitting but scarcely
training — and the result is nonetheless poorer.

The explanation consistent with the evidence is that aggregation itself
discards the personal component. No amount of local training preserves it,
because a single global parameter vector cannot represent fifteen individuals
whose personal components are approximately orthogonal; averaging cancels them.
This is the non-IID personalisation problem rather than a training-dynamics
problem, and it accounts for the amusement collapse (0.8722 → 0.5170) more
economically than drift does.

E = 10 additionally reduces communication (103.32 MB against 171.55 MB at
E = 1) by converging in fewer rounds. E = 5 is nevertheless retained as the
primary configuration: it is the standard value of McMahan et al. (2017), all
LOSO results employ it, and the difference is immaterial. The sweep is
reported as sensitivity analysis.

---

## 5. Subject difficulty does not transfer between methods

| Quantity | Value |
|---|---|
| Local-only, mean pairwise seed–seed r | +0.7590 (range +0.566 to +0.918) |
| Local-only, reliability of 5-seed mean (Spearman–Brown) | 0.9401 |
| FedAvg, mean pairwise seed–seed r | +0.8646 (range +0.741 to +0.934) |
| FedAvg, reliability of 5-seed mean | 0.9696 |
| Attenuation ceiling for a cross-method correlation | 0.9547 |
| **Observed cross-method r** | **+0.0483 (p = 0.864)** |

Both measures are highly reliable, so attenuation cannot account for the
near-zero correspondence. The two quantities appear to measure different
properties: local-only difficulty reflects how learnable an individual is from
their own recording, whereas federated difficulty reflects how well a model
averaged across the cohort fits that individual — that is, how typical they
are. Internal consistency and typicality are not the same trait.

**Bounded claim.** At n = 15 the Fisher-z 95% interval on r = 0.048 is
approximately [−0.48, +0.55]. Independence cannot be asserted. What is
excluded is a strong correspondence: given a reliability ceiling of 0.95, any
r above roughly 0.55 is inconsistent with these data.

**Deployment implication.** An individual's local-model performance does not
predict whether federation will serve them. A deployed system cannot triage
users into "federate" and "train locally" on the basis of the one signal it
naturally possesses.

---

## 6. Statistical declarations

### 6.1 Pre-specification

| Test | Status |
|---|---|
| Local-only vs FedAvg (macro-F1) | Pre-specified: stage A3 established local-only as the floor federation must clear |
| Local-only vs centralised (amusement F1) | Pre-specified on mechanistic grounds (arousal/valence, A3 §3.4) rather than selected from among four classes |
| Local-only vs centralised (macro-F1) | Exploratory |
| Centralised vs FedAvg (within-subject) | Exploratory |
| Local-epoch sweep | Sensitivity analysis; p-values descriptive only |
| Cross-method correlation | Exploratory, arising from a refuted analysis (§7) |

### 6.2 Multiple comparisons

Four paired tests were conducted on the same fifteen subjects. Applying
Bonferroni correction over the three pairwise comparisons gives a threshold of
0.0167.

| Test | p | Survives correction |
|---|---|---|
| Local-only vs centralised (amusement) | 0.008 | yes |
| Local-only vs FedAvg (macro) | 0.041 | **no** |
| Local-only vs centralised (macro) | 0.188 | no |
| Centralised vs FedAvg (macro) | 0.359 | no |

Only the amusement comparison survives. The local-only versus FedAvg result is
reported as uncorrected, declared as such, and the dominance argument is
advanced on the cost-and-benefit grounds of §3.1 rather than on it.

---

## 7. Refuted analyses

Recorded because their absence from the manuscript should be deliberate rather
than accidental, and because two of them appeared convincing.

**Personalisation–equity trade-off — refuted.** Correlating local-only
strength against federated gain (F − L) yielded Spearman ρ = −0.7185
(p = 0.0026), apparently indicating that federation assists individuals with
weak local models and harms those with strong ones. The statistic is
mathematically biased toward negative values, since
cov(L, F − L) = cov(L, F) − var(L). Here cov(L, F) = +0.00043 against
var(L) = 0.00846, so the observed correlation was approximately −var(L), which
is the artefact in its entirety. Oldham's unbiased alternative — correlating
the difference against the mean of the two measures — returns ρ = +0.0786
(p = 0.781), and ρ = +0.1978 (p = 0.517) when the two ceiling subjects
(local F1 = 1.0000) are excluded. No effect exists.

**"Per-subject estimates are dominated by noise" — refuted.** Advanced to
explain the near-zero cross-method correlation, and contradicted by the
reliability analysis of §5 (seed–seed r = 0.759).

**"Client drift from premature local convergence" — refuted.** See §4.1.

**"Local-only significantly exceeds centralised at macro level" — retracted.**
Asserted from point estimates before testing; Wilcoxon returns p = 0.188
(A3 §5.1).

---

## 8. Limitations

1. **Cohort size.** N = 15 has bounded three separate claims: the macro
   comparison is underpowered at approximately 0.30; the equivalence margin is
   marginal at ±0.05 and fails at ±0.03; the cross-method correlation interval
   spans roughly one unit. This is the binding constraint on the study.

2. **Server-side validation** under LOSO is not pure cross-device federation
   (§1).

3. **Adam rather than SGD** deviates from canonical FedAvg, deliberately, for
   comparability (§1).

4. **Client fraction fixed at C = 1.0.** Partial participation is a
   large-scale cross-device concern and would add variance to an already
   underpowered study; it remains available as a sensitivity check.

5. **No personalised federated baseline.** The results point directly toward
   methods retaining subject-specific components; none was evaluated. This is
   the most substantial gap and the most likely reviewer request.

6. **Pooled standard deviations** in §2 conflate between-fold and
   between-seed variance and require recomputation at stage G.

---

## 9. Consequences for the study

**The value of federation is confined to the cold-start setting.**

| Setting | Local-only | Recommendation |
|---|---|---|
| New user, no labelled data (LOSO) | undefined | Federation is the only option, and it costs 0.0163 macro-F1 against centralised — statistically equivalent within ±0.05 |
| User with labelled history (within-subject) | 0.9088 | Federation is dominated: lower utility, ~100 MB transfer, disclosure channel |

This is a usable contribution. Once an individual possesses their own labelled
history, federating degrades their twin, consumes bandwidth and opens a
disclosure channel for no measured return.

**The amusement pattern is now established across three conditions.**

| Training data | Amusement F1 |
|---|---|
| Own data only (local-only) | 0.8722 |
| Own and others' (centralised, within-subject) | 0.7060 |
| Others' data only (centralised, LOSO) | 0.4320 |
| Others' data, averaged (FedAvg, LOSO) | 0.3375 |

Monotonic across four points and two protocols. Distribution of training
degrades the valence-dependent class specifically, and aggregation degrades it
further than pooling does.

**Stage C hypothesis, unchanged.** Differential privacy is expected to degrade
the marginal subjects identified at A3 §3.3 before the reliably learnable ones,
producing a more bimodal distribution of per-subject fidelity. The federated
result strengthens the prior: the mechanism that concentrates loss in amusement
is observable before any privacy mechanism is applied.

**Personalised federated learning is now a required discussion point** rather
than an optional one. Note the terminology collision: an established
personalisation algorithm is named Ditto (Li, Hu, Beirami & Smith, ICML 2021),
distinct from Eclipse Ditto, the Digital Twin platform employed here. First
mention requires a footnote and "Eclipse Ditto" must be used consistently
throughout. Both citations require verification.

---

## 10. Open items

- Fix the equivalence margin at ±0.05 for all subsequent claims; disclose its
  post hoc construction.
- Recompute pooled deviations with seeds averaged within fold (stage G).
- Consider an SGD ablation and a partial-participation check (stage G).
- Consider a personalised FL baseline; at minimum, engage with the literature
  in the discussion.
- Verify McMahan et al. (2017), Li et al. (ICML 2021), Zhao et al. (2018) and
  Russell (1980) before citing.
- Binary task not yet run; secondary throughout.
