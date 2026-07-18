# Stage C — Differentially private federated averaging (user-level)

Status: complete. 450 runs.
Register: drafted for the manuscript; tighten at writing time.

Provenance: `results/loso/dp_fedavg/multiclass/eps{0.5,1,2,4,8,16}/`.
Accountant and mechanism verified by `06_dp_fedavg_sweep.py --self-test`
before any result was accepted. Read alongside `A3_baseline_results.md` and
`B_fedavg_results.md`, which supply the non-private anchors.

---

## 1. What is protected, and why the previous approach was abandoned

This stage implements **user-level** differential privacy: the guarantee
concerns the participation of a whole client — a person — not an individual
10-second window.

The prior pipeline applied Opacus per client, which yields a **sample-level**
guarantee. An adversary is then prevented from learning which seconds of a
recording were used, but may still learn that an individual participated at
all. For a Human Digital Twin, whose subject is a person, a sample-level
guarantee answers a question this study does not ask. The distinction is the
single most consequential methodological change in this stage and will be the
first thing a reviewer checks.

### 1.1 Configuration

| Item | Value |
|---|---|
| Mechanism | DP-FedAvg (McMahan et al., 2018) |
| Granularity | User-level (client participation) |
| δ | 1e-4, fixed for all conditions |
| Rounds | 30, fixed (not early-stopped) |
| Clip norm C | 2.2 (pilot median) |
| Accountant | RDP, Gaussian mechanism, Balle et al. (2020) conversion |
| Sampling rate | 1.0 — no amplification by subsampling |
| Averaging | Equal weight (required; see §1.2) |
| Inherited unchanged from Stage B | Architecture, splits, E = 5, C = 1.0, per-subject normalisation, 5 seeds |

Everything except the privacy mechanism is inherited from Stage B, so the
difference between the two stages isolates the cost of privacy alone.

### 1.2 Three consequences of the privacy requirement

**Rounds must be fixed.** The noise multiplier is calibrated in advance for a
specific number of compositions. Early stopping would leave budget unspent;
running longer would exceed it. Thirty rounds matches the mean at which
non-private FedAvg converged (31.4), so both conditions receive comparable
training length. Model selection by validation score across those rounds costs
no budget: under LOSO the validation subjects are not clients, so their data
was never in the protected set.

**Equal-weight averaging replaces sample-count weighting.** Weighting by client
data volume, as Stage B does, would make the sensitivity of the sum depend on
how much data a client holds, so a single clip norm could no longer bound one
client's influence and the stated guarantee would not hold. The cost is that
WESAD's uneven recording lengths (284–302 windows) no longer inform the
average.

**No amplification by subsampling.** Full participation (C = 1.0) forgoes the
privacy amplification that partial participation would provide, which would
substantially reduce the required noise. This is a real cost of the Stage B
configuration and is stated rather than concealed.

### 1.3 The clip norm cancels from the signal-to-noise ratio

A pilot over 36 client-rounds with neither noise nor clipping measured update
norms of min 1.683, median 2.202, max 2.618 — a narrow distribution, so
clipping at the median distorts relative magnitudes only mildly.

It is worth recording that the clip norm does not influence the
signal-to-noise ratio. The clipped signal is min(u, C) and the noise on the
averaged update has standard deviation σC/n, so for C ≤ u the ratio is n/σ
irrespective of C. What C governs is the effective step size: at C = 1.0 every
client (minimum norm 1.683) would have been clipped to identical magnitude,
discarding update-size variation and shrinking each step roughly 2.2-fold
across a fixed budget. The harmful case is C substantially exceeding typical
updates, since noise scales with C while signal does not.

---

## 2. The calibrated grid

Noise is chosen to **meet** a privacy target; the target is not inferred from
a convenient noise level. The prior pipeline did the latter and reported ε
near 47 and 78, which correspond to no meaningful guarantee.

| Target ε | σ | Achieved ε |
|---|---|---|
| 0.5 | 35.839 | 0.5000 |
| 1.0 | 19.218 | 1.0000 |
| 2.0 | 10.373 | 2.0000 |
| 4.0 | 5.682 | 4.0000 |
| 8.0 | 3.186 | 7.9998 |
| 16.0 | 1.844 | 15.9998 |

Verified: ε falls monotonically with σ, rises with composition, calibration
inverts the accountant at every target, clipping bounds a norm of 303.603 to
exactly 1.000000, sub-threshold updates pass through unchanged, σ = 0
reproduces plain averaging exactly, and injected noise measures 2.0039 against
an expected 2.0.

This verification was disproportionately important. An under-noised run trains
well and reports a confident ε that is simply false — and ε is the claim.

---

## 3. Results

15 folds × 5 seeds = 75 runs per budget; 450 runs in total.

| ε | σ | Macro-F1 | Baseline | Stress | Amusement | Meditation |
|---|---|---|---|---|---|---|
| ∞ | 0 | 0.7649 ± 0.0994 | 0.9089 | 0.9213 | 0.3375 | 0.8918 |
| 16.0 | 1.844 | 0.6054 ± 0.1108 | 0.7354 | 0.7356 | 0.2480 | 0.7027 |
| 8.0 | 3.186 | 0.5439 ± 0.1024 | 0.6751 | 0.6816 | 0.2320 | 0.5870 |
| 4.0 | 5.682 | 0.3658 ± 0.1215 | 0.3472 | 0.4816 | 0.1790 | 0.4555 |
| 2.0 | 10.373 | 0.3523 ± 0.1177 | 0.3835 | 0.4377 | 0.1584 | 0.4295 |
| 1.0 | 19.218 | 0.3038 ± 0.1103 | 0.3884 | 0.3729 | 0.1367 | 0.3174 |
| 0.5 | 35.839 | 0.2493 ± 0.0779 | 0.2684 | 0.3018 | 0.1280 | 0.2992 |

Both macro-F1 and every per-class F1 are monotonic in ε.

**The cliff lies between ε = 4 and ε = 8**, where macro-F1 rises from 0.3658 to
0.5439 — a step of 0.178, larger than any other interval in the grid and
larger than the entire range from ε = 0.5 to ε = 4 (0.117).

---

## 4. Strong privacy is unattainable, not merely expensive

Utility must be judged against chance, not against zero. For a uniform random
predictor on the observed prevalences, per-class F1 = 2p(0.25)/(p + 0.25):

| Class | Prevalence | Chance F1 |
|---|---|---|
| Baseline | 0.3951 | 0.3062 |
| Stress | 0.2215 | 0.2349 |
| Amusement | 0.1222 | 0.1642 |
| Meditation | 0.2611 | 0.2555 |
| **Macro** | | **0.2402** |

One-sample tests against 0.2402, across 15 fold means with seeds averaged
within fold:

| ε | Macro-F1 | Above chance by | t | p (t) | p (Wilcoxon) |
|---|---|---|---|---|---|
| **0.5** | **0.2493** | **+0.0091** | **1.20** | **0.2501** | **0.2769** |
| 1.0 | 0.3038 | +0.0636 | 7.21 | < 0.0001 | 0.0001 |
| 2.0 | 0.3523 | +0.1121 | 9.01 | < 0.0001 | 0.0001 |
| 4.0 | 0.3658 | +0.1256 | 10.13 | < 0.0001 | 0.0001 |
| 8.0 | 0.5439 | +0.3037 | 29.16 | < 0.0001 | 0.0001 |
| 16.0 | 0.6054 | +0.3652 | 17.27 | < 0.0001 | 0.0001 |

**Principal finding.** At ε = 0.5, user-level DP-FedAvg over a twelve-client
cohort yields a model statistically indistinguishable from random guessing
(p = 0.25). Strong privacy is not costly in this setting; it is unattainable.

**Statistical detectability is not utility.** At ε = 1 and ε = 2 the models are
significantly above chance yet score 0.3038 and 0.3523 on a four-class task —
practically worthless. Only ε ≥ 8 is usable, at 71–79% of non-private
performance, and ε = 8 is a weak guarantee.

**The gap between the budget that protects and the budget that functions is
the substance of this stage.**

### 4.1 Why the arithmetic is unforgiving

Sensitivity is C regardless of cohort size, so the noise added to the sum does
not diminish as clients are added; only the averaging divides it. Hiding one
individual among twelve is intrinsically expensive. Noise is furthermore added
per parameter, so its norm scales with √d — approximately 164.5 for this
27,076-parameter model — while the clipped signal is bounded by C irrespective
of dimension.

Learning survives at all only because noise is zero-mean and partially cancels
across 30 rounds (accumulating as √T) whereas the signal accumulates coherently
(as T), improving the effective ratio by roughly √30 ≈ 5.5.

None of the available levers alters this materially: fewer rounds gain about
1.7-fold; the model is already minimal at 27k parameters; the cohort is fixed
at 15; and subsampling amplification would require partial participation,
halving an already small cohort.

---

## 5. The disparate-impact hypothesis is falsified

Stages A3 and B carried a prediction: differential privacy would degrade
amusement — the fragile, valence-dependent, minority class — disproportionately.

Retention, expressed as the fraction of the non-private value preserved:

| ε | Macro | Baseline | Stress | Amusement | Meditation |
|---|---|---|---|---|---|
| 0.5 | 0.326 | **0.295** | 0.328 | **0.379** | 0.336 |
| 1.0 | 0.397 | 0.427 | 0.405 | 0.405 | **0.356** |
| 2.0 | 0.461 | **0.422** | 0.475 | 0.469 | 0.482 |
| 4.0 | 0.478 | **0.382** | 0.523 | **0.530** | 0.511 |
| 8.0 | 0.711 | 0.743 | 0.740 | 0.687 | **0.658** |
| 16.0 | 0.791 | 0.809 | 0.798 | **0.735** | 0.788 |

**Amusement is never the worst-retained class at any budget.** At ε = 0.5 and
ε = 4 it is the best retained. The class suffering most proportionally at
strong budgets is baseline — the most abundant and most separable class.

### 5.1 Retention is nonetheless the wrong instrument

Baseline begins at 0.9089 and has considerable room to fall; amusement begins
at 0.3375, already close to the floor, and cannot fall as far proportionally.
In absolute terms at ε = 0.5, baseline loses 0.64 F1 while amusement loses
0.21. Retention ratios are confounded by starting level and cannot support a
disparate-impact claim in either direction.

**Defensible statement.** Differential privacy degrades all four classes
substantially, and no class-specific privacy penalty is detectable.

---

## 6. Federation and privacy impose different kinds of harm

This is the stage's most substantive contribution, and it emerged from the
hypothesis failing rather than succeeding.

**Federation (Stage B), centralised → FedAvg:**

| Class | Δ |
|---|---|
| Baseline | +0.005 |
| Stress | +0.003 |
| Meditation | +0.022 |
| **Amusement** | **−0.095** |

**Privacy (Stage C), FedAvg → DP-FedAvg:** approximately uniform across all
four classes at every budget.

> **Distributing the data causes disparate harm. Adding privacy does not.**

The literature routinely conflates these, reporting that differential privacy
harms minority classes without separating the penalty attributable to
federation from the penalty attributable to the privacy mechanism. Both are
measured here independently, on one pipeline, with the same architecture,
splits and seeds — the difference between stages B and C being the privacy
mechanism alone.

---

## 7. Errors and corrections

Recorded because two of them briefly reached the analysis.

**Predicted "ε ≥ 4 survives" — wrong.** ε = 4 retains 48%, which is severe
degradation.

**Then predicted "nothing survives, even ε = 16" — also wrong.** ε = 16 retains
79% and ε = 8 retains 71%; both are usable. The reasoning compared noise norm
against signal norm within a single round and neglected that noise is
zero-mean and partially cancels across rounds while signal accumulates (§4.1).

**Predicted disparate impact on amusement — falsified.** See §5.

**Data incident, recovered.** A three-fold spot-check (`--epsilons 1 8 --folds
0 1 2`) wrote to the same directories as the full sweep and overwrote
`summary.json` for ε = 1 and ε = 8 with three-fold aggregates (ε = 1 macro
0.3417 against a true 0.3038; ε = 8 amusement 0.2951 against a true 0.2320).
Per-fold files were unaffected and identical, the generator being seeded
deterministically per fold, seed and ε, so the summaries were rebuilt from
them with assertions on file and fold counts. All figures in this document
derive from the per-fold files. The corrupted ε = 8 amusement value had briefly
supported an incorrect claim that amusement was proportionally more robust.

---

## 8. Limitations

1. **Cohort size.** Twelve clients per fold. User-level DP at this scale is
   punishing for structural reasons (§4.1), and no configuration escapes it.
   This is the binding constraint, now for the fourth time in the study.

2. **No sample-level comparison yet.** The contrast between protecting a
   person and protecting a window is the natural counterpart to this stage and
   remains unrun.

3. **No amplification by subsampling** (§1.2).

4. **Amusement remains near the floor throughout**, so per-class conclusions
   about it are weakly supported: its standard deviations (0.139–0.248) are
   comparable to its means (0.128–0.248).

5. **Fixed 30 rounds** was chosen to match non-private convergence, not
   optimised for the private setting. Fewer rounds would permit lower noise;
   the trade-off is unexplored.

---

## 9. Consequences for the study

**The paper's central tension is now quantified.**

| Budget | Guarantee | Utility |
|---|---|---|
| ε = 0.5 | strong | at chance |
| ε = 1 | strong | worthless |
| ε = 8 | weak | usable (71%) |
| ε = 16 | very weak | usable (79%) |

No budget offers both meaningful protection and meaningful utility at this
cohort size.

**Stage D is now the decisive experiment.** The formal guarantee at ε = 8 is
weak on paper. Whether an adversary can actually exploit it is unknown, and it
is precisely the question this study exists to answer. If membership inference
achieves little better than chance at ε = 8, then formal budgets are severely
conservative in this setting and practitioners are surrendering utility for
protection they already possessed. That is a finding the field needs, and it
does not depend on the twin-layer probe succeeding.

**A sample-level comparison is now required rather than optional.** The
contrast — user-level DP protects the person but is infeasible at n = 12;
sample-level DP is feasible but protects the window rather than the person —
is the formal-versus-empirical argument the study is built upon.

---

## 10. Open items

- Run the sample-level comparison.
- Consider a rounds/noise trade-off sweep (fewer rounds permit lower σ).
- Recompute Stage 2 deviations with seeds averaged within fold (stage G).
- Verify McMahan et al. (2018), Balle et al. (2020), Abadi et al. (2016)
  before citing.
- Binary task not yet run; secondary throughout.
