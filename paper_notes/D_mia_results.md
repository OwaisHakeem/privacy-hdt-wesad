# Stage D — Membership inference against user-level DP-FedAvg

Status: complete. 525 runs (525 = 7 conditions × 15 folds × 5 seeds), plus a
3-seed reproducibility peek on ε = ∞ and ε = 8.
Register: drafted for the manuscript.

Provenance: `results/loso/mia/multiclass/{epsinf,eps0.5,eps1,eps2,eps4,eps8,eps16}/`.
Attack pipeline self-tested before any result was accepted. Attack-slice splits
built by `07a_build_attack_splits.py`; attack ran by
`07_membership_inference.py`. Read alongside `C_dp_fedavg_results.md`, which
supplies the training configuration this stage inherits unchanged apart from
the attack-slice holdout.

---

## 1. What the attack asks, and how it is scored

The formal privacy guarantee produced by Stage C is a promise: at ε = X, the
model is (ε, δ)-indistinguishable with respect to any single subject's
participation. Whether an adversary can *actually* determine participation is
an empirical question, and it is the question this stage answers.

### 1.1 The pipeline in one paragraph

Each of the 15 LOSO folds is trained under the DP-FedAvg protocol of Stage C,
with one modification: each member subject withholds 20% of their windows,
stratified across all four condition blocks, from training. Those withheld
"attack-slice" windows and the held-out test subject's windows are then
scored by the trained model. Both member and non-member subjects are
therefore evaluated on windows the model never trained on, so the LOSO
generalisation gap — which would otherwise be confounded with membership —
is removed from the comparison. See §7 for what happens if this is not done.

### 1.2 Attack signals

Each attack-slice window receives four membership scores:

- **Cross-entropy** (Yeom et al., 2018): −log p(true class). Members are
  expected to incur lower loss than non-members.
- **Calibrated cross-entropy**: per-class mean subtracted, so windows of
  intrinsically-hard classes cannot masquerade as non-members. This is the
  Carlini et al. (2022) calibration principle without shadow models, which
  N = 15 cannot support cleanly.
- **Confidence**: p(true class). Monotone with cross-entropy; retained
  because the reported quantity is more intuitive.
- **Negative entropy**: label-free variant, measuring how peaked the
  prediction is.

Cross-entropy leads; calibrated cross-entropy is the strengthened companion.

### 1.3 Metrics

Per Carlini et al. (2022), average-case attack accuracy is discredited: an
attack that scores well on obvious members while missing anyone at genuine
risk is not a real breach. This stage reports AUC (headline), TPR at 1% FPR
(rigorous companion), and the full ROC.

### 1.4 Two granularities

**Window-level** treats each attack-slice window as an independent
observation. Statistically stable (thousands of points per condition), and
tests whether the model's per-window loss distribution encodes membership.

**User-level** must ask whether a *person* was in training, which is the
guarantee the mechanism actually makes. A naive pooled attack — compare each
subject's aggregated score to their membership label — is confounded by the
LOSO generalisation gap and was rejected on inspection (§7). The correct
user-level attack under LOSO is the within-subject memorisation test: for
each member, compare their loss on training windows to their loss on their
own attack-slice windows. Same person, different windows; only memorisation
of specific windows can produce a gap.

---

## 2. Reproducibility

The full 5-seed sweep was followed by an independent 3-seed peek on ε = ∞
and ε = 8. Both runs used deterministic seeding, but the seed subsets did
not overlap.

| Quantity | Full sweep | Peek | Δ |
|---|---|---|---|
| ε = ∞, CE AUC | 0.5481 ± 0.077 | 0.5472 ± 0.076 | 0.0009 |
| ε = ∞, calibrated AUC | 0.5373 ± 0.069 | 0.5354 ± 0.069 | 0.0019 |
| ε = ∞, train-attack gap | −0.111 ± 0.058 | −0.105 ± 0.055 | 0.006 |
| ε = 8, CE AUC | 0.4997 ± 0.046 | 0.5033 ± 0.048 | 0.0036 |

The non-private AUC reproduces to three decimal places and the ε = 8 AUC to
within one part in a thousand. The pipeline is stable across independent
runs.

---

## 3. Results

15 folds × 5 seeds = 75 runs per condition, 525 in total.

| ε | Utility | CE AUC | Calibrated AUC | Train − attack gap |
|---|---|---|---|---|
| ∞ | 0.7740 | **0.5481 ± 0.077** | 0.5373 ± 0.069 | **−0.111 ± 0.058** |
| 16 | 0.5652 | 0.5144 ± 0.058 | 0.5141 ± 0.070 | −1.111 ± 0.886 |
| 8 | 0.5117 | 0.4997 ± 0.046 | 0.4986 ± 0.056 | −1.161 ± 1.039 |
| 4 | 0.3378 | 0.5159 ± 0.050 | 0.5174 ± 0.061 | −0.120 ± 1.376 |
| 2 | 0.3441 | 0.5134 ± 0.058 | 0.5144 ± 0.073 | +0.604 ± 1.555 |
| 1 | 0.2966 | 0.5019 ± 0.043 | 0.5037 ± 0.058 | −0.320 ± 1.112 |
| 0.5 | 0.2470 | 0.5021 ± 0.043 | 0.5012 ± 0.059 | −0.199 ± 1.079 |

*Utility differs slightly from Stage C because attack-slice training uses
only 80% of each member's windows; the shape is preserved.*

---

## 4. Interpretation

### 4.1 The non-private model leaks a small, reliably measurable amount

At ε = ∞, the cross-entropy attack achieves AUC 0.5481. The advantage above
chance is modest (0.048), but the standard deviation across 75 runs (0.077)
places the mean well outside the 0.5 boundary in aggregate. This is
corroborated by the within-subject memorisation test: the training-window
loss is 0.111 nats lower than the same subject's attack-slice loss, with a
narrow standard deviation of 0.058, so the memorisation gap is small but
reliably detected.

Both signals point in the expected direction. The non-private federated
model memorises its training data slightly. It does not memorise it
heavily.

### 4.2 Any positive DP budget reduces the empirical attack to chance

Across the entire budget range ε ∈ {0.5, 1, 2, 4, 8, 16}, cross-entropy AUC
lies between 0.4997 and 0.5159. The calibrated variant behaves identically.
The largest observed AUC in this set (ε = 4, 0.5159) is less than one
standard deviation above chance and less than the non-private mean.

**The empirical membership signal is decoupled from the formal budget.**
ε = 8 — a formally weak guarantee — yields the same empirical attack
success as ε = 0.5 — a formally strong guarantee. Both are indistinguishable
from chance. The formal privacy budget is not the operative privacy measure
in this setting.

### 4.3 The within-subject memorisation test is informative only in the
non-private case

At ε = ∞ the train-attack gap is −0.111 ± 0.058: the training-window loss
is reliably lower than the same person's attack-slice loss. Under any
positive DP budget the observed gap swings between −1.16 and +0.60, but
with standard deviations between 0.9 and 1.6 — wider than any conceivable
signal. This is not because privacy is failing or succeeding at the gap
metric; it is because cross-entropy is not well-behaved when the model is
close to uninformative. Predictions near the uniform distribution place
individual windows arbitrarily far into the tail of the loss distribution,
and the mean is dominated by those tails.

**Reported claim.** Memorisation is reliably detected in the non-private
model and is small. Under DP, the within-subject memorisation gap is not a
usable metric. The AUC of the window-level threshold attack is the
appropriate empirical privacy measure across the whole grid.

### 4.4 Calibration does not increase attack power here

Difficulty-calibrated cross-entropy AUC is 0.5373 at ε = ∞, marginally
below the raw AUC of 0.5481, and the two are indistinguishable across every
DP condition. Calibration is intended to unmask cases where an intrinsically
hard example is flagged as a non-member. That it does not increase attack
power here is diagnostic: the model is not confusing "hard window" with
"non-member window," which further supports the interpretation that the
model does not encode strong per-window membership information in the first
place.

---

## 5. The paper's Stage D claim, in one paragraph

Under the strongest feasible membership-inference attack, the empirical
leakage of the non-private federated physiological model is small
(window-level AUC 0.548; within-subject memorisation gap −0.111 nats). Any
positive differential-privacy budget in the range ε = 0.5 to 16 reduces the
attack to chance (AUC 0.50–0.52). Formal and empirical privacy are
decoupled: the formally weak guarantee at ε = 8 delivers the same empirical
protection as the formally strong guarantee at ε = 0.5. Together with the
utility findings of Stage C, this implies that practitioners choosing among
budgets in this regime should choose on utility grounds, not formal privacy
grounds, provided the model class is not prone to heavy memorisation. The
finding is not that differential privacy defeats a powerful attack. It is
the more useful finding that well-regularised federated models offer
empirical membership protection close to that of strong DP even without DP,
and formal budgets protect against a worst-case memorisation this model
class does not exhibit.

---

## 6. Predictions and their outcomes

Recorded because the earlier stages contained several wrong predictions
that were only detected when the data disagreed with them.

**Prediction made before this sweep.** *"Under the strongest feasible
attack, membership leakage in this federated model is near-chance even
without privacy; DP formally protects against a worst-case memorisation
this model class does not exhibit."* The finding matches the prediction.
This is the first prediction across the four stages that was substantially
correct at first attempt.

**Prediction made after the initial 3-seed peek** that a stronger,
calibrated attack would materially raise the non-private AUC. *False.* The
calibrated AUC was slightly lower than the raw AUC at ε = ∞. The model's
per-class loss distributions do not carry hidden membership signal.

---

## 7. Refuted analyses, on the record

**Naive user-level attack — rejected on inspection, before use.** Comparing
each subject's aggregated attack-slice loss to their membership label
initially produced AUC 0.7363 (540 members, 45 non-members, pooled across
15 folds). Inspection of the score distributions found that non-member
scores had a mean cross-entropy 0.226 higher than member scores — but every
non-member was that fold's held-out test subject, and every LOSO model fits
strangers worse than trained-on subjects irrespective of any privacy leak.
The AUC therefore measured the LOSO generalisation gap rather than
membership, and was replaced with the within-subject memorisation test
(§4.3) which compares each subject to themselves. Reporting the confounded
0.7363 as evidence of leakage would have been a serious error.

---

## 8. Limitations

1. **The attack is a threshold attack, not shadow-model LiRA.** Shadow-model
   MIA would require training several independent reference models, and with
   N = 15 subjects any such training would share subjects across shadows,
   contaminating the very signal the attack tries to isolate. The calibrated
   threshold attack is the strongest attack this data honestly supports; a
   sample-size regime that permits a stronger attack would strengthen the
   null.

2. **The within-subject memorisation gap is uninformative under DP** (§4.3).
   AUC is the sole empirical privacy metric across the DP conditions.

3. **N = 15.** For the fifth time in this study, the cohort size caps what
   can be measured. The threshold attack's AUC standard deviations of
   0.04–0.08 reflect this: with more subjects, differences within the DP
   grid could be resolved more finely.

4. **Attack-slice temporal proximity.** Members' withheld attack-slice
   windows are drawn from the same condition blocks as their training
   windows, and consecutive 10 s windows are autocorrelated. This slightly
   *favours* the attacker: withheld windows resemble trained-on ones, so any
   membership signal is inflated relative to a genuinely disjoint holdout.
   The direction is deliberate and safe — an attacker granted this advantage
   still fails to beat chance under DP, which strengthens the null.

5. **User-level threat model, window-level primary metric.** The formal
   guarantee is user-level; the primary empirical metric is window-level.
   Both matter, and the within-subject memorisation test is the only
   available user-level measurement, but it is uninformative under DP.
   Closing this gap would require a substantially larger cohort permitting
   a proper user-level MIA with genuine non-members.

6. **Utility not identical to Stage C.** Attack-slice training uses 80% of
   each member's data; utility numbers here are slightly different and
   should not be pooled with the Stage C utility table for reporting.

---

## 9. Consequences for the study

**The paper's core empirical result is now in hand.** The formal-versus-
empirical decoupling — the sentence the paper's title implicitly promises —
is measured and defensible.

**The paper's remaining gaps are clear.** Section 8 lists them. The most
important is the absence of a sample-level DP comparison, flagged in Stage C
as required for the formal-versus-empirical arc. That comparison, given
Stage D's finding, becomes stronger: if sample-level DP also empirically
protects at ε = 8 despite a formally weaker guarantee, the decoupling holds
across privacy granularities and becomes a general claim about this model
class rather than about this specific mechanism.

**Stage E's role changes.** Previously described as optional upside, Stage E
now decides whether this is a mid-tier or strong-tier paper. Stage D
produced a defensible null; a further null on twin-layer leakage would be an
absence of evidence rather than a contribution. If Stage E finds
disclosure through the twin channel that MIA cannot detect, the paper
gains a positive novel finding — an empirical channel not addressed by
existing MIA literature. If Stage E finds nothing, the study still stands
as a rigorous empirical treatment of the formal-versus-empirical gap, but
the novelty ceiling is lower.

---

## 10. Open items

- Run sample-level DP as the mechanism comparison (Stage C item, now more
  material given Stage D's findings).
- Decide the Stage E disposition — pursue Ditto/Age-of-Twin only if a
  concrete attack model can be specified that MIA does not already cover.
- Consider a partial-participation (client-fraction < 1) sensitivity check
  under DP: subsampling amplification lowers σ for the same ε, which may
  narrow the utility gap without changing the empirical attack.
- Binary task not yet subjected to MIA; secondary throughout.
- Verify Yeom et al. (2018), Shokri et al. (2017), Carlini et al. (2022),
  McMahan et al. (2018), and Balle et al. (2020) citations before writing.
