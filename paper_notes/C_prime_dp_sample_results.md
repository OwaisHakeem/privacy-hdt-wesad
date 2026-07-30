# Stage C-prime — Sample-level differential privacy, and the mechanism comparison

Status: complete. 450 runs.
Register: drafted for the manuscript; tighten at writing time.

Provenance: `results/loso/dp_sample/multiclass/eps{0.5,1,2,4,8,16}/`.
Pilot at `results/loso/dp_sample/multiclass/pilot.json`. Accountant and
mechanism verified by `08_dp_sample_sweep.py --self-test` before any result was
accepted; the class-collapse diagnostic is `08a_diagnose_clipping.py`. Read
alongside `C_dp_fedavg_results.md`, which supplies the user-level comparator.

---

## 1. What changes, and what does not

Stage C protected the participation of a whole **person**. This stage protects
the participation of a single **10-second window**. Everything else — model,
splits, feature columns, per-subject normalisation, 30 fixed rounds, δ = 1e-4,
the ε grid, five seeds, LOSO protocol — is inherited unchanged, so the
difference between the two stages isolates the effect of privacy granularity
alone.

### 1.1 Why the comparison is necessary

Sample-level DP is the guarantee most commonly reported in the federated-health
literature. It is also the weaker guarantee: an adversary is prevented from
learning which seconds of a recording contributed to the model, but may still
learn that a person participated at all. For a Human Digital Twin, whose
subject is a person, that is the wrong unit of protection.

The question this stage answers is therefore not "which mechanism is better"
but "what does the choice of granularity actually cost, and where does the cost
land". Stage C alone could not distinguish a property of user-level DP from a
property of differential privacy in this setting.

### 1.2 Configuration

| Item | Value |
|---|---|
| Mechanism | DP-SGD per client (Abadi et al., 2016), via Opacus 1.6 |
| Granularity | Sample-level (window participation) |
| δ | 1e-4, matching Stage C |
| Rounds | 30, fixed |
| Local epochs | 5, matching Stage B and C |
| Batch size | 32, Poisson-subsampled |
| Clip norm | 3.0 (pilot median 2.9697) |
| Server aggregation | Plain FedAvg, sample-weighted, no server-side noise |
| Accountant | Opacus RDP, calibrated per client for rounds × E local epochs |

Two points warrant statement in the Methods.

**Server aggregation adds no noise.** Each client's guarantee is local, and
averaging is post-processing of differentially private outputs, so the
guarantee carries through without further noising. This is the structural
difference from Stage C, where noise had to be applied at aggregation because
that is where a client's whole contribution is visible.

**Amplification by subsampling applies here and did not in Stage C.** Poisson
subsampling at batch 32 over roughly 230 windows per client gives a sampling
rate near 0.14, which reduces the noise required to reach a given ε
substantially. Stage C used full participation (C = 1.0) and so obtained no
amplification. This asymmetry is the principal reason the two mechanisms
diverge on aggregate utility, and it is a genuine property of the mechanisms
rather than an artefact of the configuration.

### 1.3 Clip norm calibrated from a pilot

A pilot over 6,955 examples with neither noise nor clipping measured
per-example gradient norms: min 0.018, p25 2.533, median 2.970, mean 3.084,
p75 3.413, max 21.611. The clip norm was set to 3.0, close to the median, per
standard practice.

The units differ from Stage C's pilot, which measured whole-client update
norms (median 2.202). That the two medians are similar is coincidental and
should not be read as the mechanisms being comparably scaled.

---

## 2. Results

15 folds × 5 seeds = 75 runs per budget; 450 in total. Spent ε matched the
target to three decimal places at every level.

| ε | σ | Macro-F1 | Amusement F1 |
|---|---|---|---|
| 16.0 | 1.458 | 0.6841 ± 0.0318 | 0.0384 ± 0.1137 |
| 8.0 | 2.329 | 0.6708 ± 0.0281 | 0.0000 ± 0.0000 |
| 4.0 | 3.988 | 0.6627 ± 0.0286 | 0.0000 ± 0.0000 |
| 2.0 | 7.141 | 0.6190 ± 0.0573 | 0.0000 ± 0.0000 |
| 1.0 | 13.188 | 0.5431 ± 0.0732 | 0.0007 ± 0.0062 |
| 0.5 | 24.521 | 0.3812 ± 0.1159 | 0.0076 ± 0.0301 |

Macro-F1 is monotonic in ε. Amusement is not, but its values are so close to
zero across the grid that the ordering carries no information.

---

## 3. Finding 1 — sample-level retains more aggregate utility at every budget

| ε | Sample-level | User-level (Stage C) | Difference |
|---|---|---|---|
| 0.5 | 0.3812 | 0.2493 | +0.132 |
| 1.0 | 0.5431 | 0.3038 | +0.239 |
| 2.0 | 0.6190 | 0.3523 | +0.267 |
| 4.0 | 0.6627 | 0.3658 | +0.297 |
| 8.0 | 0.6708 | 0.5439 | +0.127 |
| 16.0 | 0.6841 | 0.6054 | +0.079 |

Sample-level DP is superior on macro-F1 at every budget, with the advantage
largest in the middle of the range. At ε = 2 it attains 81% of the non-private
federated result (0.7649), where user-level attains 46%. Where Stage C found
that ε = 0.5 produced a model statistically indistinguishable from random
guessing, sample-level at the same budget remains well above chance (0.3812
against 0.2402).

The mechanism is amplification by subsampling (§1.2), and the direction was
predicted in advance. This is not a novel observation in itself — the
amplification result is standard — but its magnitude in this setting is
useful: the choice of privacy granularity is worth between 0.08 and 0.30
macro-F1, which is larger than the entire cost of federation measured in
Stage B (0.016).

---

## 4. Finding 2 — sample-level destroys the rare class entirely

This is the more consequential result, and it inverts the apparent conclusion
of §3.

| ε | Amusement F1, sample-level | Amusement F1, user-level |
|---|---|---|
| 16.0 | 0.0384 | 0.2480 |
| 8.0 | **0.0000** | 0.2320 |
| 4.0 | **0.0000** | 0.1790 |
| 2.0 | **0.0000** | 0.1584 |
| 1.0 | 0.0007 | 0.1367 |
| 0.5 | 0.0076 | 0.1280 |

Under sample-level DP the model never predicts amusement at all across three
consecutive budgets. Under user-level DP the same class retains between 0.128
and 0.248 across the identical grid. The aggregate macro-F1 advantage in §3 is
therefore purchased by abandoning one class of four.

### 4.1 The cause is the noise, not the clipping

A diagnostic run with `noise_multiplier = 0` and the same clip norm of 3.0
recovered every class: stress 0.7310, amusement 0.9091, meditation 0.9934.
Clipping at 3.0 is therefore not responsible. The same diagnostic run with and
without class-weighted loss behaved identically, which excludes an interaction
between class weighting and per-example clipping.

The explanation consistent with the evidence is a straightforward consequence
of what sample-level DP protects. Amusement constitutes 12.2% of windows,
roughly 36 per subject. At batch size 32, a mini-batch contains approximately
four amusement examples. Per-example Gaussian noise is applied to every
example's gradient contribution, so the signal available for a class is
proportional to how many of its examples appear per step. A class with four
examples per batch has its gradient signal drowned; a class with twelve or
more does not.

User-level DP does not exhibit this because its noise is applied once to a
client's aggregate update, in which every class the client holds is already
represented. The unit of protection determines the unit at which signal is
lost.

### 4.2 Consequence for the interpretation of macro-F1

Reporting macro-F1 alone would have shown sample-level DP as uniformly
superior. It is not; it is superior on three classes and catastrophic on the
fourth. This is a general caution for the privacy–utility literature, which
reports aggregate utility metrics almost universally, and it applies with
particular force to clinical settings where the minority class is frequently
the clinically significant one.

---

## 5. The mechanism comparison, stated for the paper

Neither granularity dominates. They fail differently:

| | User-level DP | Sample-level DP |
|---|---|---|
| Protects | the person | the window |
| Matches the HDT threat model | yes | no |
| Aggregate utility | lower | higher |
| Rare-class utility | degraded but retained | eliminated |
| Amplification by subsampling | unavailable at C = 1.0 | available |
| Failure mode | uniform degradation across classes | selective erasure of the rare class |

**Reportable claim.** The choice of privacy granularity is not a
utility–privacy trade-off along a single axis. It determines *where* the loss
falls. User-level DP sacrifices aggregate performance while preserving the
relative standing of the classes; sample-level DP preserves aggregate
performance by sacrificing the rarest class outright. A practitioner selecting
a mechanism on published macro-F1 figures alone would select the mechanism that
silently removes the class they are most likely to care about.

### 5.1 This completes a three-way account of where harm lands

Combined with Stages B and C, the study now distinguishes three distinct
mechanisms of harm, each landing on a different vulnerability:

| Source | Harms the class that is | Evidence |
|---|---|---|
| Federation | hard to **transfer** between people | Stage B: amusement −0.095, other classes unaffected |
| User-level privacy | (no class-specific penalty detectable) | Stage C §5 |
| Sample-level privacy | **rare** in absolute count | This stage §4 |

The binary task supplies the corroborating case for the third row: there,
privacy harmed stress (22.2% prevalence) far more than non-stress, in both
retention and absolute terms (`binary_results.md` §5).

The familiar claim that "differential privacy harms minority classes" is
therefore too coarse in two respects. It does not distinguish the penalty
attributable to federating the data from the penalty attributable to the
privacy mechanism, and it does not distinguish the two privacy granularities,
which harm minorities by entirely different routes and to entirely different
degrees.

---

## 6. Errors and corrections

Recorded because both reached the analysis and one consumed several hours of
compute.

**Clip norm assumed rather than piloted.** The first sweep used
`clip_norm = 1.0`, the Opacus default, without measuring per-example gradient
norms. The observed median was 2.97, so nearly every gradient was clipped to
approximately a third of its magnitude before noise was added. Every condition
collapsed: ε = 0.5 gave macro-F1 0.154, below the chance level of 0.240, and
ε = 4 returned macro-F1 0.14 identically across five different folds — the
signature of a degenerate constant prediction. Approximately four hours of
compute were discarded. The equivalent pilot had been run for Stage C; it
should have been mandatory here.

**Silent no-op state load.** After correcting the clip norm, all classes still
collapsed at every ε including ε = 16, the weakest budget in the grid.
`SampleDPClient._load_global_state` called `load_state_dict(global_state,
strict=False)`. Opacus wraps the model in `GradSampleModule`, whose parameter
keys carry a `_module.` prefix, while the aggregated global state uses plain
keys. Under `strict=False` no key matched, the load silently did nothing, and
every client trained on its own trajectory while ignoring the aggregated model
entirely. The fix adds the prefix and loads with `strict=True`, so any future
mismatch fails loudly. The diagnostic sequence that isolated it is recorded in
§6.1 because the reasoning is reusable.

### 6.1 The diagnostic sequence

1. Clipping only, zero noise, with and without class-weighted loss — classes
   collapsed in **both**, excluding a class-weighting interaction.
2. A single Opacus client, no federation, no noise, evaluated on its own
   training data — macro-F1 1.0 with an exactly correct prediction
   distribution, proving that local DP-SGD training and clipping were sound.
3. Client and global state dictionary keys compared after stripping — an exact
   match, proving that aggregation itself was sound.
4. By elimination, the fault lay in the load between the two, which is where
   it was.

The general principle: when a federated result collapses but a single client
trains correctly, the fault is in the transfer between them, and `strict=False`
is the first thing to inspect.

---

## 7. Limitations

1. **The amusement result rests on a class with 36 windows per subject.**
   The finding is unambiguous — F1 is exactly zero at three budgets — but the
   mechanism proposed in §4.1 is inferred from the batch arithmetic rather
   than measured directly. A batch-size sweep would test it: if the
   explanation is correct, larger batches should partially recover amusement
   at fixed ε.

2. **No batch-size or local-epoch sensitivity analysis was run** for this
   stage, unlike Stage B where E was swept. The sample-level results are
   reported at a single configuration.

3. **Amplification is available to sample-level and not to user-level**
   because Stage C used full participation. A fairer comparison would give
   user-level DP partial participation and its own amplification. This is
   noted as a confound in the §3 comparison, though it does not affect §4,
   which concerns where the loss falls rather than how much of it there is.

4. **N = 15.** For the sixth time in this study, the cohort size bounds what
   can be resolved.

5. **GPU utilisation held near 35%** throughout, indicating the pipeline was
   input-bound rather than compute-bound. This affected runtime only, not
   results.

---

## 8. Consequences for the study

**The formal-versus-empirical argument now spans two mechanisms.** Stage D
established that at user level, empirical attack success is decoupled from the
formal budget. Whether the same holds at sample level is not yet measured and
is the natural next experiment: the attack pipeline of Stage D applied to these
models, giving attack-AUC against ε for both mechanisms on one axis.

**The paper gains a second mechanistic result.** Section 5.1 is a cleaner and
more complete statement than the study could make before this stage, and it is
the kind of claim that requires exactly the matched-configuration design this
pipeline was built for.

**Nothing in Stages A to D is revised by this stage.** Sample-level DP is
reported as a comparator; user-level remains the mechanism the Human Digital
Twin argument requires.

---

## 9. Open items

- Run the Stage D membership-inference attack against the sample-level models.
- Consider a batch-size sweep to test the §4.1 mechanism directly.
- Consider giving user-level DP partial participation, for a comparison in
  which both mechanisms have access to amplification (§7.3).
- Binary task not run under sample-level DP; secondary throughout.
- Verify Abadi et al. (2016) and the Opacus accounting references before
  writing.
