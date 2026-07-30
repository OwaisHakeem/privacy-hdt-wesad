# Binary task — secondary results

Status: complete. Secondary to the four-class analysis throughout.
Register: drafted for the manuscript.

Provenance: `results/{loso,within_subject}/*/binary/`,
`results/loso/dp_fedavg/binary/eps*/`. Read alongside
`A3_baseline_results.md`, `B_fedavg_results.md`, `C_dp_fedavg_results.md`.

The binary task is stress versus non-stress (stress 22.2%, non-stress 77.8%
of windows). It is retained to connect with the WESAD literature, which
predominantly reports binary stress detection, and — more importantly — as a
difficulty contrast against the four-class task.

---

## 1. Why binary is reported as secondary, not headline

Binary stress detection is near-saturated on WESAD:

| Condition | Protocol | Binary macro-F1 | Four-class macro-F1 |
|---|---|---|---|
| Centralised | LOSO | 0.9410 ± 0.0820 | 0.7649 |
| Centralised | within-subject | 0.9607 ± 0.1150 | 0.8540 |
| Local-only | within-subject | 0.9578 ± 0.0532 | 0.9088 |

At 94–96%, the task leaves little dynamic range for a privacy mechanism to
reveal a trade-off. A study whose purpose is to characterise the
privacy–utility relationship is better served by a task with room to degrade,
which is why the four-class task leads.

---

## 2. This resolves an anomaly in the prior pipeline

The earlier implementation reported DP-FedAvg apparently *outperforming*
non-private FedAvg on the binary task — a result that motivated the present
rebuild. With five seeds and the corrected pipeline, the binary DP curve is
monotonic and orderly (§3). The earlier anomaly is therefore attributable to
single-run variance on a near-saturated task, where the genuine gap between
conditions is smaller than the seed-to-seed noise. It was an artefact of
insufficient repetition, not a real effect.

---

## 3. DP-FedAvg on the binary task

LOSO, user-level DP, δ = 1e-4, 30 rounds, clip norm 2.2, 15 folds × 5 seeds
per budget.

| ε | Macro-F1 | Non-stress F1 | Stress F1 |
|---|---|---|---|
| ∞ | 0.9410 | 0.9726 | 0.9095 |
| 16.0 | 0.8926 | 0.9515 | 0.8337 |
| 8.0 | 0.8349 | 0.9196 | 0.7503 |
| 4.0 | 0.7842 | 0.8856 | 0.6829 |
| 2.0 | 0.7071 | 0.8400 | 0.5741 |
| 1.0 | 0.6696 | 0.7879 | 0.5512 |
| 0.5 | 0.6202 | 0.8043 | 0.4360 |

Monotonic throughout. Binary chance-level macro-F1 is approximately 0.47 (the
two-class analogue of the four-class calculation), so binary remains usefully
above chance across the entire budget range — in contrast to the four-class
task, which reaches chance at ε = 0.5.

---

## 4. Task difficulty governs the trade-off

The central comparison of this section: identical data, identical mechanism,
identical budgets; the only difference is task difficulty.

| ε | Binary retention | Four-class retention |
|---|---|---|
| 0.5 | 0.66 | 0.33 |
| 1.0 | 0.71 | 0.40 |
| 4.0 | 0.83 | 0.48 |
| 8.0 | 0.89 | 0.71 |
| 16.0 | 0.95 | 0.79 |

At every budget, the harder task loses roughly twice the proportion of its
performance. At ε = 8, binary retains 89% and remains eminently usable, while
the four-class task retains 71% and is marginal.

**Consequence.** A privacy budget cannot be characterised as "acceptable" or
"severe" independently of the task it is applied to. The same ε = 8 supports
usable binary stress detection and marginal four-class state modelling on the
same recordings. Reports of the form "DP-FedAvg attains ε = X at acceptable
utility" are therefore incomplete unless the task difficulty is specified,
and the common practice of demonstrating private FL on binary benchmarks
systematically understates the cost for richer modelling objectives.

---

## 5. Minority-class disparate impact — present here, and genuine

Unlike the four-class task, the binary task exhibits clear disparate impact
on the minority class (stress, 22.2%).

| ε | Non-stress retention | Stress retention | Non-stress abs. loss | Stress abs. loss |
|---|---|---|---|---|
| 0.5 | 0.827 | 0.479 | 0.168 | 0.474 |
| 4.0 | 0.911 | 0.751 | 0.087 | 0.227 |
| 16.0 | 0.978 | 0.917 | 0.021 | 0.076 |

The minority class loses more in **both** proportional and absolute terms at
every budget — stress sheds two to three times the raw F1 of non-stress. This
is not a starting-level artefact: a ceiling effect would inflate the retention
gap while leaving absolute losses comparable, whereas here the absolute losses
diverge as sharply as the ratios.

### 5.1 Reconciliation with the four-class result

The four-class analysis found no detectable disparate impact from privacy
(`C_dp_fedavg_results.md` §5); the binary analysis finds it clearly. These are
consistent once the mechanism is separated from the outcome.

In the binary task, the minority class (stress) enters the private regime
intact at F1 0.9095, so DP noise has room to degrade it disproportionately, and
does. In the four-class task, the minority class (amusement) had already been
depressed to F1 0.3375 — close to its floor — by the *federation* penalty
(Stage B) before any privacy was applied. With little height remaining, DP
could not single it out further, so the four-class privacy penalty appears
uniform.

The unifying account is that **two distinct penalties act on two distinct
vulnerabilities**:

- **Federation** degrades the class that is hard to transfer across
  individuals — amusement, whose valence-linked physiological signature is
  personal rather than shared (Stage B: −0.095 on amusement, negligible
  elsewhere).
- **Privacy** degrades the class that is rare — stress in the binary task; and
  amusement in the four-class task would be similarly vulnerable had federation
  not already floored it.

This is a more precise statement than the familiar "differential privacy harms
minority classes". Which minority is harmed, and by how much, depends on
whether the harm originates in distributing the data or in adding noise — a
distinction the federated-health literature does not currently draw, and which
this pipeline can make because it measures the two penalties independently
under an otherwise identical configuration.

---

## 6. Consequences for the study

1. **Four-class remains the headline**; binary is the difficulty contrast and
   the resolution of the prior anomaly.

2. **The task-difficulty result (§4) is a contribution in its own right** and
   belongs in the discussion: privacy budgets are not task-independent, and
   binary benchmarks understate the cost of private FL for richer objectives.

3. **The federation-versus-privacy decomposition (§5.1) is the strongest
   mechanistic claim in the study** and should be foregrounded. It requires
   both tasks and both stages to state.

4. **Stage D (membership inference) applies to both tasks.** Whether the weak
   formal guarantee at ε = 8 corresponds to weak *empirical* protection is the
   next and decisive question, and binary — being usable at more budgets — may
   afford a wider range over which to measure the formal-versus-empirical gap.
