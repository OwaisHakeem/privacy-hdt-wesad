# Paper #1 — Skeleton

Working title (choose at draft time — three candidates):
- "Where Privacy Hurts: Decomposing Federation and Differential-Privacy
  Penalties in Wearable Stress Detection"
- "Differential-Privacy Granularity as a Fairness Decision in Federated
  Wearable Health"
- "Formal Budgets, Empirical Risk, and Class Fairness: A Matched-Configuration
  Study of Private Federated Learning on WESAD"

Target venue tier: IEEE J-BHI / Computers in Biology and Medicine / IEEE
Internet of Things Journal. Rigour-led, not novelty-led.

Register: British English, formal academic. Numeric values shown explicitly.

---

## The three-sentence pitch (write this first, revisit last)

Federated learning and differential privacy are combined routinely in
wearable-health papers, and their utility cost is reported as a single
privacy–utility number. We show, on one matched pipeline over WESAD, that this
conflates three separable effects: federation and privacy degrade different
classes through different mechanisms, and the choice of differential-privacy
granularity is a fairness decision rather than a privacy one, because both
granularities deliver near-identical empirical protection while formal ε
overstates the real membership risk of both. The practical consequence is that
a practitioner choosing a mechanism on published macro-F1 alone would adopt the
configuration that silently eliminates the clinically important minority class.

---

## Contribution list (the four claims, each mapped to evidence)

C1. **Federation and privacy harm different subgroups through different
    mechanisms.** Federation harms the class hard to *transfer* between people
    (amusement, a personal-valence signature); privacy harms the class that is
    *rare* in count (stress in the binary task). Evidence: Stage B centralised→
    FedAvg −0.095 on amusement alone; binary DP disparate impact on stress in
    both retention and absolute loss. [Notes: B, binary]

C2. **Formal ε overstates empirical membership risk, across both DP
    mechanisms.** Non-private attack AUC ≈ 0.549; any ε ∈ [0.5, 16] collapses it
    to chance (0.498–0.512) for user-level *and* sample-level. Evidence: Stage D
    both mechanisms. [Notes: D, dp-mechanism-comparison]

C3. **DP granularity is a fairness decision, not a privacy one.** Sample-level
    beats user-level on macro-F1 at every ε (amplification) but eliminates the
    rare class (amusement F1 = 0.000 at ε=2,4,8); user-level preserves it;
    both give identical empirical privacy. Evidence: Stage C vs C-prime vs D.
    [Notes: dp-mechanism-comparison]

C4. **Privacy budgets cannot be quoted task-independently.** Same data, same
    mechanism: binary retains 66% utility at ε=0.5 where 4-class collapses to
    chance. Evidence: binary vs multiclass Stage C. [Notes: binary]

Honest novelty statement (for our own discipline, not the paper): C2 confirms
a known theoretical result (Yeom e^ε−1 bound; generalisation-beats-DP,
arXiv 2110.05524) in a new domain. C1 and C3 are, to our search, not published
elsewhere on FL-health. The paper's weight rests on C1+C3 plus the rigour of a
matched-configuration design.

---

## Section-by-section

### Abstract (~200 words, write last)
State the conflation, the matched-configuration method, the three separated
effects, and the practitioner consequence. One sentence of numbers: non-private
0.78 macro-F1, user-level DP at ε=0.5 falls to chance, sample-level holds 0.38
but zeroes the rare class, attack AUC ≤ 0.51 under any budget. End on the
fairness-not-privacy sentence.

### 1. Introduction
- Wearable physiological monitoring is sensitive; FL + DP is the standard
  privacy response.
- The field reports privacy cost as one utility number at one ε. That hides
  structure.
- Gap: no work separates the federation penalty from the privacy penalty, or
  compares DP granularities on where the harm lands, on one pipeline.
- Our contributions: C1–C4 as a bulleted list.
- Explicit scope limit stated up front: N=15 (WESAD), single architecture,
  first paper of a programme; the Digital Twin deployment layer is future work.

### 2. Related work
Four threads, each closed with "what we add":
- **FL for wearable stress/affect on WESAD.** FedStress (87.6% acc), federated
  WESAD (89.69%), HE-FedStress (F1 89.7%). → our utility is representative, not
  the contribution; we study *where* the cost falls, not headline accuracy.
- **DP in federated health.** DP-FedAvg (McMahan 2018), Wei et al. 2020. → we
  run both user- and sample-level and compare their *fairness* signatures.
- **MIA vs DP, empirical-vs-formal.** Yeom 2018 (e^ε−1), arXiv 2110.05524
  (generalisation beats DP under early stopping), arXiv 1912.11328 (neither
  local nor central DP consistently better; favourable only in a small ε
  interval — CLOSEST OVERLAP, must differentiate). → we confirm the decoupling
  in wearable FL and, crucially, tie it to class fairness, which these do not.
- **Subject/class fairness under DP.** Subject-MIA (arXiv 2206.03317);
  "DP hurts minorities" literature. → we show *which* minority and *why*,
  separating the federation route from the privacy route.

### 3. Data and preprocessing
WESAD, 15 subjects, 10 s non-overlapping windows, 144 features (81 chest +
63 wrist), 4-class (baseline/stress/amusement/meditation) + binary secondary.
Class prevalences (baseline 39.5%, stress 22.2%, amusement 12.2%, meditation
26.1%) — state these early; the whole fairness argument keys off them.
Per-subject normalisation. LOSO protocol; train/val/test separation enforced
structurally (cite the test-set-leakage failure mode we designed out).

### 4. Method
- 4.1 Model: 27k-param MLP (144→128→64→C). Deliberately minimal; justify (avoids
  memorisation artefacts at N=15; keeps the study about mechanism, not capacity).
- 4.2 Federation: FedAvg, E=5, per-client class weights. Epoch sweep as
  sensitivity (E∈{1,5,10}).
- 4.3 User-level DP: DP-FedAvg, Gaussian at aggregation, RDP accountant,
  δ=1e-4, clip 2.2 (pilot median), ε grid {0.5,1,2,4,8,16}+∞.
- 4.4 Sample-level DP: Opacus DP-SGD per client, plain FedAvg aggregation
  (post-processing), Poisson batch 32, clip 3.0 (per-example pilot median),
  same grid.
- 4.5 Membership inference: attack-slice design (stratified 80/20 per subject,
  removes the LOSO generalisation-gap confound); calibrated threshold attack;
  AUC + TPR@1%FPR; within-subject memorisation gap. Justify why not shadow-model
  LiRA (N=15 contaminates shadows).
- 4.6 Protocol: 15 folds × 5 seeds; paired tests; chance-level baselines.

### 5. Results
- 5.1 Baselines (Stage A): centralised LOSO 0.7812, local-only 0.9088,
  centralised within-subject 0.8540. Binary 0.9410.
- 5.2 Federation penalty (Stage B): FedAvg ≈ centralised on LOSO (TOST, disclose
  the margin-selection caveat); dominated within-subject; the −0.095 amusement
  effect; E-sweep robustness.
- 5.3 User-level privacy penalty (Stage C): the ε curve; ε=0.5 at chance
  (t=1.20, p=0.25); usable only ε≥8; disparate-impact hypothesis falsified for
  4-class (floor/ceiling confound explained).
- 5.4 Task dependence (binary): 66% retention at ε=0.5 vs 4-class collapse;
  genuine stress disparate impact (retention AND absolute).
- 5.5 Sample-level privacy (Stage C-prime): beats user-level at every ε; the
  amusement annihilation (F1=0 at ε=2,4,8); the clipping-only diagnostic
  proving noise not clipping is the cause.
- 5.6 Empirical privacy (Stage D, both mechanisms): non-private AUC 0.549;
  chance under any budget for both; reproducibility check (0.5488 vs 0.5481);
  the calibration and within-subject-gap caveats.

### 6. Discussion
- 6.1 The three-way decomposition (C1+C3 unified): a table — Federation harms
  transferability; user-level privacy harms uniformly (no class-specific
  penalty detectable); sample-level privacy harms rarity. One pipeline measures
  all three.
- 6.2 Granularity as a fairness decision (C3): the practitioner-picks-the-wrong-
  mechanism argument. This is the sentence the title promises.
- 6.3 Formal vs empirical (C2): budgets are conservative here because the model
  does not memorise; tie to generalisation-beats-DP literature; do NOT overclaim
  ("we broke privacy" is wrong — the attack is weak because leakage is low).
- 6.4 Implications for Human Digital Twins: user-level is the granularity the
  twin argument requires (the unit is a person), and it is precisely the one
  that is infeasible at strong ε for a clinical cohort — motivates larger
  cohorts (our own data collection) and personalised FL.

### 7. Limitations
N=15 bounds ~6 separate claims (list them); single architecture; no personalised
-FL baseline (biggest reviewer gap — pre-empt it); server-side validation under
LOSO; amplification available to sample- but not user-level (fairness of the
utility comparison — state it); within-subject gap uninformative under DP;
Adam not SGD; GPU ~35% (workload not GPU-bound — irrelevant to results, note
only if space).

### 8. Conclusion
Restate C1–C4 in one paragraph. The programme: this paper establishes the
measurement; paper #2 brings the Digital Twin (Eclipse Ditto) deployment and a
twin-specific disclosure channel.

---

## Figures and tables (the money items)

- **F1 (money plot):** two panels sharing the ε axis — utility-vs-ε (user vs
  sample, two lines) and attack-AUC-vs-ε (user vs sample, two lines, chance
  line at 0.5). Shows utility diverges but privacy does not.
- **F2:** per-class F1 vs ε for both mechanisms — the amusement line hitting
  zero for sample-level while user-level holds. This is C3 in one picture.
- **F3:** the decomposition table as a figure — three harm sources × which class.
- **T1:** full per-class F1 at every ε, both mechanisms (appendix-grade detail).
- **T2:** binary vs 4-class retention at each ε (task dependence).
- **T3:** MIA AUC + TPR@1%FPR, both mechanisms, all ε.

---

## Citations to verify before drafting (never cite unseen)

Confirmed via search this session: FedStress (IJLTEMAS 2025, 87.6% WESAD);
federated WESAD 89.69% (PMC cross-domain framework); HE-FedStress (IJRIAS,
F1 89.7%); arXiv 1912.11328 (neither local nor central DP consistently better —
MUST cite and differentiate); arXiv 2110.05524 (generalisation beats DP under
early stopping); arXiv 2206.03317 (subject-MIA); Yeom 2018 (e^ε−1 bound, via
arXiv 1902.08874); Wei et al. 2020 (DP-FL algorithms).

Still to verify by reading, not memory: McMahan 2017 (FedAvg), McMahan 2018
(DP-FedAvg / user-level), Abadi 2016 (DP-SGD), Shokri 2017 (shadow MIA),
Carlini 2022 (LiRA / TPR@low-FPR), Schmidt 2018 (WESAD), the Opacus RDP
accounting reference, and Ditto (Li ICML 2021) — note the terminology collision
with Eclipse Ditto, disambiguate in a footnote when paper #2 arrives.

---

## Draft order (recommended)

1. Results section from the six notes files (the numbers are fixed; this is
   assembly, and it surfaces any inconsistency).
2. Method (also fixed; describes what was done).
3. Discussion (where the thinking is — the three-way decomposition and the
   fairness argument).
4. Related work (position against the searched literature).
5. Introduction (now that the story is proven to cohere).
6. Abstract (last).
7. Figures in parallel with Results.
