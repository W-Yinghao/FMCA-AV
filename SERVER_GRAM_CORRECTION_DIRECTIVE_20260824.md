# Server directive: Gram correction of the certificate pipeline

**Date:** 2026-08-24
**Companion to:** `paper_tpami/THEORY_COMPLETION_NOTES.md` (the Gram correction theorem and the completed proofs in `main.tex`)
**Integrates with:** `T0_VALIDITY_GATE_VERDICT_20260824.md` (gate failed; instrument claims suspended) and `T0_REMEDIATION_PREREG_FROZEN_20260824.md` (frozen remediation)
**Standing:** This document is the draft of a pre-registration appendum. Per the two-step discipline, freeze it as `prereg/T0_APPENDUM_GRAM_CORRECTION_20260824.md` **before** any run it prescribes; nothing here retroactively validates any previously reported number.

---

## 1. What the theorem changes, in one paragraph

With ridge whitening `W_l = (R̂_l + ρ s_l I)^{-1/2}`, the coordinate functions at stage `l` have Gram matrix `G_l = W_l R_l W_l = R_l (R_l + ρ s_l I)^{-1} ≠ I`. Multiplying the whitened cross-matrices `B_{l,l+1}` directly therefore inserts `F_l F_l^*` at every interior interface — which is **not** an orthogonal projection. The error splits exactly as

```
I − F_l F_l^*  =  (I − P_l)  +  Q_l (I − G_l) Q_l^* ,
P_l = F_l G_l^{-1} F_l^* ,   Q_l = F_l G_l^{-1/2} ,
```

where the first term is the genuine finite-subspace omission (the paper's object) and the second is a coordinate-metric / ridge-shrinkage artifact. **Everything the pipeline has reported so far — evaluator and training loss alike — is the sum of the two**, i.e. a *regularized interface surrogate*, not the pure projection defect of the paper's theorems. This is a candidate mechanism for both T0 failures (the random-encoder ratio win and the slow calibration convergence of the hierarchical arms), consistent in direction but not yet demonstrated causally.

**Decision adopted (per the theory notes' recommendation): Path 1** — make the measured and trained object the orthogonal-projection defect by inserting Gram corrections, rather than renaming the current object. Contingency: if the Gram-corrected *training* loss proves unstable (§5), fall back to Path 2 **for training only** — train the surrogate, evaluate corrected, and label the training objective "regularized interface surrogate" in all claims.

---

## 2. Evaluator specification (Stage-C, corrected)

### 2.1 Quantities

Per stage `l`, with frozen Stage-B statistics (`μ_l`, `R̂_l`, `W_l`, ridge `ρ s_l`), coordinate features `z_l = W_l (h_l − μ_l)`:

- **Gram estimate** `Ĝ_l = mean over Stage-B′ samples of z_l z_lᵀ` — estimated on a split **disjoint from Stage-C evaluation data** (reuse the Stage-B pool; the Gram is a Stage-B object).
- **Cross matrices** `B_{l,l+1} = Ê[z_l z̄_{l+1}ᵀ]` (conditional mean over descendants, as now) and `B_{0,L} = Ê[z_0 z̄_Lᵀ]` — unchanged.
- **Corrected composition and endpoint**:

```
Ĉ_comp = Ĝ_0^{-1/2} B_{0,1} Ĝ_1^{-1} B_{1,2} Ĝ_2^{-1} ⋯ Ĝ_{L-1}^{-1} B_{L-1,L} Ĝ_L^{-1/2}
Ĉ_dir  = Ĝ_0^{-1/2} B_{0,L} Ĝ_L^{-1/2}
Δ̂      = Ĉ_dir − Ĉ_comp        (full matrix; δ_F, δ_op as before)
```

- **Regularized inversion**: eigendecompose `Ĝ_l`; retain eigenvalues `≥ τ` (pre-register `τ`; proposal: `τ = 10^{-3} · λ_max(Ĝ_l)`); pseudo-invert on the retained subspace; **log the retained rank per layer per run**. `Ĝ^{-1/2}` likewise.
- **Metric-artifact diagnostic**: report `‖I − Ĝ_l‖_2` per layer, and the defect recomputed with corrections applied cumulatively interface-by-interface, so the surrogate-vs-projection gap is attributed per interface. This quantifies how much of every previously reported defect was ridge artifact.

### 2.2 Reporting format (until the T0 gate is re-passed)

Every defect appears as the tuple `(numerator, denominator, N, ρ, τ, retained ranks)` for **both** the corrected and the uncorrected quantity, labeled `projection` and `surrogate` respectively. No cross-arm ratio comparisons. The suspended list of the T0 verdict remains suspended and now **additionally includes the uncorrected controlled-chain empirical certificates**; the population analytic values of Wave 0 are unaffected (they never passed through ridge whitening).

---

## 3. Finite-sample certificate (two tiers, now implementable)

With frozen network, coordinates, and Gram corrections, compute per-matrix high-probability radii `r_l` (edges) and `r_D` (endpoint) by matrix Bernstein (Tropp), using a pre-registered feature-norm bound (proposal: empirical max `‖z‖` on Stage-B′ with a disclosed 1.5× margin; clip and log violations). Union-bound the joint event; independence across matrices is not required. Then:

```
Tier 1 (true endpoint):        [σ_k(Ĉ_comp) − δ̂_op − r_D]₊              ≤ σ_k(C_dir)
Tier 2 (population path):      [σ_k(Ĉ_comp) − δ̂_op − r_D − 2 r_P]₊     ≤ [σ_k(C_comp) − ‖C_dir − C_comp‖_2]₊
r_P = Π_l (1 + r_l) − 1
```

Report both tiers. Tier 2 is the population path certificate; Tier 1 alone must never be captioned as one.

**Validation target (pre-registered):** on the Wave 0 grid, the Tier-appropriate corrected certificate violates its population target at rate ≤ the nominal α (proposal α = 5%). The E-A5 finding — 13–21% violation rate with magnitude falling an order of magnitude from N=10³ to 10⁵ — is the empirical anchor this must absorb. If violations persist above α after both the Gram correction and the radii, that residual is a real theory gap and gets reported as such, not absorbed into tolerances.

---

## 4. Control battery changes (from the completed theory)

1. **Drop** the single-sided-rotation "must worsen" acceptance criterion. The theorem shows a one-sided rotation may increase, decrease, or preserve the defect; it is no longer a valid negative control. Keep two-sided orthogonal gauge invariance (must be unchanged, as before).
2. **Add** the stronger invariance test the correction unlocks: apply a random **well-conditioned invertible** (not merely orthogonal) reparameterization `A_l` per layer. The **corrected** triplet must be invariant to tolerance (`P_l` depends only on the subspace); the **uncorrected** surrogate will move. Run both — the pair is simultaneously an acceptance test for the new code and a demonstration of the artifact it removes.
3. **Relabel** per-interface telescoping terms as *interface leakage diagnostics*: the theory notes establish they can cancel and depend on expansion direction, so no causal per-layer attribution language.

---

## 5. Training-loss specification

Path 1 applied to training: insert the interior Gram inverses into the composed product inside the loss.

- **Default variant:** Gram matrices computed per batch under the same ridge policy, **detached** (stop-gradient) or EMA-tracked, mirroring the existing EMA-target design; gradients flow through the `B` matrices only.
- **Ablation variant:** fully differentiable Gram inversion (consistent with `GATE1_APPENDUM_DIFFERENTIABLE_WHITENING_20260818`).
- **Stability contingency:** if neither variant trains stably on the v8 recipe, invoke the Path-2 fallback for training only (§1) and say so in the run record.

Naming discipline for history: all completed runs (v5–v8, plug-in, star, ablations) were trained on the **surrogate** objective. Their accuracy, layer-wise probe, and effective-rank results remain valid as reported phenomena of that objective; no run record is rewritten, only relabeled.

---

## 6. Sequencing (integrated with the frozen T0 remediation)

The frozen remediation's structure is preserved; the Gram fix slots in **before** any GPU spend, and may make part (b) unnecessary.

**Step 0 — freeze this appendum.**

**Step 1 — code + unit tests (CPU, first).** Implement §2 and §3. Acceptance:
- analytic chain with deliberately non-orthonormal `F`: corrected defect matches the analytic projection defect to numerical precision; uncorrected shows the predicted `G`-bias;
- six counterexamples re-verified under the corrected evaluator;
- invertible-reparameterization pair test (§4.2);
- ridge sweep `ρ` over two orders of magnitude: corrected defect stable within radii, uncorrected drifts monotonically. Pin each with a mutation test.

**Step 2 — Wave 0 recompute (CPU).** Corrected empirical certificates + two-tier radii over the existing grid; evaluate the α = 5% coverage target of §3.

**Step 3 — extended-sample diagnostic, amended (CPU; remediation part (a)).** Run as frozen, but compute corrected and uncorrected defects side by side, plus the per-interface metric-artifact diagnostic. Pre-registered hypothesis **H-G1:** the hierarchical arms' non-convergence is substantially ridge/Gram artifact, so the **corrected** defect's last-doubling change at N = 10,000 falls below the 0.02 replanning threshold. Pre-registered read:
- H-G1 holds → the retraining of part (b) is **re-scoped**: existing checkpoints may be interpretable under the corrected evaluator without retraining for convergence; part (b) then proceeds only for the training-object mismatch (arms whose *claims* require the corrected training loss), at three seeds not five.
- H-G1 fails → part (b) proceeds as frozen (n_cal = n_val = 10,000), **with the corrected training loss and corrected evaluator**, so the expensive runs are spent on the paper's object, not the surrogate.

**Step 4 — T0 gate re-evaluation.** The frozen gate criteria apply verbatim, evaluated on the **corrected** statistic: last-doubling change < 0.005 at the ceiling for every arm; the random-encoder row scores worse than every trained arm on the reported statistic (or, failing that, the ratio is permanently abandoned for the `(numerator, denominator)` pair with the measured scale floor). Only a pass lifts the suspension; only then does the star control (and any cross-arm separation claim) return to the queue.

**Step 5 — backfill.** Corrected numbers flow to the paper's controlled-validation section; `THEORY_COMPLETION_NOTES.md` §"尚未完成" items 1–2 close (implementation half); the unconditional error-propagation theorem (item 1's theory half) remains open on the theory side and is not claimed.

---

## 7. Explicitly not asserted

Inherited from the theory notes, binding on all run records and captions:

1. No claim that old defect numbers, corrected post hoc, become valid projection certificates — the surrogate training and the ridge bias are entangled in those runs.
2. No claim that the Gram mechanism *causes* the T0 failures until H-G1 is actually tested — it is a theoretically grounded candidate.
3. Without a spectral gap, Weyl-type bounds certify sorted singular values, not semantic directions.
4. The multi-descendant variance theorem covers fixed independent parents and population centering of raw cross-moments; no unconditional claim for fixed total-view budgets, estimated whitening, or training gradients.
5. Accuracy from any reduced-training-set remediation runs is not comparable to the v8 tables (as already frozen).

---

## 8. Cost estimate

Steps 1–3 are CPU-bound (K = 128 matrices; Gram eigendecompositions are trivial). The only GPU spend is the conditional part (b), which this directive can only shrink, never enlarge: the Gram-corrected diagnostic either removes the need for convergence retraining (H-G1 holds) or ensures the retraining that does happen is spent on the object the paper actually defines.
