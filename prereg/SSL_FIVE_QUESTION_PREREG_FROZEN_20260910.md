# Frozen before compute — the five-question SSL argument, 2026-09-10

Frozen BEFORE any job in this wave is submitted.  Scope: the five
questions the paper's SSL section must answer.  Canonical method: the
**Gram-corrected** pipeline — every certificate quantity reported in
this wave is the projection (Gram-corrected) readout, with the
surrogate kept alongside only as a diagnostic.

Backbone note that makes the cross-method comparison legitimate: the
baseline stack and the gate stack instantiate the SAME class
(`fmca_av/resnet.py:CIFARResNet`, width 64), so per-stage features are
the same objects on both sides and no adapter changes the network.

## Q1 — final-layer competitiveness

Endpoint: CIFAR-10 linear-probe top-1 on frozen backbone features,
3 seeds, against 5 external SSL methods trained at a MATCHED budget
(200 epochs, batch 256, same backbone, same data).

P1.1  Our best arm lands within 3.0 points of the best external
      baseline's mean.  (Competitive, not necessarily best — the
      claimed advantage is shallow-layer, not endpoint.)
Refuted if the gap exceeds 3.0 points; then Q1 is reported as
"not competitive at this budget" and the paper says so.

## Q2 — useful representation appears at shallower DEPTH

Endpoints, per backbone stage (layer1..layer4), frozen features,
convex probe, 3 seeds:

P2.1  At layer2, our composed arm (V7) exceeds EVERY external
      baseline's layer2 accuracy, three-seed ranges disjoint.
P2.2  Iso-accuracy depth at three PRE-COMMITTED absolute targets
      (70%, 75%, 80% top-1 on CIFAR-10): V7 reaches each target at a
      stage no deeper than every baseline, and at strictly lower
      cumulative inference FLOPs for at least one target.
      FLOPs counted as backbone prefix + the linear head actually
      used; a method that never reaches a target is reported as
      "not reached", never extrapolated.
Refuted if any baseline matches V7's layer2 range, or if no target
shows a FLOPs saving.

Pre-committed caveat (written now, not after): an external baseline
carries NO training signal at intermediate layers, so P2.1 is a
practical-competitiveness statement.  The MECHANISM claim rests on
Q3's V7-vs-V4 contrast, where both arms are trained at every tap.
This split must survive into the paper's wording.

## Q3 — is the shallow advantage from cross-layer COMPOSITION?

Already-collected contrast (v8 wave, 3 seeds, CIFAR-10/100) is
extended, not re-litigated.  The four-row ladder maps onto existing
arms: V1/V2 = endpoint only (2view/mview); V3/V4 = per-edge trained
separately, NO composition; V7 = composed + endpoint anchor.

P3.1  V7 > V4 at EVERY backbone stage, three-seed ranges disjoint,
      on each dataset measured.
P3.2  The layer2 margin V7-minus-V4 is positive on tin200 as well,
      where V7 LOSES to V4 at the final layer.
Refuted per dataset; a dataset where P3.1 fails is reported as a
failure of the composition attribution on that dataset, and the
claim is narrowed to the datasets where it holds — currently the
final-layer contrast holds on 2 of 3.

## Q4 — does the representation actually support composition?

Reported jointly, never singly (a low defect alone proves nothing):
Gram-corrected composition defect on the held-out evaluation split
with resampled view paths; endpoint operator strength; feature
effective rank; probe accuracy.

P4.1  The Gram-in-training arm (see below) lowers the Gram-corrected
      defect relative to V7 at equal probe accuracy (within 1.0
      point) and without lowering endpoint effective rank.

DISCLOSED IN ADVANCE, because it is already known and must not be
buried: V7's endpoint effective rank is ALREADY markedly below flat
(CIFAR-10 56.7 vs 98.3; CIFAR-100 71.2 vs 112.5).  This is a real
narrowing, consistent with closure confining the endpoint to the
path-supported subspace.  It is not collapse (56.7 of 128 directions)
and it will be reported as a cost of the method, in the same table as
the shallow-layer gain.

### The two Gram roles, kept separate

Type 1 (measurement geometry): whether the corrected projection is the
right way to evaluate operators in non-orthonormal coordinates.
ALREADY ESTABLISHED; those results cannot serve as type-2 evidence.

Type 2 (Gram inside training): a new arm whose composed loss uses the
Gram-corrected composition.  Single-change requirement: EMA, the
faithful_bootstrap product recipe, alpha/beta, stop-gradient, view
budget and epochs are held identical to V7.  Any result from an arm
that changed more than the Gram treatment is void for P4.1.

## Q5 — what does conditional multi-view actually buy?

Estimation level (no training):
P5.1  With parents held FIXED and child views resampled, the
      dispersion of the cross-moment estimate falls monotonically in
      the number of child views m, at approximately 1/m.
Refuted if dispersion is flat in m or falls slower than 1/sqrt(m).

Learning level: already collected (mview beats 2view on the probe in
both loss forms, three datasets); not re-run.

Compute-matched control (P5.2) is NOT part of this wave; it is
declared out of scope here so that no result from this wave is later
presented as if it settled the budget question.

## Gates

- Every fleet is preceded by ONE QC probe unit; no babysitter starts
  without a passing probe.
- Units addressed by real (method, seed) ids.
- Loud failures: a unit that fails writes its reason; no silent skips.
- Two-commit: results, then interpretation.
