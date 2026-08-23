# Preregistration: the closure theory's own predictions (frozen 2026-08-23)

Frozen BEFORE any layer-wise measurement is computed.  Every gate so
far reported final-layer probe and kNN.  The path-supported closure
theory does not predict effects there; it predicts them at the
**interior interface**.  This document states what the theory predicts,
what would refute it, and the decision rule -- all before looking.

## 0. Where the theory attaches to this backbone

`level_stages = [1, 2, 3]`, so the three view levels tap backbone
stages 1, 2, 3 (`layer2`, `layer3`, `layer4`); `layer1` (stage 0) sits
below every tap.  In the Theorem-1 defect expansion

    Delta = sum_j Phi_0^* T_{0,1} P_1 ... T_{j-1,j} (I - P_j) T_{j,j+1} ... Phi_L

the only interior projector for a two-edge chain is `P_1`, carried by
**stage 2 (`layer3`)**.  That is where "the intermediate subspace must
be closed under its two neighbouring operators" has to show up, if it
shows up anywhere.  Stage 3 is the endpoint, which the flat arm
optimizes directly.

## 1. Statistics (defined before measurement)

For variant V and stage l:
- `probe(V, l)`: convex multinomial probe (LBFGS from zeros) on the
  frozen pooled stage-l features.  Convex, so the number is a property
  of the features, not of a probe seed.
- `effrank(V, l)`: entropy effective rank of the stage-l test-feature
  covariance -- the same formula the gate already uses at stage 3.
- `cca(V, l -> l+1)`: canonical correlations across the interface
  (contraction factors of the data-processing argument).

Contrasts, always V7 minus a baseline, per seed:
- `D_probe(l) = probe(V7, l) - probe(flat, l)`
- `D_rank(l)  = effrank(V7, l) - effrank(flat, l)`

`additive` is the mechanism control: it is multi-level and multi-view
but **path-blind** (per-operator faithful traces, no composition).

## 2. Predictions

**P1 (interior redistribution).** `D_probe(2) > D_probe(3)`.  The
endpoint penalty is known (~-4pt); the theory says quality is moved
inward, not destroyed.  Strong form: `D_probe(2) > 0`.

**P2 (anti-compression).** `D_rank(2) > 0`, and `D_rank(2) > D_rank(3)`.
Closure fights the information bottleneck: for `C_comp` to catch a
large `C_dir`, interior operators must stay near-isometric on the
relevant subspace, so the interior representation must compress less.

**P3 (retention).** `cca(V7, 1->2)` and `cca(V7, 2->3)` exceed the
flat values: more of each layer is handed forward rather than
contracted away.

**P4 (mechanism specificity -- the decisive one).** P1-P3 hold for V7
but NOT for additive.  If additive shows the same interior profile,
the effect comes from the multi-level architecture and the extra
views, not from path-supported composition, and the theory's
distinctive claim is unsupported even if P1-P3 "pass".

**P5 (dataset ordering).** Effects are at least as large on CIFAR-100
as on CIFAR-10, matching where the instrument already resolves
explicit from additive closure.

## 3. Refutation conditions

- `D_probe(2) <= D_probe(3)` refutes P1: closure training then costs
  accuracy uniformly and the "redistribution" reading is dead.  This
  is the outcome that would make the honest headline "the method is
  simply worse", and it must be reported as such.
- `D_rank(2) <= 0` refutes P2 and, with it, the compression-conflict
  explanation of the 4pt endpoint gap.
- Additive matching V7 on P1-P3 refutes P4 regardless of P1-P3.

## 4. Power and honesty

n = 3 seeds per cell.  No p-values will be reported at this n.  The
pre-committed evidence standard is the one already used for the
CIFAR-100 separation: **per-seed values printed in full, and a claim
of separation only when the three-seed ranges do not overlap.**
Anything weaker is reported as a trend, not a finding.

## 5. Standing caveat on the implementation

The trained v7 objective is not the pure `||C_dir - C_comp||_F^2`
whose gradient motivates these predictions.  It is a normalized
closure ratio against an EMA operator target (beta = 128), plus
`alpha = 0.2` per-edge faithful traces -- which are deliberately
path-blind, being the fix for the vanishing product gradient -- plus a
leaf flat term over the endpoint views.  So V7 is a mixture of a
path-aware term and two path-blind terms, and the predictions above
apply in proportion to beta's share.  A clean test of the pure
objective would need an arm with `alpha = 0` and no leaf term; the
existing `alpha = 0` ablation (83.44) is the closest available and
will be profiled too, as a disclosed secondary.

## 6. Units

CIFAR-10 (`gate1_20260820_v8`) and CIFAR-100
(`gate1_20260821_c100pilot`): `product_endpoint`, `final_mview`,
`additive_mview`, seeds 1-3.  Secondary, CIFAR-10 only:
`gate1_20260821_v8_alpha0` seed 1.  No training; profiles read frozen
checkpoints.
