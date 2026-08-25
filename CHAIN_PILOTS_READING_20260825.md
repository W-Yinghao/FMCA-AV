# Chain pilots reading — 2026-08-25

Interpretation commit for `results/chains/20260825_pilots/` (results
committed at 417f045).  Three disjoint 8000-sample draws per cell;
coordinate rule: budget 64, variance floor 1e-2, both declared in the
records.

## Pilot A3 (depth chains): the depth profile is real

**Claim, at the disjoint-range standard.**  On every one of six cells
(resnet18/50/152 x cifar10/100), the projected closure defect falls
from `stem->endpoint` to `layer2->endpoint` in all three seeds, and the
stem and layer2 three-seed ranges are DISJOINT in all six cells.  All
values sit 2-5x above the pairing-shuffled null (<= 0.18).  Reading:
the later the chain starts, the more of what the endpoint carries is
reachable through the retained interface coordinates — a depth
profile of interface sufficiency, measured without resampling any view.

**The ordering is not an artifact of the coordinate rule.**  Sweeping
the variance floor over 1e-3 / 1e-2 / 1e-1 moves the absolute defect
(more retained directions = more measured omission, as a declared part
of the estimand must), but stem > layer1 > layer2 holds at every floor.

**The control behaves.**  Random-init resnet50: every chain is REFUSED
— pooled random activations retain 1 of 1024 directions above the
floor — and the probe curve declines with depth (39.3% stem -> 31.9%
layer4) where the pretrained curve ascends (44.7% -> 88.1%).  No
pretrained structure, no measurable chain, no ascending profile.

## Pilot A1 (self-stitch): the answer is no, and that is the result

The pilot's question: is there an interval where local metrics look
fine, the path composition does not, and the bypass intervention then
fails?  At this configuration the answer is NO, in all three seeds:

- The intervention outcome varies ~50x across intervals
  (endpoint_cka_drop 0.003 -> 0.17), highly reproducible across seeds.
- The projected path defect is nearly FLAT across the same intervals
  (0.16-0.42) and its rank agreement with the intervention flips sign
  across cells (resnet18 negative in every seed, resnet152/cifar100
  positive in every seed).  It does not order intervals by
  bridgeability.  CKA fails the same way (-1.0 to +0.8 by cell).
- The local ridge residual (relative_mse) tracks the intervention at
  spearman +1.0 in all 18 seed-cells — but this is close to
  tautological: the bypass IS the ridge map whose residual relmse
  measures, so this is a consistency check, not a discovery.

**A correction on the record.**  The 2-interval smoke run was read here
as "the bypass agrees with the defect, not CKA."  The full grid refutes
that generalization; the smoke reading was an n=1 overread, same
failure mode as the beta=16 episode.

**Why this is coherent, not a contradiction.**  The 3-state defect
measures whether what the endpoint carries passes through the 64
retained coordinates at the middle tap — suffix sufficiency of the
TAP.  The bypass measures whether the tap is linearly REACHABLE from
the earlier tap.  Different estimands; the grid demonstrates the
difference.  The depth-chain result (A3) is about the first estimand
and stands.  What A1 shows is that the path defect must not be sold as
a stitchability predictor at matched budget — the strategy's
"statistical stitchability" track needs an estimand-level redesign
before any further compute goes to it.

## Status of the framework claims after this round

- Depth-sufficiency profile: supported, three-seed disjoint, control-
  validated, floor-robust.  This is the figure-grade result.
- "Local metrics miss what the path sees" as a stitching claim: NOT
  supported at this configuration; do not use A1 as evidence for it.
- Conditioning lesson (engine-level, from the crashed first round):
  ridge whitening plus low-variance directions makes G^{-1/2} an
  amplifier; the variance floor is now a declared knob bounding
  ||I-G|| — stem deviation 1.000 -> 0.024.  Any past result computed
  on pooled features WITHOUT a floor should be treated as suspect of
  the same artifact; the gate hierarchies used 128 fixed coordinates
  and are not affected.

---

# Revision under the gram_bound rule — same day, later

The lambda_max-relative floor died on ConvNeXt (probe 92%, every chain
refused, 1 direction of 384 retained), and was replaced by the
native-anchored Gram bound: keep directions with
`lambda * f >= (1-f) * rho * s_native`, which bounds every level's
`||I-G||` by `f` by construction.  The full grid was rerun under the
new rule (results commit above this one).  Three findings survive, one
is unlocked, and one interpretation from the morning does NOT survive
its own control.

## What the full grid shows (gram_bound 0.05, 3 disjoint draws)

11 of 12 cells pass the disjoint-range standard: resnet18/50/152,
densenet121 and convnext_tiny are monotone 3/3 with first-vs-last
ranges disjoint on both datasets.  The one exception is vit_b_16 on
cifar100 (monotone 1/3), and ViT was also the weakest cell under the
old rule: the flatness is rule-invariant and consistent with a
constant-width residual stream attenuating less with depth.

## The control result that changes the caption

Under the adaptive rule the random-init network is measurable, and its
depth profile looks like everyone else's: monotone decline in all
three seeds.  Its defect LEVELS overlap the pretrained ones (stem
[0.98,1.57] vs [1.14,1.60]; layer2 [0.448,0.631] vs [0.365,0.451] —
touching at the boundary), and its endpoint dependence is full strength
(top 1.04-1.30 vs pretrained 0.91-0.95): random pooled features carry
as much input information as trained ones, so this is not a
defect-of-nothing artifact.

**Therefore: the monotone depth profile is a property of deep pooled
chain geometry under the declared coordinate rule, and is NOT by
itself evidence of learned depth structure.**  The morning's caption
("a depth profile of interface sufficiency" of the pretrained
network) overclaimed.  What separates trained from random in these
records is the probe DIRECTION (ascending 45->88% vs descending
39->29%): training changes what the dependence is about, not how much
closure the chain has.  The valid use of the chain profile is
comparisons that hold architecture and coordinate rule fixed — which
is exactly the comparison the gate experiments make (same backbone,
different objectives), and a cross-architecture depth figure must be
captioned as architecture geometry, not learning.

## Stitch, second pass: the negative got sharper

Under the new rule the rank agreement between path defect and
intervention outcome is NEGATIVE in 16 of 18 seed-cells (the two
exceptions are one mixed cell).  Lower measured closure defect on an
interval now systematically accompanies WORSE bypass outcomes,
because deeper intervals have lower defect (the depth profile) and
less linearly replaceable computation.  The 08-25 morning conclusion
stands, stronger: the path defect at matched budget must not be used
as a stitchability score.

## L-scan (E-D2): the price list has its curve

At fixed ends, refining the factorization from 2 to 15 interfaces
grows the measured defect superlinearly (c10: 0.32 -> 193) with the
shuffled null growing in proportion (0.12 -> 30), while the Tier-2
compounded radius r_P explodes to vacuity (3.4e7).  Observed growth
stays inside the predicted envelope everywhere.  The operational
reading the paper can print: do not factor a chain finer than the
calibration budget supports, and the certificate's own radii say where
that line is.
