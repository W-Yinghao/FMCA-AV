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
