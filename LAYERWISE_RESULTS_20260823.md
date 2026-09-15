# Layer-wise results: verdicts against this morning's frozen predictions

Predictions frozen in `LAYERWISE_PREDICTION_PREREG_FROZEN_20260823.md`
before any of this was computed.  Nineteen profiles, no training.
Raw output: `results/gate1/LAYERWISE_AGGREGATE_20260823.txt`.

## Probe accuracy by backbone stage (mean of three seeds)

| dataset | arm | layer1 | layer2 | **layer3** (interior) | layer4 (endpoint) |
|---|---|---|---|---|---|
| CIFAR-10 | V7 explicit | 57.49 | 78.60 | **85.44** | 85.46 |
| CIFAR-10 | flat M-view | 55.92 | 68.53 | **84.81** | 88.77 |
| CIFAR-10 | additive | 59.14 | 76.33 | **78.35** | 78.39 |
| CIFAR-100 | V7 explicit | 33.40 | 54.97 | **58.73** | 56.06 |
| CIFAR-100 | flat M-view | 30.70 | 43.84 | **60.42** | 61.17 |
| CIFAR-100 | additive | 33.70 | 49.04 | **49.76** | 49.18 |

## Verdicts

**P1 (interior redistribution): PASS on both datasets, seed ranges
disjoint.**  CIFAR-10 `D(2) = +0.63` (per seed +0.98, +0.73, +0.19)
against `D(3) = -3.31` (-3.19, -3.10, -3.65).  CIFAR-100 `D(2) =
-1.69` against `D(3) = -5.11`, also disjoint.  The strong form
`D(2) > 0` holds on CIFAR-10 only.

The shape is sharper than the prediction asked for.  **The flat arm is
steeply endpoint-loaded; V7 saturates at the interior.**  On CIFAR-10
V7 gains 0.02 points from layer3 to layer4 while flat gains 3.96.  On
CIFAR-100 V7 *declines* from 58.73 to 56.06: its best layer is the
interior interface, not the endpoint.  Truncating V7 at layer3 is free
on CIFAR-10 and better than not truncating on CIFAR-100.

**P2 (anti-compression): REFUTED on both datasets, with the opposite
sign.**  `D_rank(2) = -11.95` (CIFAR-10) and `-15.46` (CIFAR-100);
at the endpoint `-41.59` and `-41.26`.  Closure-trained
representations are markedly *more* compressed: endpoint effective
rank 56.7 against flat's 98.3 on CIFAR-10, 71.2 against 112.5 on
CIFAR-100.

**P3 (interface retention): holds, not specific.**  Mean top-8
canonical correlation across the deepest interface: V7 0.994 / 0.993,
additive 0.999 / 0.999, flat 0.946 / 0.944.  Both hierarchical arms
hand nearly everything forward; flat contracts sharply at the last
interface.  Additive is the highest, so this is not a composition
effect.

**P4 (mechanism specificity): SPLIT, and the split is informative.**
On the probe the interior advantage *is* specific: V7 `+0.63` against
additive `-6.46` on CIFAR-10, `-1.69` against `-10.66` on CIFAR-100,
disjoint in both.  On effective rank it is not: V7 `-11.95` against
additive `-13.64`.  So **explicit composition is what buys interior
representation quality; the compression comes from hierarchical
training in general.**

**P5 (dataset ordering): NOT supported.**  The redistribution gap
`D(2) - D(3)` is 3.94 on CIFAR-10 and 3.42 on CIFAR-100 -- equal
within noise, not larger where the defect separates.

## The mechanism, revised

P2's refutation kills the compression-conflict reading of the endpoint
gap, and the data supports the opposite mechanism, which connects to
factorization rather than to the information bottleneck:

> Closure confines the endpoint to the subspace the path can build.
> The endpoint's effective rank collapses (57 against flat's 98).  A
> lower-dimensional endpoint is less linearly separable, and that is
> what the ~3-5 point endpoint gap is.

This is consistent with `FACTORIZATION_ANALYSIS_20260823.md`, where
hierarchical training moves endpoint dependence into the path-supported
subspace.  The cost is not that the interior retains too much; it is
that the endpoint is allowed to ask for too little.

## Disclosed post-hoc observations (not preregistered)

1. **The largest effect in the study is at layer2, which was not the
   frozen focus**: V7 - flat = `+10.07` (CIFAR-10) and `+11.13`
   (CIFAR-100), tight across seeds.  Additive gets `+7.79` / `+5.19`
   there, so most of it is multi-level attachment giving shallow
   layers direct supervision, with a real but smaller composition
   increment on top.
2. **The alpha=0 arm** (closest available to the pure objective, one
   seed): 58.64 / 80.55 / 83.68 / 83.64 -- the same saturating shape
   with slightly lower values, so the shape is not an artifact of the
   path-blind alpha term.

## What this changes for the paper

The gate reported final-layer probe only, which is the one place the
theory predicts nothing.  On the measurement the theory does care
about, the method is not behind: it reaches its ceiling two stages
early and holds it.  The honest headline is **quality redistribution
with a factorization-induced endpoint cost**, and the endpoint gap
should be reported alongside the layer-wise curve rather than alone.
