# Two follow-ups: a clean 2x2, and a correction to the beta=16 claim

## 1. The redistribution does NOT come from the nesting

The star control keeps the composed loss and breaks only the view
nesting.  Profiling it separates two things that were confounded.

Probe by stage, mean over seeds:

| dataset | arm | layer1 | layer2 | layer3 | layer4 | endpoint gain |
|---|---|---|---|---|---|---|
| CIFAR-10 | nested V7 | 57.49 | 78.60 | 85.44 | 85.46 | +0.01 |
| CIFAR-10 | star (composed loss, broken nesting) | 59.48 | 81.87 | 84.58 | 84.42 | -0.16 |
| CIFAR-10 | flat | 55.92 | 68.53 | 84.81 | 88.77 | **+3.96** |
| CIFAR-100 | nested V7 | 33.40 | 54.97 | 58.73 | 56.06 | -2.67 |
| CIFAR-100 | star | 33.76 | 55.76 | 59.02 | 57.88 | -1.14 |
| CIFAR-100 | flat | 30.70 | 43.84 | 60.42 | 61.17 | +0.75 |

The star arm saturates exactly like the nested arm; on CIFAR-100 its
interior is marginally *better* than nested (`D(2) = -1.39` against
`-1.69`).  So the layer-wise redistribution does not require the
Markov nesting.

### The 2x2 this completes

|  | nested views | star views |
|---|---|---|
| **composed loss** | V7: redistribution YES, defect LOW (0.372-0.378) | star: redistribution YES, defect HIGH (0.481-0.515) |
| **per-edge / endpoint loss** | additive, flat: redistribution NO | (not run) |

Two factors, two separate effects, each with its own control:

- **The composed loss form** produces the interior redistribution.
  Its control is the additive arm, which shares the architecture and
  the views but uses per-edge traces: `D(2) = -6.46` against V7's
  `+0.63` on CIFAR-10.
- **The nested view construction** produces the low certificate
  defect.  Its control is the star arm, which shares the loss and the
  architecture but draws every view from the root: defect 0.481-0.515
  against 0.372-0.378, disjoint.

This supersedes the reading in `LAYERWISE_RESULTS_20260823.md` that
attributed the redistribution to path-supported composition *on nested
views*.  The composition is doing it; the nesting is not.  The two
claims should be made separately, each against its own control.

## 2. Correction: the beta=16 plug-in claim was an n=1 artifact

Reported on 08-22 from a single seed: "beta=16 halves Barlow Twins'
closure defect at no accuracy cost."  At three seeds:

| row | probe, per seed | defect, per seed |
|---|---|---|
| barlow base | 85.30, 84.45, 85.22 | 0.475, 0.646, 0.489 |
| barlow +plugin beta=16 | 85.12, 85.40, 85.11 | 0.241, **0.597**, 0.333 |
| barlow +plugin beta=32 | 83.02, 83.05, 83.22 | 0.203, 0.329, 0.346 |
| vicreg base | 87.01, 87.03, 86.95 | 0.507, 0.626, 0.541 |
| vicreg +plugin beta=16 | 79.93, 83.31, 81.95 | 0.268, **0.660**, 0.209 |

**The accuracy half holds and is now stronger**: barlow beta=16 gives
85.21 +- 0.13 against a base of 84.99 +- 0.38 -- the closure
regularizer at this weight is free, confirmed at three seeds.

**The defect half does not.**  Barlow beta=16 averages 0.391 +- 0.151,
not 0.241: seed 2 lands at 0.597, essentially the base value.  Against
base 0.537 that is a ~27% mean reduction with a range that overlaps
the base range, not the ~55% reduction reported.  beta=32, which costs
2pt of accuracy, is the setting with a reliable defect reduction
(0.293 +- 0.064, no seed above 0.346).

Corrected statement: **beta=16 is accuracy-free but does not reliably
reduce the defect; beta=32 reduces the defect reliably and costs about
2 points.**  The exchange is real but it is an exchange, and the
single-seed sweep point that appeared to escape it did not.

The meeting brief and `fig_beta_sweep` are updated accordingly.
