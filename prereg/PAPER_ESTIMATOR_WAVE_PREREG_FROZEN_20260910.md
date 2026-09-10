# Frozen before compute — the paper-estimator wave, 2026-09-10

Frozen BEFORE any unit of this wave runs.  **Disclosure: written after
seeing the paper-arm QC probe survive three epochs with a falling
closure ratio; no aggregate has been viewed.**

## Why this wave exists

An audit of the implementation against ssl_section_preview and
ssl_supplement_preview found that EVERY existing arm (V1-V7) uses a
ridge-regularized whitener, while the supplement specifies truncated
symmetric inverse roots and states in as many words that the ridge
substitution "defines a different operator estimate".  The existing
corpus is therefore internally consistent but is not the paper's
estimator.  This wave produces the paper's results under the paper's
estimator, trained from scratch.

Superseded, not deleted: the V1-V7 corpus stands as a loss-form ladder
under the ridge estimator and will be labelled that way wherever it is
used.  It is not evidence about the paper's method.

## Arms (CIFAR-10, 3 disjoint seeds each)

The ablation ladder is the PAPER'S OWN, quoted from it:

    full      Eq. (25) complete
    beta0     "retains only the dependence objectives and covariance
              regularization" -- the no-composition control
    lambda0   "removes the aggregation head and score"
    alpha0    drops the adjacent-edge scores
    T2        "reduces to one FMCA-AV pair, without an intermediate
              projection" -- the flat control, inside the same estimator

Holding the estimator fixed across the ladder is the point: the V1-V7
ladder varied recipe and estimator together, so its contrasts could not
isolate the composition constraint.

## Predictions

P1. full beats beta0 on layer2 linear probe, three-seed ranges
    disjoint.  This is the composition attribution, now inside one
    estimator.
P2. full beats T2 at layer2 and T2 beats full at layer4 -- the shallow
    /endpoint trade the ridge corpus showed, reproduced under the
    paper's estimator.
P3. full's relative closure error (frozen-coordinate evaluation,
    supplement §1.5) is below beta0's, disjoint over seeds.
P4. No prediction for lambda0 or alpha0; they are measured and reported
    either way.

Refutation is per prediction; a failed prediction is reported as such
and narrows the claim rather than being re-framed.

## Evaluation protocol (supplement §1.5), fixed now

Stage B on a calibration sample of roots fixes level means, Gram
matrices, retained eigenspaces and transforms.  Stage C uses
INDEPENDENT evaluation roots with separate edge and endpoint draws;
evaluation never redefines the feature spaces.  Reported together:
retained ranks, the cutoff tau=1e-3, sampling budgets, endpoint norm,
closure residual, and the relative error ||E||_F / ||Cdir||_F -- a
ratio of NORMS, undefined rather than zero when the denominator
vanishes.

Representation quality is measured on encoder features BEFORE the
projection heads, by the same convex probe used for every other arm and
for the external baselines.

## Carried over unchanged

The 15 matched-budget external baseline units are estimator-independent
(they are other methods) and are reused as-is.  BYOL remains excluded
from competitiveness claims as a collapsed run.

## Gates

- The fleet waits on the paper-arm QC probe completing.
- Retained ranks are logged every epoch; if they stay far below K at
  convergence, every claim from this wave carries that caveat.
- Loud failures; two-commit; disjoint-seed evidence standard.
