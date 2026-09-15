# Effective rank is not a collapse test, and two of my earlier readings
# overstated what it showed

2026-09-13, from the BYOL normalization wave.

## The observation that forced this

`BYOL_NORMALIZATION_PREREG_FROZEN_20260913` predicted (P-B1) that a
fixed BYOL would show layer1 effective rank above 20 of 64, against the
collapsed runs' 5.1. The fixed run reads **6.28**. That clause is
REFUTED, and it should not have been written, because the same profile
shows effective rank does not track collapse at all:

    arm                      layer1 probe  eff_rank    layer4 probe  eff_rank
    BYOL + BatchNorm (fixed)     55.21       6.28         74.76      30.47
    BYOL no BN (collapsed)       43.83       5.11         36.55      58.47
    simclr                       56.12       8.40         81.97      44.27
    vicreg                       56.33      10.03         81.61      33.63

The COLLAPSED arm has the highest layer4 effective rank of the four --
58.47, above simclr's 44.27 and above the fixed BYOL's 30.47 -- while
having by far the worst probe accuracy. A representation that has
failed can be spectrally flat and useless at the same time: high
entropy rank is what noise looks like.

## The test that does discriminate

Whether probe accuracy RISES with depth. That is the defining symptom
in the original diagnosis and it separates cleanly:

    BYOL no BN     43.8 -> 44.3 -> 41.5 -> 36.6      falls    collapsed
    BYOL + BN      55.2 -> 64.2 -> 72.0 -> 74.8      rises    not collapsed

## Applying it to two arms I called degenerate

Both were described using effective rank, and both rise monotonically
with depth across nine block taps:

    endpoint_only  46.5 54.6 56.5 67.3 73.6 74.2 74.7 76.7 77.0   rises
    K256           47.9 54.6 58.2 70.1 76.6 78.9 79.9 79.5 78.6   rises
    composition    50.7 55.4 58.1 72.0 79.6 81.8 82.9 83.1 82.7   rises

Neither is collapsed in the sense BYOL was. Two things I wrote are
therefore too strong and are corrected:

- `PAPER_ESTIMATOR_RESULTS_20260911`, endpoint_only: "the comparator is
  degenerate", "this is not a healthy flat baseline losing to a
  composed one", "It is the endpoint objective ... failing to train a
  well-conditioned representation." The conditioning claim stands --
  effective rank 1.03-1.09 of 128 with over 99% of variance in one
  direction and a trace 5-13x the reference is real and unusual. The
  claim that it is not a usable comparator does not: it decodes
  monotonically to 77.0%.

- `ROBUSTNESS_AXES_RESULTS_20260912`, K256: "does not train to a usable
  representation on any seed." Too strong for the same reason; it
  decodes monotonically to 78.6%. The scale runaway is unchanged and
  remains the finding: trace 291 / 2367 / 70648 against a reference
  arm's 69.7, with 99.87% of the variance in one direction on the third
  seed.

## What this changes downstream

The endpoint_only correction matters most. I had used "the comparator
is degenerate" to argue that composition beating it 82.70 to 73.45,
disjoint, did not license the frozen grid's strong cell, and that an
estimator-controlled flat comparison therefore still did not exist.
That argument is weaker than I stated it. endpoint_only is functional,
so the 9.25-point gap is a comparison between two working arms that
differ in conditioning, not a comparison against a broken one.

I am not replacing it with the strong cell either. The arms differ in
more than the composition constraint -- one is anisotropic to a degree
the other is not -- and the frozen appendum's own rule was that
conditioning is reported beside accuracy, not that it voids the
comparison. The accurate statement is that composition beats
endpoint_only by 9.25 points at the endpoint with disjoint three-seed
ranges, and that endpoint_only reaches that number with a feature
covariance concentrated in one direction.

## Rule for future preregs

A collapse criterion is a statement about whether the representation
carries information, so it must be written against a functional
measure -- probe accuracy and its direction with depth -- and not
against a spectral summary alone. Effective rank and trace are reported
beside it as conditioning, which is a different property.
