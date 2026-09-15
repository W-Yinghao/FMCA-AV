# Frozen before compute — the external baselines at their own recipes

Frozen 2026-09-13, before any unit of this wave runs.

## Why

`configs/ssl_matched` holds one copy-pasted objective/model block
repeated across every method. All six carried the same
`temperature 0.2, off_diagonal_weight 0.0051, invariance_weight 25.0,
variance_weight 25.0, covariance_weight 1.0`, the same
`projection_hidden_dims [512, 512]`, `projection_dim 128`, and a
projector with no normalization. No method was ever configured as
itself; MoCo's queue size and EMA momentum were not in its config at
all and ran on code fallbacks.

This was not visible while every arm trained, and it became visible only
because BYOL — the one method whose collapse avoidance IS the missing
BatchNorm — collapsed outright. The other five each carry an
anti-collapse term in the loss and so absorbed the same defect quietly.

## What changes, and what deliberately does not

CHANGED, because each method's own recipe specifies it:

    all           BatchNorm in projector and predictor
    simclr        projector 2048 -> 128,  temperature 0.5
    moco_v2       projector 2048 -> 128,  temperature 0.2,
                  EMA momentum 0.999 (was a 0.996 code fallback),
                  queue 4096 stated explicitly rather than inherited
    barlow_twins  projector 2048-2048 -> 2048, lambda 0.0051
    vicreg        projector 2048-2048 -> 2048, 25/25/1
    simsiam       projector 2048-2048 -> 2048, predictor bottleneck 512
    fastsiam      the same, at FOUR views

NOT CHANGED: the optimizer. Every arm keeps SGD, lr 0.03, momentum 0.9,
weight decay 1e-4, cosine, batch 256, 200 epochs, 45000 images.

That is a deliberate limit and it is a real one. lr 0.03 at batch 256 is
close to the published SimSiam and MoCo CIFAR settings but is well below
what SimCLR, Barlow Twins and VICReg are normally trained at, and those
three are therefore still expected to be understated. Per-method
learning-rate selection is a search, the matched-budget prereg forbids
tuning a baseline in either direction, and a search run now would be
choosing baseline numbers after having seen our own. The gap is declared
instead of closed.

## Arms

Six methods, three disjoint seeds each, CIFAR-10, checkpoints at epochs
20 and 200. Per-stage probe curves come from `run_baseline_layerwise.py`
on each checkpoint, which is the same convex probe the MAJOR arms are
scored with.

`simsiam` is registered as its own method in this wave. It was
previously reachable only as `fastsiam` at two views, where the two
coincide; beyond two views the shared pair loop computes independent
SimSiam pairs rather than FastSiam's mean-of-others target, so FastSiam
was not implemented at all. It now is, and a test pins both the identity
at two views and the difference at four.

## Predictions

P-R1. simsiam and fastsiam do NOT collapse, on all three seeds: probe
      accuracy rises with depth, as BYOL's did once it had its
      BatchNorm. Both are predictor-plus-stop-gradient methods with no
      negatives and no variance term, so without BN they would have
      collapsed exactly as BYOL did; this wave never runs them without.
      REFUTED if accuracy falls with depth on any seed.

P-R2. Every method's endpoint accuracy is at least as high as its
      `ssl_matched` counterpart: simclr >= 82.28, vicreg >= 81.66,
      barlow_twins >= 80.81, moco_v2 >= 79.30 at layer4, per seed range.
      REFUTED for any method whose range falls below.

P-R3. No prediction about where any of them lands relative to MAJOR.
      Reported either way. MAJOR's main run is at 45000 images and 200
      epochs, so after this wave both sides match on data and epochs;
      the remaining asymmetry is that MAJOR's optimizer was chosen for
      MAJOR and these six share one that was chosen for none of them.

## Gates

- Probe before fleet: seed 1 of `simsiam` first, because it is the one
  arm in this wave whose failure mode is already demonstrated in this
  repo. The rest follow once it trains without collapsing.
- Loud failures; three seeds; the existing `ssl_matched` runs and every
  MAJOR result are untouched.
