# Frozen before compute — ridge full model, replicated with epoch-20 checkpoints

Frozen 2026-09-14, before any unit runs.

## Why

The results draft makes the ridge formulation (V8 `product_endpoint`:
ridge estimator, EMA endpoint target) the practical method, and wants
the representation read at epoch 20 as well as 200. The archived ridge
runs of 20–22 August saved only `last.ckpt`, so an epoch-20 number for
the ridge method cannot be recovered from them. This wave retrains the
same configuration with `checkpoint_milestones: [20, 200]`.

It is a NEW set of runs under the current code (commit at freeze time),
not the archived ones, and it is reported as such. The code has changed
since August in ways tested to leave ridge configs numerically identical
in resolution (estimator switch, K-scaled guard at K=128, step-based
checkpointing, added `retained_min` logging); whether the trained
outcome replicates is what P-R1 asks rather than assumes.

## Arms

`product_endpoint`, ridge, CIFAR-10 and CIFAR-100, three seeds each,
45,000 training images, 200 epochs. Config diff against the archived
`gate_v8` / `gate_c100` templates is exactly `checkpoint_milestones`.

Separately, CIFAR-100 seed 4 of the ARCHIVED collection failed on 21
August at CUDA driver initialization and is rerun into its archived
root so that collection reaches n = 5. Its failed record is preserved
beside it. It is a September run under current code among August runs
and is labelled so wherever that collection is summarised.

## Predictions

P-R1 (replication). At epoch 200 the new units land inside or
    overlapping the archived ranges: CIFAR-10 SGD probe [83.63, 85.56],
    stage-2 convex [77.72, 80.69], stage-4 [83.80, 85.58]; CIFAR-100 SGD
    [55.44, 56.19], stage-2 [54.81, 55.13], stage-4 [55.75, 56.33].
    REFUTED if any of the six new ranges is disjoint from its archived
    counterpart. A refutation means the archived numbers are not
    reproducible under current code and the draft may not mix the two
    collections.

P-R2 (epoch 20). No prediction. The epoch-20 profile is a diagnostic of
    where the representation is early in training. It is reported
    alongside the truncated arm's epoch-20 profile at the same taps.

## Gates

- Ungated across seeds: this arm trained ten units in August and the
  runner has since trained both estimators on this variant family; a
  failure here fails in minutes at the divergence guard.
- Milestone profiles are emitted by the target-state sweep on
  completion. No certificate is emitted for ridge arms; the draft's
  operator diagnostics are truncated-coordinate by design.
