# Frozen before compute — the Gram model at 45000 images and 800 epochs

Frozen 2026-09-13, before any unit of this wave runs.

## Why this wave exists, and a defect it exposes

`GateDataModule` carves the SSL training set out of CIFAR-10's 50000 by
subtracting the calibration and validation splits. The Gram configs use
n_calibration = 10000 and n_val = 10000, so they trained on **30000**
images, not 45000.

That is not confined to the Gram arm. Every `gate_paper` arm — the
entire MAJOR grid — carries the same split and trained on 30000 images,
while the matched external baselines trained on 45000
(`dataset_sizes.train` in their own run records). Two documents state
"45000 training images" for the like-for-like table. That statement is
false for our side of it. Recorded here, and separately in the results
files, because it bears on a headline comparison and not only on this
wave.

## Arms

CIFAR-10, 3 disjoint seeds each, 800 epochs, batch 256, CIFARResNet
width 64, nested view tree, `gram_tau = 0.1`, `gram_corrected_closure`
true — every method-side value exactly as `gate_v8_gram` had it.

    full        product_endpoint   V7: per-edge operator composition
                                   aligned against an independently
                                   estimated endpoint operator, with the
                                   Gram-corrected closure
    ablation    product_only       the same corrected composition with
                                   the endpoint constraint removed

Changed from `gate_v8_gram`, and nothing else:

    n_calibration  10000 -> 2500       so training is 45000 images
    n_val          10000 -> 2500
    max_epochs       200 -> 800
    checkpoint_milestones            [20, 200, 800]

Checkpoints at 20, 200 and 800 are kept alongside `last.ckpt` and do not
affect selection. They exist so the representation can be read at three
points on the trajectory; that is not recoverable after the fact.

The calibration sample drops from 10000 roots to 2500. Stage-B
coordinates are fitted on it, so every certificate from this wave is
less well conditioned than the 10000-root ones and is NOT comparable
to them at face value. Declared here rather than discovered later.

## The standing verdict this wave runs into

`GRAM_TRAINING_CONDITIONING_APPENDUM_20260910` rule G1: if the arm
trips the divergence guard at tau = 0.1, the result is reported as "the
Gram-corrected closure is not trainable under this recipe", and is NOT
patched by raising the guard or lowering beta.

It already did. The probe of the full arm failed at tau = 0.1 with
train/loss 228 at epoch 0, 609 at epoch 1 and 1164 at epoch 2 against a
bound of 700, with `gram_retained_min` 85.6 then 91.0 of 128.

This wave does not touch the guard, beta, or tau. It changes the data
split and the epoch budget. Whether more steps per epoch — 176 at 45000
images against 117 at 30000 — carries the arm past the initialisation
regime the appendum identified as the cause is an open question and is
exactly what the probe answers, cheaply: the previous failure took 278
seconds.

## Predictions

P-G1. The full arm trips the divergence guard again, within the first
      five epochs. Stated in the direction the existing evidence points:
      the appendum traced the failure to Gram conditioning AT
      INITIALISATION, and a larger training set does not change the
      initialisation. REFUTED if it reaches epoch 20.

P-G2. The ablation (product_only) trains to completion, as it already
      does at 30000 images and 200 epochs (probe 71.55 - 72.90).
      REFUTED if it fails.

P-G3. No prediction about accuracy at 800 epochs for either arm, on
      either the 45000-image gain or the 4x epoch budget. Measured and
      reported.

## Interpretation grid, committed now

    P-G1 holds
        G1 governs and is now supported at two data scales: the
        Gram-corrected closure is not trainable under this recipe. The
        wave stops there for the full arm; no guard is raised, no beta
        is lowered, no tau is re-selected.

    P-G1 refuted (it trains)
        The earlier failure was specific to the 30000-image schedule,
        not to the recipe. G1's verdict is then too broad and gets a
        correction, and the full arm proceeds to 800 epochs.

## Gates

- Probe before fleet, per arm, on its own seed 1. Seeds 2 and 3 only
  after seed 1 reaches status complete.
- Every GPU partition here caps at 24 hours and a unit of this wave is
  estimated at ~32, so units WILL be killed and resumed. The runner
  resumes from `last.ckpt` with a finiteness and scheduler-compatibility
  guard. The submitter's retry cap now counts only attempts that made no
  progress, so a walltime kill does not consume it.
- `gram_retained_min` is logged every step. Per G2 of the earlier
  appendum, if it stays below 128 at convergence the result carries that
  caveat rather than being reported without it.
