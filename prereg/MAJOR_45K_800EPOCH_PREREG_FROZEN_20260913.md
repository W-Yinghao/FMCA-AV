# Frozen before compute — MAJOR's main run, 45000 images and 800 epochs

Frozen 2026-09-13, before any unit of this wave runs.

## Why

Two reasons, one a defect and one a budget.

**The defect.** `GateDataModule` takes the SSL training set as what is
left of the 50000 after the calibration and validation splits. Every
`gate_paper` arm asks for n_calibration = 10000 and n_val = 10000, so
the entire MAJOR grid trained on **30000 images**, while the matched
external baselines it is tabled against trained on **45000** — their
own run records read `dataset_sizes.train: 45000`. Two documents state
"45000 training images" for that comparison. The statement is false on
our side of it, and the baselines have had a 1.5x data advantage
throughout. The direction of the layer2 result is unaffected (MAJOR
wins there by 11.5 points), but "matched budget" was not true and this
wave makes it true.

**The budget.** Every MAJOR result so far is at 200 epochs. 800 is the
budget the project intends for the main run.

## Arms

    paper_composition   Eq. (25) complete, the paper's truncated
                        estimator at tau = 1e-3   = MAJOR

CIFAR-10 and CIFAR-100, 3 disjoint seeds each — 6 units.

Changed from `gate_paper` / `gate_paper_c100`, and nothing else:

    n_calibration  10000 -> 2500      so training is 45000 images
    n_val          10000 -> 2500
    max_epochs       200 -> 800
    checkpoint_milestones           [20, 200, 800]

Every method-side value is untouched: estimator, tau, alpha, beta,
gamma, leaf reward, no EMA, no stopped-gradient endpoint, gradients
through the Gram transforms, the K-scaled divergence guard.

Checkpoints at 20, 200 and 800 are kept beside `last.ckpt` and do not
affect selection. They exist so the representation can be read at three
points on the trajectory, which is not recoverable afterwards.

## Declared costs

- The Stage-B calibration sample drops from 10000 roots to 2500.
  Certificates from this wave are less well conditioned than the
  10000-root ones and are **not comparable to them at face value**. The
  CIFAR-10 certificates already on disk read retained_ranks
  [128,128,128] at 10000 roots; whether that survives a 2500-root
  calibration is an open question this wave answers rather than assumes.
- 800 epochs at 45000 images is roughly 6x the compute of the existing
  200-epoch, 30000-image units, which had a median wall of 5.4h. The
  estimate is ~32h per unit, ~195 GPU-hours for the wave.
- Every GPU partition here caps at 24 hours, so units will be killed and
  resumed. That is the design, not a failure.

## Predictions

P-M1. The 800-epoch, 45000-image arm beats its own 200-epoch,
      30000-image counterpart at layer4 on CIFAR-10, three-seed ranges
      disjoint against [82.43, 82.84]. Both a 4x epoch budget and a 1.5x
      data increase point the same way and it would be strange if
      neither moved the number.
      REFUTED if the ranges overlap or the new arm is lower.

P-M2. On CIFAR-100 the same, against [44.66, 45.91].

P-M3. No prediction on whether the MAJOR-versus-baseline gaps change.
      The baselines are at 200 epochs and 45000 images; this arm will be
      at 800 and 45000, so any comparison after this wave is
      epoch-mismatched in OUR favour and must be labelled that way or
      redrawn at a matched epoch budget. Recorded now so it cannot be
      quietly presented as like-for-like later.

P-M4. No prediction on retained ranks under the 2500-root calibration.
      Measured and reported; if they fall below 128 the wave's
      certificates carry that caveat.

## Interpretation grid, committed now

    P-M1 and P-M2 hold
        The main run supersedes the 200-epoch grid as MAJOR's headline
        number, and the ablation ladder underneath it is then at the
        wrong budget and must be re-run before any ablation contrast is
        quoted alongside it.

    either refuted
        MAJOR does not benefit from the longer schedule or the extra
        data on that dataset. Reported as such. It does NOT license
        re-running with a different learning rate or schedule to find a
        number that does improve; this prereg forbids that search.

## What is NOT in this wave

The ablation ladder — beta0, lambda0, alpha0, T2, endpoint_only — stays
at 200 epochs and 30000 images. Until it is re-run, no ablation
contrast may be quoted against this wave's numbers: P1's one-point
composition attribution rests on full-versus-beta0 at a budget this
wave leaves behind.

The Gram-corrected arm (ridge + correction) is abandoned as a wrong
implementation and is not part of MAJOR.

## Gates

- Probe before fleet, per dataset, on its own seed 1. Seeds 2 and 3 only
  after seed 1 reaches status complete.
- The runner resumes from `last.ckpt` with a finiteness and
  scheduler-compatibility guard; the submitter's retry cap counts only
  attempts that made no progress.
- Loud failures; two-commit; disjoint-seed evidence standard.
