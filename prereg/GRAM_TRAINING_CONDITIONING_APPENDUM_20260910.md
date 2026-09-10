# Appendum, frozen before the type-2 fleet — 2026-09-10

Amends SSL_FIVE_QUESTION_PREREG_FROZEN_20260910 for the Gram-in-training
arm only.  **Disclosure: this is written AFTER seeing two failed QC
probes of that arm and a diagnostic run, and BEFORE any type-2 fleet
or aggregate.**  Nothing here touches Q1, Q2, Q3 or Q5.

## What the probes showed

Probe 1 died in 6 seconds: `gram.py` built its identity matrices on the
CPU because until now it was only ever called from the analysis path.
Fixed and pinned by tests; not a scientific finding.

Probe 2 cleared that and tripped the divergence guard at the end of
epoch 1 (train/loss 1.253e3 against a bound of 700).

## What the diagnostic showed, including where I was wrong

My first hypothesis was that the corrected composition amplifies
near-degenerate directions.  Measured on the TRAINED V7 checkpoint it
was refuted: minimum Gram eigenvalue 0.22-0.31, amplification 3.3-4.5x,
truncation inert at every tau up to 0.1, and the corrected closure
ratio essentially equal to the surrogate one (inflation x0.90-1.02).

My second hypothesis -- that the corrected term enters at a different
weight -- was refuted by the same numbers.

Re-measuring at the state the probe actually dies in, namely random
initialisation, reinstated the first hypothesis:

    state            min eig (level 1)   G^-1 amplification   closure inflation
    initialisation   0.029 - 0.035       28x - 34x            x3.9 - x4.2
    converged V7     0.22 - 0.31         3.3x - 4.5x          x0.90 - x1.02

The diagnostic on the trained checkpoint was asking about the wrong
state.  Recorded because the same mistake is easy to repeat.

## The rule, chosen by criterion rather than by outcome

`loss.gram_tau` is a spectral floor on the training-time inverses:
directions with Gram eigenvalue below `tau * lambda_max` are dropped
rather than inverted.

Criterion, fixed before choosing the number: **cap early-training
amplification at the level the converged model exhibits naturally
(~4x)**.  From the measured sweep that is tau = 0.1 (amplification
4.7-6.0x at initialisation, ~100 of 128 directions retained), and it
retains 128/128 on the trained checkpoint, so the floor is inert once
the representation is healthy and cannot distort the converged
solution.  tau = 0.05 was rejected as too weak (10.9-12.5x) and
tau = 0.2 as more truncation than the criterion asks for.

The number was NOT selected by looking at probe accuracy or closure
outcomes, and it will not be re-selected if the arm underperforms.

## Consequences for P4.1

P4.1 is unchanged.  Two additional rules:

G1. If the arm still trips the divergence guard at tau = 0.1, the
    result is reported as "the Gram-corrected closure is not trainable
    under this recipe", NOT patched by raising the guard or lowering
    beta.  A guard raise would make the arm untestable rather than
    successful.
G2. The retained-direction count is logged every step
    (`gram_retained_min`).  If it stays below 128 at convergence, the
    corrected arm optimised a truncated geometry and P4.1 is reported
    with that caveat attached, not without it.
