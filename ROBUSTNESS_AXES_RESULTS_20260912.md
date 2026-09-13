# The two robustness axes, T=4 and K=256 — CIFAR-10, 2026-09-12

**Provenance disclosure, stated first because it limits everything
below.** No pre-registration covers these arms.  `aa98113` introduced
them with a rationale and no frozen prediction, and their three-seed
aggregates were seen while checking which units had completed, BEFORE
any reading rule was written.  They are therefore reported
DESCRIPTIVELY.  Nothing here is a preregistered test, nothing here may
be quoted as a confirmed or refuted prediction, and the appropriate
follow-up for any claim below is a frozen prereg and a fresh wave.

## What was measured

    arm                probe    eff_rank   top share      trace
    MAJOR composition  82.84      78.32      0.0905        69.7   (reference, seed2)

    T4   seed1         75.71      30.33      0.1309       120.8
    T4   seed2         77.05      31.74      0.1393       127.9
    T4   seed3         76.67      29.59      0.1356       107.6

    K256 seed1         78.80       4.09      0.7451       291.4
    K256 seed2         78.42       1.23      0.9744      2367.4
    K256 seed3         76.91       1.01      0.9987     70648.0

eff_rank is the entropy effective rank of the encoder feature
covariance, out of 128 (out of 256 for K256).  trace is that
covariance's trace.

## T=4: consistent, and consistently low-rank

Three seeds agree closely -- probe 75.7-77.1, effective rank 29.6-31.7,
trace 108-128.  The arm trains stably and reproducibly.  It sits about
six points below the composed arm at K=128 and uses roughly a quarter
of the available dimensions.

Its frozen-coordinate certificate reads retained_ranks [40, 101, 98,
20]: the two interior levels are healthy and the endpoint level is at
20 of 128.

## K=256: spectral collapse with scale runaway

Not a robustness result in the ordinary sense.  Across seeds the
effective rank falls to 4.09, 1.23 and 1.01 of 256 while the covariance
trace rises to 291, 2367 and 70648 -- against the reference arm's 69.7.
The third seed puts 99.87% of its variance in one direction at a trace
a thousand times the reference.

The probe still decodes at 76.9-78.8 because it standardizes per
dimension, which is the same reason `paper_endpoint_only` decoded at
73% with effective rank 1.04.  Probe accuracy does not detect this
failure and must not be used to argue the arm is fine.

## The divergence guard did not fire, and could not have

The guard bounds the OBJECTIVE's trace terms, which are computed on
whitened operators and are therefore scale-invariant by construction;
at K=256 those bounds are the K-scaled ones and were never approached.
Nothing in the guard watches the raw representation scale, so an
encoder is free to inflate its output covariance by three orders of
magnitude without tripping anything.  That is a coverage gap in the
guard, not a bug in these runs.

`"collapsed": false` is likewise true for all six units, because that
field is `test_accuracy < 0.15`.

## Against the stated reason for choosing these axes

`aa98113` chose K=256 on the grounds that "every trained arm converges
to retained_min = 128 of 128 and every frozen-coordinate certificate
reports retained ranks [128,128,128], so the operator dimension is
saturated".

That premise does not hold.  Across the CIFAR-10 MAJOR grid only
`composition` and `beta0` reach 128; T2 sits at 15-18, alpha0 at 20-21,
lambda0 at 45-65, endpoint_only at 30-47.  On CIFAR-100 even
`composition` reads [90, 106, 19].  And K=256 itself retains 21 at the
binding level, so raising K does not raise the retained dimension.

The honest summary is that the saturation premise was true of two arms
on one dataset and was generalised.  What a properly preregistered K
sweep should ask is no longer "does K bind" but "why does the endpoint
level cap near 20 regardless of K".

## What may and may not be said

MAY: both axes were run at three seeds; T=4 trains stably at roughly a
quarter rank and six points below the reference; K=256 shows spectral
collapse with scale runaway on every seed.

CORRECTION 2026-09-13.  That last clause originally read "K=256 does
not train to a usable representation on any seed".  Too strong: K256's
probe accuracy rises monotonically with depth, 47.9 to 78.6 across nine
taps, so the representation is functional.  The anisotropy and the
scale runaway are unchanged and remain the finding.  See
`EFFECTIVE_RANK_IS_NOT_A_COLLAPSE_TEST_20260913.md`.

MAY NOT: that either result confirms or refutes anything, that K=256
"shows K binds", or that the endpoint-level cap near 20 has been
explained.  None of that was preregistered and the numbers were seen
first.
