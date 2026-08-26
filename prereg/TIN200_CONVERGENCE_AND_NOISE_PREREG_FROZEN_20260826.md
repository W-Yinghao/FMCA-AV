# Frozen before compute — two predictions, 2026-08-26

## A. tin200 convergence (extends the T0 mechanism to a third dataset)

The enlarged-budget T0 study established: hierarchical-arm ratios fall
~1/N toward a nonzero limit (noise-floor from above), flat arms
converge immediately, and the arm separation WIDENS with N.

Frozen predictions for the tin200 convergence sweeps (3 arms, seed-1
checkpoints, --mode convergence --gram-corrected, N doubling as far as
the pooled calibration allows):

A1. flat's ratio moves by less than 0.02 total across all doublings.
A2. V7 and additive ratios FALL monotonically from N=1250 onward, with
    per-doubling drifts shrinking (each drift below its predecessor).
A3. At the largest common N, the ordering V7/additive < flat holds
    with a gap LARGER than at N=2500 (the small-budget gate value).

Refutation: any hierarchical ratio rising between successive N by more
than 0.01, or the gap at max N shrinking below the N=2500 gap.

## B. i.i.d. additive pixel noise on a second encoder (post-hoc made testable)

On resnet50, gaussian_noise and shot_noise (i.i.d. additive/Poisson
pixel noise) gave the two tightest, lowest severity-ladder defects of
nine types; speckle (multiplicative) and impulse (sparse) drifted up;
the FAMILY-level prediction was refuted.  The narrowed hypothesis is
frozen here for an INDEPENDENT encoder before any resnet18 corruption
job runs:

B1. On resnet18 (IMAGENET1K_V1, penultimate, protocol otherwise
    identical), the three-seed ranges of gaussian_noise and shot_noise
    both land strictly below the ranges of defocus_blur and
    motion_blur.
B2. No prediction for speckle_noise: it is measured as the boundary
    case and reported either way.

Refutation: any overlap between {gaussian, shot} and {defocus, motion}
ranges on resnet18.

## C. severity-ladder refinement (exploratory, no directional freeze)

gaussian_noise and defocus_blur re-measured at severities 1,2,3,4,5
(five interfaces instead of three).  The finite-sample L-scan predicts
the measured defect rises with interface count while the shuffled null
rises in proportion; recorded as exploratory context for the
corruption estimand, not as a confirmatory test.
