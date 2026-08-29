# Reading: a rejected mechanism and a reinstated finding

## 1. The Gram mechanism is rejected as the cause of the T0 failures

The directive named it a "theoretically grounded candidate, not
demonstrated causally", and made H-G1 the test.  H-G1 fails: the
correction moves the defect by 0.1-1.3% and leaves the convergence
drift untouched.  So the theorem stands -- the uncorrected product
really is a regularized interface surrogate, and the paper must say so
-- but its empirical bite in this pipeline, at ridge 1e-3, is small.

The practical consequences run in our favour:

- **The corrected training loss (directive §5) is no longer urgent.**
  If the metric artifact is 1% of the defect, training against the
  surrogate and evaluating corrected costs almost nothing, which is the
  Path-2 fallback arriving on empirical rather than stability grounds.
  The minimal decisive triangle in the strategy's §10 can therefore be
  built on the existing training code.
- **The old runs are relabelled, not invalidated.**  Their numbers move
  by ~1% under correction, so the accuracy, layer-wise probe and
  effective-rank phenomena survive verbatim.
- **The T0 failures need a different explanation.**  Whatever makes the
  hierarchical arms converge slowly, it is not the coordinate metric.
  The one lead the data offers is structural: c10 V7's root Gram
  retains 21 of 128 directions, and it is also the worst-converging
  arm.  A near-degenerate root subspace estimated from a few thousand
  samples is a plausible next candidate -- and, unlike the Gram
  mechanism, it is arm-specific, which matches the observation that
  flat arms converge fine while hierarchical arms do not.

## 2. Withdrawn on 08-24, reinstated now

`DEFECT_DECOMPOSITION_READING_20260824.md` withdrew the claim that the
instrument separates closure-trained arms from untrained ones, because
at n_calibration = 2500 the flat arm had the *smallest* numerator on
CIFAR-10 (1.955 against V7's 2.078).  At n_calibration = 10000 that
reverses cleanly: V7's numerator is 1.024, flat's is 1.870.

The withdrawal was correct on the evidence available; the evidence was
small-sample.  The reinstated claim is narrower than the original and
now carries its conditions:

> At a calibration budget of 10000, and reported as
> (numerator, denominator, N), the composed-training arm attains a
> strictly smaller closure numerator than both the flat and the
> additive arm on CIFAR-10, and beats additive on both terms at
> matched denominator on CIFAR-100 with disjoint three-seed ranges.

What still does **not** hold, and stays suspended: absolute defect
values (the arms are not converged even at N=10000), any cross-arm
ratio comparison, and the CIFAR-10 V7-versus-additive separation, whose
seed ranges still overlap.

## 3. Where this leaves the T0 gate

Criterion 1, convergence: **still failing** (0.044-0.117 per doubling
at N=10000 against < 0.005 required).  Criterion 2, degenerate
separation: **not yet evaluated at the enlarged budget** -- the
random-encoder control has only been run at the old one.  Both are now
cheap and are queued.

The honest position for the manuscript: the multi-component readout the
strategy's §3.7 prescribes -- endpoint spectrum and mass, path spectrum
and mass, Δ operator norm, ranks, conditioning, N -- is reportable
today.  A single scalar cross-model instrument is not, and the T0
verdict on that stands.
