# T0 remediation protocol (frozen 2026-08-24, before any run)

The validity gate failed on two counts
(`T0_VALIDITY_GATE_VERDICT_20260824.md`): a never-trained encoder beats
trained V7 on the ratio, and no hierarchical arm is within an order of
magnitude of the convergence tolerance at the ceiling of the clean
held-out budget.  This document freezes the remediation before it runs.

## 1. Diagnosis

Both failures share a root: the held-out budget.  `n_calibration +
n_val = 5000` is too small for Stage-B to converge on the hierarchical
arms, and an unconverged ratio is exactly the quantity a weak encoder
can flatter.  Flat arms converge inside the budget (last-doubling
change 0.006-0.008) which is why only the hierarchical rows fail.

## 2. Two responses, run in parallel

**(a) Extended-sample diagnostic -- no training, CPU, runs first.**
The test split is held out from SSL training just as the calibration
and validation splits are, so Stage-B may draw on it as long as
Stage-C does not evaluate on the same images.  Stage-B is fitted on
`calibration + val + first half of test` (up to 10000) and Stage-C
evaluates on the second half of test (5000).  This buys one further
doubling at zero training cost and answers, before any GPU is spent,
whether N=20000 is even the right target.

Preregistered read: if the last-doubling change at N=10000 is still
above 0.02 for the hierarchical arms, then n_calibration = n_val =
10000 is insufficient too, and the retraining below is re-planned at a
larger budget rather than launched.

**(b) Retraining at a larger held-out budget.**
`n_calibration = 10000`, `n_val = 10000`, leaving 30000 SSL training
images on CIFAR-10 and CIFAR-100.  Permits a convergence sweep to
N=20000, two doublings past today's ceiling.

Arms: `product_endpoint`, `additive_mview`, `final_mview`, three seeds,
both datasets -- the rows that carry instrument claims.  The star
control is NOT retrained yet; it is queued only if the gate passes,
because there is no point spending on a control for a suspended claim.

## 3. Disclosed deviation

The SSL training set shrinks from 45000 to 30000 images, so **accuracy
from these runs is not comparable to the v8 gate** and is not to be
placed in the same table.  These runs exist to make the instrument
interpretable; the accuracy tables stay with the v8 runs.

## 4. Gate criteria (frozen)

The suspension in `T0_VALIDITY_GATE_VERDICT_20260824.md` lifts only if
BOTH hold on the retrained arms:

- **Convergence**: last-doubling change < 0.005 for every arm at the
  new ceiling.
- **Degenerate separation**: the random-encoder row, evaluated under
  the identical enlarged protocol, scores *worse* (higher) than every
  trained arm on whatever statistic is being reported.

If convergence holds but the random encoder still wins on the ratio,
the ratio is abandoned permanently and the reported statistic becomes
the `(numerator, denominator)` pair with an explicit scale floor -- the
random row's denominator is 2.2-2.5 against 4.1-10.3 for trained rows,
so the floor is measurable rather than arbitrary.

## 5. What stays suspended meanwhile

Every absolute defect value, the pooled-calibration comparison, the
CIFAR-100 explicit-versus-additive separation, and the star-control
separation.  Untouched, because none of them used the ratio: the
layer-wise probe and effective-rank profiles, the 2x2 separating loss
form from view nesting, the accuracy tables, and Wave 0.
