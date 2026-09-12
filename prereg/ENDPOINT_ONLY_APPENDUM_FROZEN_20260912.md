# Appendum — how the endpoint_only arm is read, frozen 2026-09-12

**Disclosure: written after all three CIFAR-10 endpoint_only units
reached status complete, and BEFORE any of their probe accuracies,
spectra or certificates were viewed.  What has been seen is the unit
status field and nothing else.**

## Why this appendum exists at all

`paper_endpoint_only` was added to the grid by commit 8ddccd5 with a
rationale but with no entry in
`PAPER_ESTIMATOR_WAVE_PREREG_FROZEN_20260910`, which names five arms
and does not include it.  It is therefore an unpreregistered arm whose
aggregate is now available, which is exactly the case that has to be
frozen before it is read rather than after.

## What the arm is

Eq. (25) with alpha = beta = lambda_mv = 0, which by the paper's own
text reduces to the endpoint objective plus the covariance
regulariser.  It is the truncated counterpart of the ridge corpus's
flat arm: same objective family, the paper's estimator.

## The contrast it settles

Under ridge the flat arms beat the composed one, consistently and by a
lot -- final_mview 88.97 against product_endpoint 84.98 on CIFAR-10,
and 61.5 against 55.9 on CIFAR-100.  Whether that ordering is a
property of the OBJECTIVE or of the ESTIMATOR could not be asked,
because no flat arm had ever run under truncation.  This arm asks it.

## Prediction, committed before looking

P-E1.  endpoint_only BEATS paper_composition at layer4, three-seed
ranges disjoint.  Stated in this direction deliberately: the flat
advantage is large and replicated across two datasets under ridge, and
the honest prior is that it survives a change of estimator.  If it
does, the composition constraint costs endpoint accuracy under the
paper's own estimator, and that is a negative result about the method
which will be reported as such and not re-framed.

REFUTED if the ranges overlap or composition wins.

## Interpretation grid, committed now

    endpoint_only > composition, disjoint
        The ridge corpus's flat advantage is estimator-independent.
        MAJOR's composition constraint is an accuracy COST at the
        endpoint, roughly the size the ridge corpus already showed.
        P1's one-point interior gain then has to be stated next to
        this, as a trade and not as a free improvement.

    ranges overlap
        The flat advantage shrinks to nothing under truncation.  The
        constraint is then free at the endpoint, and the ridge corpus's
        4-6 point flat advantage was an artifact of ridge coordinates
        -- the same failure mode already found for the endpoint-rank
        "cost", which reversed sign under this estimator.

    composition > endpoint_only, disjoint
        The ordering REVERSES under the paper's estimator.  The
        strongest available result for the method, and precisely
        because it is the strongest it gets the heaviest caveat: one
        dataset, one estimator, n=3, and a ridge corpus that says the
        opposite.  It would need the CIFAR-100 replication before it
        is stated anywhere outside a results file.

## Reading rules, fixed now

- Probe accuracy on encoder features before the projection heads, same
  convex probe as every other arm.
- Endpoint effective rank is reported ALONGSIDE accuracy, always.  A
  flat arm that wins on accuracy while collapsing the endpoint is the
  failure mode the paper names, and alpha0 and T2 have already shown
  this grid can produce it.
- The certificate is not evidence here: an arm with no composition term
  has no closure to satisfy, so a low closure error from it means
  nothing about the method.
