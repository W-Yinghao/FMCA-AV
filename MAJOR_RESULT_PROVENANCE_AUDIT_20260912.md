# Which standing results are MAJOR's — provenance audit, 2026-09-12

Purpose: establish, per claim, whether the project's current main results
are evidence about MAJOR (the method, under the paper's truncated
estimator) or about the superseded ridge corpus.  Written because
`PAPER_ESTIMATOR_WAVE_PREREG_FROZEN_20260910` says the V1-V7 corpus
"is not evidence about the paper's method" and "will be labelled that
way wherever it is used" -- this is that labelling, applied to
everything that currently stands.

Nothing here re-reads or re-adjudicates a result.  It sorts existing
results by which estimator produced them.

## The discriminator, verified in code

Estimator selection is not a config knob.  It is the variant prefix:

    hierarchy_module.py:629   mode = "truncated" if self.variant.startswith("paper_") else "ridge"
    hierarchy_module.py:161   paper_ arms additionally assert no EMA, no
                              stopped-gradient endpoint, gradients through
                              the Gram transforms
    objective.py:128          whiten_chain_batch(mode="ridge") is the DEFAULT

So any call site that does not pass `mode=` computes under ridge, and
`spectral_tau` is set only in `configs/gate_paper` and
`configs/gate_paper_c100`.  The eight MAJOR variants are
paper_composition, paper_beta0, paper_lambda0, paper_alpha0, paper_T2,
paper_endpoint_only, paper_T4, paper_K256.  Every other gate variant
(final_2view, final_mview, additive_2view, additive_mview, amdim_cross,
product_only, product_endpoint) is ridge.

Certificates have their own split:

    run_paper_certificate.py        truncated_whitener, stamps
                                    "estimator": "truncated"   <- MAJOR
    run_gate1_unit.certificate_evaluation
                                    fit_level_coordinates(ridge), no
                                    truncated branch at all
    chainspec.py                    zero occurrences of "truncated"
    plugin_module.py:100            whiten_chain_batch(...) with no mode=

The consequence worth stating plainly: a certificate quantity is MAJOR
only if it came out of `run_paper_certificate.py`.  Certificates
computed by any other path are ridge even when the model being measured
is a MAJOR arm.

## Inventory

    estimator       complete units    result roots
    MAJOR/truncated            26     4
    ridge                     286    46

No result root mixes the two -- the separation is clean, and that is
worth keeping.  MAJOR certificates: 12 files, all stamped
`estimator: truncated`, CIFAR-10 only (composition 1-3, beta0 1-3,
alpha0 1-2, T2 1-2, lambda0 1 and 3).

MAJOR unit coverage: CIFAR-10 14/18 complete (3 endpoint_only in
flight, lambda0 seed2 failed); CIFAR-100 12/18 complete (3
endpoint_only and alpha0 seed2 in flight, lambda0 seeds 1-2 failed).

## Claim-by-claim

MAJOR = evidence about the method.  INDEP = measurement that does not
involve a whitener, so it is valid under either estimator, BUT is only
MAJOR evidence if the model it measured is a MAJOR arm.  RIDGE = not
evidence about the method.

### Backed by MAJOR today

    P1 composition attribution (full > beta0 at layer2, disjoint
       three-seed ranges, ~1 point)            MAJOR   CIFAR-10
    P2 full > T2 at layer2 (~10 points); the
       predicted T2 > full at layer4 REFUTED   MAJOR   CIFAR-10
    P3 relative closure error full < beta0,
       disjoint, at matched endpoint norms     MAJOR   CIFAR-10
    lambda0 instability / guard trips          MAJOR   both datasets
    alpha0, T2 low closure error as the
       endpoint-vanishing failure mode         MAJOR   CIFAR-10
    Like-for-like vs external baselines:
       layer2 +11.5 disjoint; endpoint
       competitive only (0.08 points)          MAJOR + INDEP baselines

The external baselines (vicreg, simclr, barlow_twins, moco_v2, byol)
are other methods and carry no estimator of ours.  They are legitimate
comparators for MAJOR arms, which is what the like-for-like table does.

### Ridge-only — not evidence about MAJOR

Every item below is currently presented as a main result somewhere in
the standing docs, and none of it is MAJOR evidence.

    V7 optimization arc 48.3 -> 85.4                     RIDGE
    v8 final gate ladder: flat anchor 89.0 /
      flat-2view 85.3 / V7 85.0 / additive 78.3 /
      product_only 57.0                                  RIDGE
    E2 +-1pt non-inferiority vs the 89.0 anchor          RIDGE
    Calibration 2500->5000, defect drop ~0.04 as the
      empirical epsilon_n noise floor                    RIDGE
    Explicit (V7 0.209) vs implicit (additive 0.199)
      closure not separating                             RIDGE
    Instrument separates trained-for-closure (~0.20)
      from not-trained (flat 0.40-0.43)                  RIDGE
    Channel spectroscopy / per-edge localization         RIDGE
    Probing-depth retention profile                      RIDGE
    Plug-in study (beta=32 lowers defect on Barlow
      and VICReg; beta=16 correction)                    RIDGE
    CIFAR-100 defect ordering V7 < additive < flat       RIDGE
    Star-tree negative control defect separation         RIDGE
    alpha=0 (83.4) and M=4 (82.1) ablations              RIDGE
    Layerwise inward redistribution, 2x2 decomposition   RIDGE
    Chain track: depth r50/r101/r152 and self-stitch     RIDGE

The last one carries a live consequence.  Commit aa98113 excluded
backbone depth as a robustness axis on the grounds that "the chain
track already measured r50/r101/r152 ... and found their stage profiles
indistinguishable, so depth is a verified null variable for this
geometry."  `chainspec.py` contains no truncated path, so that
verification is a ridge measurement.  The decision may well be right --
stage-profile invariance to depth is plausibly estimator-robust -- but
it is not currently supported by MAJOR evidence, and the commit message
should not be read as if it were.

### One outright contradiction, already adjudicated

The ridge corpus reported that composition costs endpoint capacity:
endpoint effective rank V7 56.7 against flat 98.3, read as "closure
confines the endpoint to the subspace the path can build".  Under MAJOR
the ordering REVERSES -- full has the highest endpoint effective rank
of any arm (78.3-79.7), above beta0 (73.8-74.9).
`PAPER_ESTIMATOR_RESULTS_20260911` states the conclusion: the cost was
an artifact of the ridge estimator, not of the composition constraint.

This is the clearest demonstration that the ridge corpus cannot be
carried over as method evidence.  It is not merely weaker evidence for
the same conclusion; on this point it supports the opposite one.

## Gaps — MAJOR main results that do not exist yet

    1. No MAJOR counterpart to the ridge flat anchor.  T2 is the
       paper's own T=2 reduction and is NOT the same control, which is
       why the shallow/endpoint trade did not reproduce.
       paper_endpoint_only is the truncated counterpart and is in
       flight on both datasets.
    2. No MAJOR certificate outside CIFAR-10, and none for
       endpoint_only, T4 or K256.  Only run_paper_certificate.py can
       produce one.
    3. No MAJOR instrument track at all -- spectroscopy, probing depth
       and the plug-in study have no truncated path in code, not just
       no results.  Each would need a truncated branch before it could
       be rerun as MAJOR.
    4. No MAJOR replication of the star-tree negative control, which is
       the load-bearing evidence that nested views (not just the loss
       form) are what lower the defect.
    5. Robustness axes T=4 and K=256 are MAJOR by construction but
       unfinished: T4 probe running, K256 probe submitted 2026-09-12.

## What this means operationally

The set of results that may currently be presented as MAJOR's main
results is exactly the first table, plus whatever the CIFAR-100 wave
and the endpoint_only arm return.  That is a real result set -- a
composition attribution with a disjoint three-seed contrast, a
certificate contrast at matched endpoint norms, a refuted prediction
reported as refuted, and a competitive like-for-like baseline table.
It is narrower than the standing brief, and every item in it is the
paper's own estimator.

The ridge corpus keeps its value as a loss-form ladder and as the
record of how the design was arrived at.  It should be cited that way
and never as evidence for the method.
