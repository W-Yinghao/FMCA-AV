# Frozen before compute — the MAJOR completion wave, 2026-09-12

Frozen BEFORE any unit of this wave runs.  **Disclosure: written after
the CIFAR-10 MAJOR adjudication (`PAPER_ESTIMATOR_RESULTS_20260911`)
and after the provenance audit (`MAJOR_RESULT_PROVENANCE_AUDIT_20260912`).
No arm of THIS wave has produced a number yet -- none of them could,
since until 2026-09-12 the code could not express two of them.**

## Why this wave exists

The audit found that the standing main results are ridge-corpus results
and that MAJOR's own result set is narrower than the project has been
presenting.  Two specific holes:

1. On identical footing -- same probe, 200 epochs, width 64, K=128,
   same nested view tree -- MAJOR's composed arm reaches 82.70 while
   the ridge corpus's composed arm (product_endpoint, the "V7 85.0"
   row) reaches 84.98 and the ridge flat anchor reaches 88.97.  That
   2.28-point gap is CONFOUNDED: the two arms differ in estimator
   (truncated vs ridge) AND in recipe (the EMA operator target, the
   faithful_bootstrap product recipe, the alpha bootstrap).  Only two
   of the four cells existed, so the gap could not be attributed.

2. The star-tree negative control -- the load-bearing evidence that
   NESTED VIEWS, not the loss form, are what lower the closure defect
   -- exists only under ridge.

## Arms (CIFAR-10, 3 disjoint seeds each)

The 2x2 completes an existing partial design.  Cells A and B are
already on disk and are NOT rerun:

    cell  recipe             estimator    status
    A     paper_composition  truncated    HAVE, n=3, 82.70   (= MAJOR)
    B     product_endpoint   ridge        HAVE, n=5, 84.98   (= V8)
    C     product_endpoint   truncated    NEW, configs/gate_x_v8trunc
    D     paper_composition  ridge        NEW, configs/gate_x_paperridge

Plus one control:

    S     paper_composition  truncated, STAR view tree
                                          NEW, configs/gate_x_star

Neither C nor D is MAJOR and neither may ever be reported as MAJOR's
result: C runs a recipe the paper forbids (no matrix EMA), D runs an
estimator the supplement calls a different operator estimate.  They are
attribution instruments for a gap between two corpora.

Cells A and B were trained at different calibration sizes (10000 vs
2500).  n_calibration is matched to the MAJOR gate's 10000 in cell C so
that C differs from A in recipe ALONE.  Calibration size affects the
certificate, not the probe, so the probe-accuracy contrast against B
carries that mismatch and is declared here rather than discovered
later.

## Predictions

P1. The gap is mostly RECIPE.  Cell C beats cell A at layer4 by at
    least 1.0 point, three-seed ranges disjoint.  Rationale fixed in
    advance: the EMA operator target was the single largest step of the
    V7 arc (75.4 -> 82.8), which is a recipe effect of ~7 points under
    ridge, and nothing about truncation should remove it.
    REFUTED if the ranges overlap or C < A.

P2. Additivity is NOT predicted.  (C - A) and (B - D) are both
    estimates of the recipe effect, one at each estimator; they are
    reported side by side and any interaction is reported as such.

P3. The star control separates under MAJOR: S's frozen-coordinate
    relative closure error exceeds cell A's, disjoint over three seeds,
    AT COMPARABLE ENDPOINT NORMS.  The endpoint-norm clause is binding
    -- alpha0 and T2 already showed that a low closure error next to a
    collapsed endpoint is the failure mode the paper names, and a HIGH
    closure error next to a collapsed endpoint is equally uninformative.
    REFUTED if the ranges overlap at matched norms.
    Disclosed in advance: under ridge this contrast was a NULL on
    CIFAR-10 and separated only on CIFAR-100.  The reason to expect
    more here is that MAJOR's instrument already resolved full vs beta0
    on CIFAR-10, which ridge could not.  If S is null too, the honest
    reading is that CIFAR-10 is underpowered for this contrast under
    either estimator, and the CIFAR-100 replication becomes the test --
    it is NOT evidence that the star tree is harmless.

P4. No prediction for whether cell C trips the divergence guard.  A
    recipe tuned against ridge coordinates has never run against
    truncated ones.  Measured and reported either way.

## Interpretation grid, committed now

    C > A and B > D    recipe dominates; MAJOR's deficit is the price of
                       the paper's own method spec, and should be
                       reported as such rather than as an estimator cost
    C ~ A and B > D    the recipe helps only under ridge; it is an
                       artifact of ridge coordinates and the ridge
                       corpus's advantage does not transfer
    C > A and B ~ D    estimator-independent recipe gain; the strongest
                       case that the paper's no-EMA constraint costs
                       real accuracy
    C ~ A and B ~ D    the 2.28 points are neither factor alone;
                       report as unexplained and do not attribute

## Gates

- PROBE BEFORE FLEET, per arm and on its own probe.  Seed 1 of each new
  arm runs alone; seeds 2 and 3 are gated on that unit reaching
  status complete.  Cell C especially: it is the first time any
  ridge-family loss path has run under truncated coordinates, which is
  also the first time those paths can receive an empty retained set.
- Divergence-guard bounds are the K-scaled ones; at K=128 they are the
  same numbers the corpus already ran under.
- Certificates come from run_paper_certificate.py only.  A certificate
  from any other path is a ridge certificate regardless of the arm.
- Probes on encoder features before the projection heads, by the same
  convex probe used for every other arm.
- Loud failures; two-commit; disjoint-seed evidence standard.

## Not in this wave

CIFAR-100 replication of the 2x2 (CIFAR-10 first: both existing cells
are CIFAR-10).  The instrument track -- spectroscopy, probing depth,
plug-in -- stays out because none of it has a truncated code path yet;
that is a code change, not a queued job, and it is not smuggled into a
wave whose other arms are ready.
