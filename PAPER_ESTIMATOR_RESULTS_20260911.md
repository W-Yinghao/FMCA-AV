# The paper estimator on CIFAR-10 — adjudication, 2026-09-11

> **Code-version confound, found 2026-09-15.** The `full` and `beta0`
> units below were trained before commit `02fa55e` (11 Sep 01:52)
> changed the final-stage multiview term; `alpha0` and `T2` were trained
> after it; `lambda0` and `endpoint_only` have no such term. P1 (full vs
> beta0) and the lambda0 comparison are within one implementation. P2
> and P3 (full vs alpha0, full vs T2) compare two implementations and
> are confounded. Under the current term the full arm itself pins its
> endpoint retained rank at 16-21 from epoch 2 (the 45k run, and a
> five-epoch rerun of this very split), so the 82.4-82.8 numbers are
> the pre-02fa55e term's. See `BIFURCATION_DIAGNOSTIC_RESULTS_20260915.md`.

## Naming

The method this paper proposes is **MAJOR**: FMCA-AV's trace objective
with a Gram-normalized composition constraint along the view chain
(section Eq. 25, supplement Eq. 6-10).

Method and estimator are separate things and are named separately,
because conflating them is what produced this wave in the first place:

    MAJOR                the method -- the objective of Eq. (25)
    truncated            the estimator MAJOR is defined on: shared
                         symmetric TRUNCATED inverse roots at relative
                         cutoff tau = 1e-3, so W R W is exactly a
                         projector and the plain ordered product already
                         IS the projected composition
    ridge plain          (R + rho s I)^{-1/2} with the plain product --
                         the supplement calls this a DIFFERENT operator
                         estimate, and it is what the V1-V7 corpus used
    ridge + correction   ridge whitening with G^{-1} inserted afterwards
                         to compensate; our own construction, not the
                         paper's

"The Gram version" is retired as a term.  All three normalize by a
Gram; they differ only in how the inverse root is regularized, and that
ambiguity is exactly what it cost us to discover.

Code identifiers still read `paper_*` (variant names, config dirs,
result roots) because a fleet is running against them; they will be
renamed to `major*` in one sweep once it lands.  MAJOR is the name used
in all prose, tables and figures from here on.


Against PAPER_ESTIMATOR_WAVE_PREREG_FROZEN_20260910.  Five arms, three
disjoint seeds each, trained from scratch under the supplement's
estimator (truncated inverse roots, tau=1e-3, plain ordered product, no
ridge anywhere).  Probes are on encoder features before the projection
heads; certificates follow supplement §1.5.

## Stage-probe accuracy

    arm         layer1          layer2          layer3          layer4        n
    full    [58.06,60.03]  [79.56,80.29]  [82.63,82.88]  [82.36,82.74]   3
    beta0   [57.87,59.18]  [78.45,79.23]  [82.10,83.04]  [82.15,82.57]   3
    alpha0  [57.26,58.73]  [76.63,77.22]  [79.99,80.83]  [79.52,80.08]   3
    lambda0 [58.08,58.53]  [76.10,76.19]  [78.84,79.07]  [78.70,79.31]   2
    T2      [53.74,55.98]  [68.37,70.53]  [69.84,71.88]  [68.50,70.69]   3

## P1 CONFIRMED, and narrowly

full beats beta0 at layer2 with disjoint three-seed ranges
([79.56,80.29] vs [78.45,79.23]).  Every other layer OVERLAPS.

The composition constraint's effect on the representation is real,
localized to the interior tap, and about one point.  That is the whole
of it: at layer3 and layer4 the two arms are indistinguishable, so the
constraint does not buy endpoint accuracy.  A one-point interior gain
is the honest size of this claim and should be written that way.

## P2 HALF REFUTED

Confirmed: full beats T2 at layer2 by ~10 points.
REFUTED: T2 was predicted to beat full at layer4; full beats it there
too, by ~12 points.

The paper's T=2 reduction is not a strong flat baseline under this
estimator.  With one edge the composed and direct operators are two
estimates of the SAME operator, so the closure term contributes only
estimation noise (its trained closure is ~0.0007 and its frozen-
coordinate relative error 0.006), and the arm ends with a narrow
representation (endpoint effective rank 20-25 against full's 78-80).
The shallow/endpoint trade seen in the ridge corpus does NOT reproduce
here, and the reason is that T2 is a different control than the ridge
corpus's flat arm.  Reported as a refutation, not re-framed.

## P3 CONFIRMED

Frozen-coordinate relative closure error, full [0.1602,0.1689] versus
beta0 [0.1733,0.1818] -- disjoint -- at endpoint norms that match
(9.08-9.29 versus 9.07-9.34) and full retained rank on both.  The
constraint lowers the closure error without shrinking the endpoint,
which is the failure mode the paper itself names ("zero closure also
permits both endpoint matrices to vanish").

The low closure errors of alpha0 (0.087-0.092) and T2 (0.006-0.008) are
exactly that failure mode and must never be quoted as successes: their
endpoint norms are 4.0 against full's 9.1, and their retained ranks are
17-21 against 128.

## The pre-disclosed cost did NOT appear

The prereg disclosed in advance that the ridge corpus's composed arm
carried a narrower endpoint (effective rank 56.7 against flat's 98.3)
and said this would be reported beside any gain.  Under the paper's
estimator the ordering reverses: full has the HIGHEST endpoint
effective rank of every arm (78.3-79.7), above beta0 (73.8-74.9).  The
cost was an artifact of the ridge estimator, not of the composition
constraint.

## lambda0: n=2, and the instability is the finding

Seed 2 tripped the divergence guard at epoch 193 (edge_trace_sum 413.9
against 400) and was NOT rescued.  Across seeds this arm swings
edge_trace_sum 115 to 388, and one certificate shows retained ranks
[88, 1, 2] with endpoint norm 0.099 -- a near-collapse.  Removing the
multiview aggregation term destabilises training and admits collapse.
The prereg gave lambda0 no prediction; this is the report.

CORRECTION 2026-09-12.  This section first said "the config is
deterministic, so a rerun would reproduce it exactly."  That is false
and is withdrawn.  A rerun of this exact unit exists -- same config,
identical hparams.yaml checksum, same seed -- and it is bit-identical
to the first on none of their 55 overlapping epochs:

    epoch    attempt 1    attempt 2
        0      31.2114      31.0043
       54      90.5777      82.0685

The trainer is built with deterministic="warn", which warns on a
non-deterministic kernel rather than refusing it, while the config
records deterministic: true; and the launcher submits to
--partition=A100,H100,L40S, so a unit and its rerun need not even land
on the same architecture.  These two did not: attempt 1 ran on an H100
NVL, attempt 2 on an A100-PCIE-40GB.  Attempt 2 was cancelled at epoch
54 -- while running about 9% BELOW attempt 1 at the same epoch -- so
whether it would have tripped the guard at all is unknown.

What survives unchanged: the instability itself, which rests on the
across-seed spread rather than on any single trajectory, and which the
CIFAR-100 replication independently reproduces (two further guard trips
at 417.7 and 400.2).  What does NOT survive: any reading in which this
arm's n=2 is a fixed property of the arm.  It is one outcome of a
fleet that is not reproducible unit-by-unit, and the guard bound of
400.0 is a hard-coded operational sentinel, not a preregistered
threshold -- all three trips land within 4.5% of it.

## endpoint_only (added 2026-09-12): P-E1 refuted, on a degenerate arm

Against `ENDPOINT_ONLY_APPENDUM_FROZEN_20260912`, frozen before these
numbers were viewed.

    arm             n        layer4 range      mean    endpoint eff.rank
    composition     3   [82.43, 82.84]        82.70    [78.3, 79.7]
    beta0           3   [82.08, 82.70]        82.31    [73.8, 74.9]
    endpoint_only   3   [73.01, 74.30]        73.45    [1.03, 1.09]
    T2              3   [67.56, 69.83]        68.93    [20.0, 24.9]

P-E1 predicted endpoint_only would BEAT composition, because the ridge
corpus's flat advantage is large and replicated on two datasets.  It
does not: composition wins by 9.25 points with disjoint ranges.  The
prediction is refuted.

CORRECTION 2026-09-13.  The paragraph below argues from effective rank
that endpoint_only is not a usable comparator.  That is too strong: its
probe accuracy rises monotonically across all nine block taps, 46.5 to
77.0, so it is functional.  The conditioning claim stands, the
"degenerate comparator" claim does not, and the 9.25-point gap is
between two working arms differing in conditioning.  See
`EFFECTIVE_RANK_IS_NOT_A_COLLAPSE_TEST_20260913.md`.

**The refutation does not license the strong reading.**  The frozen
grid's third cell says a reversal would be "the strongest available
result for the method".  This is not that, because the comparator is
degenerate:

    seed   probe    eff_rank   top eigenvalue share    trace
    1      74.30      1.04            0.9964          855.1
    2      73.01      1.03            0.9968          968.6
    3      73.03      1.09            0.9907          336.8

All three seeds put over 99% of the feature variance in a SINGLE
direction and inflate the trace by 5-13x against composition's 71.5.
The arm decodes at 73% anyway -- the probe standardizes per dimension,
so the surviving fraction of a percent of variance still carries class
information -- but this is not a healthy flat baseline losing to a
composed one.  It is the endpoint objective, under the paper's
estimator, failing to train a well-conditioned representation.

So what this arm settles is narrower than what it was added for.  It
does NOT make the "best paper-grid accuracy vs best ridge accuracy"
comparison estimator-controlled, because the ridge flat arm
(final_mview, 88.97) is not degenerate and this one is.  What it does
show is consistent with the paper's own warning that closure alone
"permits both endpoint matrices to vanish": remove alpha, beta and
lambda_mv together and the endpoint does degenerate, under truncation,
on every seed.

A like-for-like estimator-controlled flat comparison still does not
exist.

## QC: the `collapsed` field does not mean what it says

Every unit record carries `"collapsed": false` including these three.
The field is `test_accuracy < 0.15` (run_gate1_unit.py:452) -- a probe
floor, not a spectral test.  An arm with effective rank 1.04 of 128 and
99.6% of its variance in one direction passes it.  BYOL was excluded
from competitiveness claims as "a collapsed run" by judgement, not by
this flag, which also did not fire on it -- and that judgement has since
been traced to a missing normalization layer on our side rather than to
BYOL.

No result already reported changes because of this; the field is not
used as a gate anywhere.  It is recorded because `collapsed: false`
reads like an assurance of exactly the property these three arms lack.

## Standing caveats

Single dataset.  The external-baseline comparison is carried over from
the matched-budget wave and is estimator-independent, but those arms
were compared against the RIDGE corpus; a like-for-like table against
these arms is not yet drawn.  BYOL remains excluded, but as OUR failed
run rather than as a collapsed method -- see the correction above and
`BYOL_COLLAPSE_DIAGNOSIS_20260913.md`.

---

# Like-for-like against the matched-budget baselines

Added after the adjudication above; the arms and the baselines are now
compared on the same footing.  Same backbone class (CIFARResNet, width
64), 45000 training images, 200 epochs, batch 256, and the SAME convex
probe run by the same code on encoder features before the projection
heads.  BYOL is excluded: it collapsed (validation score 0.9965, probe
declining with depth), and the prereg forbade re-tuning a baseline's
recipe, so it is reported as not successfully trained rather than beaten.

CORRECTION 2026-09-13.  The attribution in that sentence is withdrawn.
The collapse traces to our own implementation: the shared `MLP` behind
every projector and predictor carries no normalization layer, and
BYOL's projector and predictor are specified with BatchNorm, which --
given BYOL has no negatives and no variance term -- is its entire
collapse avoidance.  The other four baselines each carry an
anti-collapse term in the loss and were unaffected.  Supplying the
normalization the method specifies is not the re-tuning the prereg
forbade; it is implementing the method.  Until a BN'd BYOL has run, no
claim about BYOL in either direction is supported by this row.  See
`BYOL_COLLAPSE_DIAGNOSIS_20260913.md`.

    method            layer1          layer2          layer3          layer4      n
    OURS full     [58.06,60.03]  [79.56,80.29]  [82.63,82.88]  [82.36,82.74]  3
    OURS beta0    [57.87,59.18]  [78.45,79.23]  [82.10,83.04]  [82.15,82.57]  3
    OURS alpha0   [57.26,58.73]  [76.63,77.22]  [79.99,80.83]  [79.52,80.08]  3
    OURS lambda0  [58.08,58.53]  [76.10,76.19]  [78.84,79.07]  [78.70,79.31]  2
    OURS T2       [53.74,55.98]  [68.37,70.53]  [69.84,71.88]  [68.50,70.69]  3
    vicreg        [56.02,56.33]  [67.21,68.06]  [78.16,78.54]  [81.44,81.66]  3
    simclr        [56.12,56.96]  [67.14,67.52]  [77.70,77.95]  [81.90,82.28]  3
    barlow_twins  [54.76,55.52]  [66.16,67.33]  [76.98,77.30]  [79.84,80.81]  3
    moco_v2       [54.64,55.48]  [65.52,65.70]  [75.14,75.51]  [78.65,79.30]  3
    byol (failed) [43.83,44.20]  [44.28,45.81]  [41.47,43.43]  [36.55,38.64]  3

## What may be claimed, and at what strength

SHALLOW LAYER, strong.  At layer2 the full arm's three-seed range is
disjoint from every baseline's, by 11.5 points.  This is the result.

ENDPOINT, competitive only.  At layer4 the ranges are also disjoint --
82.36 against the best baseline's 82.28 -- but by 0.08 points, which one
more seed could erase.  This must be written as "competitive at the
endpoint", which is what the frozen P1.1 asked for (within 3.0 points,
comfortably met), NOT as "best at the endpoint".  A 0.08-point margin
is not a ranking.

## The division of labour this table does NOT settle

An external baseline carries no training signal at intermediate layers,
so part of the 11.5-point layer2 margin is the trivial fact that we
train there and they do not.  The mechanism claim rests on full versus
beta0 -- both trained at every tap, differing only in beta -- which is
one point, disjoint, and localized to layer2.

So the two contrasts do different jobs and must be written separately:
the baselines establish that the method is practically competitive and
markedly better shallow; beta0 establishes that the composition
constraint is what produces the interior gain, and bounds how large that
effect is.  Quoting the 11.5 points as evidence for the composition
constraint would be wrong.
