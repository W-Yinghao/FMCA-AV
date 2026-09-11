# The paper estimator on CIFAR-10 — adjudication, 2026-09-11

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
against 400) and was NOT rescued; the config is deterministic, so a
rerun would reproduce it exactly.  Across seeds this arm swings
edge_trace_sum 115 to 388, and one certificate shows retained ranks
[88, 1, 2] with endpoint norm 0.099 -- a near-collapse.  Removing the
multiview aggregation term destabilises training and admits collapse.
The prereg gave lambda0 no prediction; this is the report.

## Standing caveats

Single dataset.  The external-baseline comparison is carried over from
the matched-budget wave and is estimator-independent, but those arms
were compared against the RIDGE corpus; a like-for-like table against
these arms is not yet drawn.  BYOL remains excluded as a collapsed run.
