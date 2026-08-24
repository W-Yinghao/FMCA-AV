# T0 validity gate: does not pass

Criteria frozen in the 08-23 plan, evaluated on ten CPU-only runs.
Two of the three checks fail.  The plan's own rule -- "T0 先于一切；
不通过则仪器主张全体收回" -- therefore applies: the instrument claims
are suspended pending the protocol fix in section 4.

## E-A1 degenerate controls: FAILS

Frozen criterion: if a degenerate row reaches a defect at or below
V7's, the ratio form is void.

| row | ratio | numerator | denominator | endpoint top |
|---|---|---|---|---|
| CIFAR-10 random encoder (never trained) | **0.339** | 0.761 | 2.249 | 0.802 |
| CIFAR-10 V7 (trained) | 0.370 | 2.078 | 5.500 | 1.352 |
| CIFAR-10 flat (trained) | 0.482 | 1.955 | 4.058 | 1.082 |
| CIFAR-100 random encoder | **0.325** | 0.798 | 2.451 | 0.846 |
| CIFAR-100 V7 (trained) | 0.375 | 3.863 | 10.290 | 1.779 |
| CIFAR-100 flat (trained) | 0.485 | 2.198 | 4.530 | 1.067 |

**A randomly initialized encoder that has never seen a gradient scores
a better normalized defect than the trained V7 arm, on both datasets.**
The criterion fires: the ratio is void as a standalone statistic.

The mechanism is the one the decomposition predicted a day earlier: a
weak encoder has a small numerator *and* a small denominator, and the
quotient flatters it.  Absolute scale does separate -- the random
encoder's endpoint operator tops out at 0.80-0.85 against 1.07-1.78 for
every trained row, and its denominator is 2.2-2.5 against 4.1-10.3 --
so the information is there, just not in the ratio.

The collapsed encoder (stage outputs shrunk to 1% of their spread)
blows up instead: ratio 167 and 201, numerator 4.0e4 and 9.7e4.
Collapse is detected loudly, which is the easy half of the test; the
random encoder is the one that matters and it walks straight through.

## E-A2 calibration convergence: FAILS

Frozen criterion: successive doubling changes the defect by < 0.005.
The clean pool is `n_calibration + n_val` = 5000; beyond that the
images were seen in SSL training, so the sweep stops there.

Defect against N (seed 1, ridge 1e-3), and the change over the last doubling:

| arm | N=625 | N=1250 | N=2500 | N=5000 | last change |
|---|---|---|---|---|---|
| CIFAR-10 V7 | 1.478 | 0.444 | 0.316 | 0.277 | 0.039 |
| CIFAR-10 additive | 1.288 | 0.540 | 0.333 | 0.230 | 0.102 |
| CIFAR-10 flat | 0.565 | 0.507 | 0.490 | 0.482 | **0.008** |
| CIFAR-100 V7 | 3.450 | 0.725 | 0.377 | 0.248 | 0.129 |
| CIFAR-100 additive | 2.953 | 0.727 | 0.409 | 0.258 | 0.151 |
| CIFAR-100 flat | 0.600 | 0.492 | 0.480 | 0.473 | **0.006** |

No hierarchical arm is within an order of magnitude of the tolerance at
the ceiling of the clean budget.  Two consequences:

1. **Absolute defect values may not be cited.**  Every reported
   hierarchical defect is a point on a steeply descending curve, not a
   measurement.
2. **The CIFAR-100 separation shrinks as the sample grows.**  V7
   against additive is 0.377 vs 0.409 at N=2500 (gap 0.032) and 0.248
   vs 0.258 at N=5000 (gap 0.010) -- a threefold reduction from a
   single doubling.  It may or may not survive at convergence; on this
   evidence it cannot be claimed.  (Single seed; the N=2500 values
   match the three-seed means, so seed 1 is representative.)
3. The arms converge at very different rates -- flat is essentially
   converged, the hierarchical arms are not -- so comparing them at a
   fixed N compares a converged number with an unconverged one.  This
   compounds the denominator confound recorded on 08-24.

## E-A5 population coverage: DELIVERS

The frozen G4 compares s_cert against the same-sample endpoint spectrum
and its violations sit at 5e-17, so it is an implementation check.
Against the *population* spectrum, on the same 1441 Wave 0 units:

| N | M=1 | M=4 | M=16 | max relative excess |
|---|---|---|---|---|
| 1,000 | 0.138 | 0.169 | 0.206 | 7.4e-2 … 2.5e-1 |
| 10,000 | 0.156 | 0.181 | 0.163 | 3.7e-2 … 5.3e-2 |
| 100,000 | 0.131 | 0.163 | 0.169 | 6.9e-3 … 1.8e-2 |

The violation **rate** is roughly flat in N at 13-21%; the violation
**magnitude** falls by an order of magnitude from N=1e3 to N=1e5.  That
is exactly the shape a finite-sample epsilon_n term should have, and it
is now measured rather than declared -- the empirical anchor Theorem 2
needs.  By case: hallucinated_path 0.528, nilpotent 0.328, closed_chain
0.306; isospectral_mismatch, leaky_interface and zero_operator are all
0.000.

## What this costs, and the fix

Suspended pending remediation: every absolute defect value, the
CIFAR-10 pooled-calibration comparison, the CIFAR-100
explicit-versus-additive separation, and the star-control separation
(same instrument, same unconverged regime).

Not affected, because they never used the ratio: the layer-wise probe
and effective-rank profiles, the 2x2 that separates loss form from view
nesting, the accuracy tables, and the Wave 0 acceptance battery.

The fix is a protocol change, not an analysis change.  The held-out
budget has to grow enough that the convergence criterion can actually
be tested -- `n_calibration = n_val = 10000` leaves 30000 training
images and permits a sweep to N=20000, two doublings past today's
ceiling.  That means retraining the arms that carry instrument claims.
Until those land, defect numbers are reported as
`(numerator, denominator, N)` triples with the convergence caveat
attached, and no cross-arm ratio comparison is made.
