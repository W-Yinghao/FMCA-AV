# Step 3 (H-G1) and the enlarged-calibration retraining

Two results landed together.  One rejects a candidate mechanism; the
other reverses a withdrawal I made on 08-24.  Results only; the reading
follows in the next commit.

## H-G1: the Gram artifact is real but empirically negligible

Extended diagnostic, checkpoints loaded (fingerprints distinct), Stage-B
pool 10000 / Stage-C 5000, projection and surrogate side by side.

| arm | surrogate last-doubling | projection last-doubling | artifact share at N=10000 | ‖I−Ĝ_l‖₂ per level | retained ranks |
|---|---|---|---|---|---|
| c10 V7 | 0.1170 | 0.1137 | 0.0041 | 1.000, 0.0062, 0.0048 | **21**, 128, 128 |
| c10 additive | 0.0441 | 0.0445 | 0.0011 | 0.0037, 0.0071, 0.0091 | 128, 128, 128 |
| c10 flat | 0.0025 | 0.0022 | 0.0125 | 0.607, 0.069, 0.0031 | 128, 128, 128 |
| c100 V7 | 0.0457 | 0.0462 | 0.0013 | 0.0104, 0.0078, 0.0055 | 128, 128, 128 |
| c100 additive | 0.0531 | 0.0534 | 0.0012 | 0.0035, 0.006, 0.0186 | 128, 128, 128 |
| c100 flat | 0.0038 | 0.0040 | 0.0122 | 0.578, 0.067, 0.003 | 128, 128, 128 |

**H-G1 fails.**  Correcting the metric moves the defect by 0.001-0.013,
between 0.1% and 1.3% of its value, and the hierarchical arms' drift is
unchanged: 0.044-0.117 per doubling at N=10000 against a 0.02
threshold.  The candidate mechanism named in the directive is now
tested and does not explain the T0 failures.

Two incidental observations, neither preregistered:

1. **c10 V7's root coordinate system retains 21 of 128 directions.**
   Its level-0 Gram sits at ‖I−Ĝ‖₂ = 1.000 -- the ridge annihilated
   most of the root subspace -- and it is also the worst-converging arm
   (0.117).  No other cell shows this.
2. The flat rows carry the *largest* metric deviation at level 0
   (0.58-0.61) yet converge fastest, so the artifact size and the
   convergence problem are not the same quantity.

## Enlarged calibration: n_calibration = n_val = 10000, 30k SSL training

18/18 complete.  Accuracy is not comparable to the v8 gate (30k against
45k training images) and is reported only to show the arms moved
together.

| dataset | arm | probe | defect | numerator | denominator |
|---|---|---|---|---|---|
| CIFAR-10 | flat M | 86.14 ± 0.33 | 0.472 ± 0.005 | 1.870 | 3.959 |
| CIFAR-10 | **V7** | 82.30 ± 0.61 | **0.192 ± 0.008** | **1.024** | 5.428 |
| CIFAR-10 | additive M | 75.19 ± 0.52 | 0.202 ± 0.017 | 1.866 | 9.249 |
| CIFAR-100 | flat M | 57.46 ± 0.16 | 0.479 ± 0.004 | 2.089 | 4.359 |
| CIFAR-100 | **V7** | 53.66 ± 0.09 | **0.190 ± 0.003** | 1.791 | 9.424 |
| CIFAR-100 | additive M | 45.61 ± 0.55 | 0.199 ± 0.005 | 1.851 | 9.315 |

At this budget V7 has the **smallest numerator of any arm on
CIFAR-10** (1.024 against flat's 1.870 and additive's 1.866), and on
CIFAR-100 it beats additive on both terms at denominators matched
within 1.2% (9.424 against 9.315), with three-seed ranges disjoint
(0.187-0.193 against 0.194-0.204).  CIFAR-10 V7 and additive still
overlap (0.184-0.200 against 0.185-0.219).

Accuracy fell 2.7-3.8 points across every arm relative to v8, as the
smaller training set predicts, and the ordering is unchanged.
