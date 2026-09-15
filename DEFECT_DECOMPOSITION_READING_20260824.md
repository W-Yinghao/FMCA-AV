# E-A1 reading: the ratio is confounded across arm families

The audit's first item asked whether the normalized defect falls
because the numerator shrank or because the denominator moved.  Both
terms were already in every unit.json.  Printed side by side, they
change what can be claimed.

## CIFAR-10 gate v8

| row | n | ratio | numerator | denominator | C_dir effrank | probe |
|---|---|---|---|---|---|---|
| V7 explicit | 5 | 0.370 | **2.078** | 5.500 | 43.0 | 84.98 |
| additive M | 5 | 0.384 | 3.815 | 9.925 | 122.9 | 78.27 |
| amdim-cross | 3 | 0.369 | 3.688 | 9.990 | 123.0 | 78.25 |
| flat M-view | 5 | 0.482 | **1.955** | 4.058 | 86.5 | 88.97 |
| flat 2-view | 3 | 0.453 | 1.917 | 4.229 | 87.8 | 85.29 |

Across rows: `corr(ratio, numerator) = -0.174`,
`corr(ratio, denominator) = -0.567`.

**The flat arm has the smallest numerator of any row -- smaller than
V7's.**  V7's lower ratio on CIFAR-10 comes from a denominator that is
36% larger (5.500 against 4.058), not from a closer composition.

## CIFAR-100 pilot

| row | n | ratio | numerator | denominator | C_dir effrank | probe |
|---|---|---|---|---|---|---|
| V7 explicit | 3 | 0.375 | 3.863 | **10.290** | 122.3 | 55.89 |
| additive M | 3 | 0.398 | 4.015 | **10.084** | 122.7 | 48.52 |
| amdim-cross | 3 | 0.411 | 4.227 | 10.282 | 122.7 | 49.03 |
| flat M-view | 3 | 0.485 | 2.198 | 4.530 | 89.5 | 61.46 |

Here `corr(ratio, numerator) = +0.983` -- on this dataset the ratio does
track the numerator.  But the flat rows still sit at a denominator near
4.5 while every hierarchical row sits near 10.

## What survives, and what does not

1. **Cross-family ratio comparisons do not survive.**  Flat rows and
   hierarchical rows differ by more than a factor of two in the
   denominator and by 35+ points of endpoint effective rank.  They are
   not on a common scale, so "the instrument separates arms trained for
   closure from arms not trained for closure" is **withdrawn as
   stated**: measured by the numerator alone, the flat arm is the
   *closest* to closure on CIFAR-10.
2. **V7 against additive on CIFAR-100 survives cleanly, and is now
   better supported than before.**  Denominators match within 2%
   (10.290 against 10.084) and endpoint effective ranks within 0.4
   (122.3 against 122.7), so the ratio gap is a numerator gap: 3.863
   against 4.015.  This is a like-for-like comparison.
3. **The star control on CIFAR-100 mostly survives.**  Star numerator
   4.806 against nested 3.863, +24% in the direction the ontology
   predicts; the denominators differ by 10% (9.305 against 10.290),
   which is small next to the numerator gap but is not zero, so the
   claim should be stated with both terms.
4. **product_only stays a counterexample under either reading**:
   numerator 18.1 against a denominator of 9.1 on CIFAR-100.

## Consequences

- Every table reporting a defect must from now on report
  `(numerator, denominator)` alongside the ratio.  The meeting tables
  and the paper's §V/§VI get the extra columns.
- The affected claims are in `POOLED_CALIBRATION_ANALYSIS_20260821.md`
  (the "instrument separates trained from untrained" reading) and in
  section 3 of `REPORT_BRIEF_20260822.md`.  Both are corrected to the
  like-for-like claim.
- The degenerate-encoder control rows (random-init and collapsed
  encoders) are still required and still pending -- they test whether a
  weak encoder can buy a low ratio outright, which this decomposition
  suggests is exactly the failure mode to fear.

## E-B2 side note (approximate, from existing logs)

Final training values, CIFAR-10 seed 1: the additive arm drives edge
mass to 205.1 against V7's 121.1 -- 69% more per-edge mass -- while its
measured defect is no better (0.384 against 0.370).  Mass at the edges
does not convert into closure, which is the nilpotent counterexample's
behaviour appearing in a real network.  The exact figure wants per-edge
Frobenius norms and sigma_1(C_comp) logged per epoch; those were never
instrumented, so this is the qualitative version only.
