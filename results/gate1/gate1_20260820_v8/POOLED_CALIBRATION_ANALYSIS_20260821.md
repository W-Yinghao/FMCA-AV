# Disclosed instrument study: pooled Stage-B calibration (2026-08-21)

Question: does the v7-vs-additive measured-defect tie at calibration
2500 come from Stage-B estimation noise?  Method: pool the equally
held-out val split into Stage-B (5000 samples), re-run the frozen
evaluation at three measurement ridges.  Changes no frozen verdicts.

## Normalized closure defect, calibration 5000 (was: calibration 2500)

| unit | r=1e-3 | r=1e-2 | r=1e-1 |
|---|---|---|---|
| v7 s1 | 0.277 | 0.269 | 0.215 (was 0.228) |
| v7 s2 | 0.250 | 0.242 | 0.197 (was 0.253) |
| v7 s3 | 0.273 | 0.266 | 0.216 (was 0.256) |
| additive s1 | 0.234 | 0.224 | 0.188 (was 0.228) |
| additive s2 | 0.264 | 0.253 | 0.199 (was 0.265) |
| additive s3 | 0.280 | 0.268 | 0.209 (was 0.249) |

Means at r=1e-1: v7 0.209, additive 0.199.  At r=1e-3: v7 0.267,
additive 0.259.

## Findings

1. Doubling calibration lowers every defect by roughly 0.04 at
   r=1e-1: a real estimation-noise floor component, consistent with
   the Thm-1 epsilon_n term shrinking with n.
2. It does NOT separate explicit from implicit closure: additive sits
   level with (marginally below) v7 at every ridge, within seed
   spread.  The tie is not a calibration-sample artifact.
3. Interpretation for the paper: by this instrument, additive
   training achieves measured closure comparable to explicit
   composition training.  The framework's demonstrated value is the
   instrument itself: edge-localized spectroscopy, the parallel-tree
   and product-only negative controls, depth profiles, and the
   plug-in defect reductions on external SSL bases.  Claims that
   explicit closure training uniquely lowers the measured defect are
   not supported at this scale.
4. Open mechanism question: selection inflation on the directly
   optimized C_dir could bias v7's normalized defect upward; testing
   that needs the cross-fitted epsilon_n debias, left to the Thm-2
   instrument work.
