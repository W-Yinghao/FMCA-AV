# Does endpoint dependence factorize through the path?

Disclosed post-hoc analysis of records already on disk (no new
compute), testing the factorization claim: if the endpoint operator
must be built along the path, the endpoint's dominant directions
should lie inside the path-supported subspace.  Run
`scripts/analyze_factorization.py` to reproduce.

## Statistics

- `alignment`: cosine between C_dir and C_comp as matrices, scale-free.
- principal angles: between the top-8 singular subspaces of C_dir and
  C_comp.  **This is the non-circular statistic** -- it asks where the
  endpoint directions sit, not how large the defect is.
- certified fraction: reported but NOT evidence on its own, because
  s_cert = [sigma(C_comp) - delta_op]_+ subtracts the defect, so a
  smaller defect scores higher by construction.

## Result

| dataset | row | alignment | angle L | per-seed angle L |
|---|---|---|---|---|
| CIFAR-10 | V7 explicit | 0.969 | 12.7° | 21.7, 12.4, 4.0 |
| CIFAR-10 | additive (path-blind) | 0.969 | 15.1° | 16.1, 16.6, 12.6 |
| CIFAR-10 | flat M-view | 0.875 | 19.1° | 17.9, 18.9, 20.5 |
| CIFAR-100 | V7 explicit | 0.968 | 13.8° | 11.7, 16.6, 12.9 |
| CIFAR-100 | additive (path-blind) | 0.967 | 15.6° | 17.8, 17.0, 12.1 |
| CIFAR-100 | flat M-view | 0.874 | 24.4° | 26.1, 25.6, 21.6 |

## Reading

1. **Hierarchical training does make endpoint dependence more
   path-factorizable.**  Alignment separates cleanly on both datasets
   with zero seed overlap: V7 and additive at 0.964-0.972, flat at
   0.869-0.878.  On CIFAR-100 the angles separate too: flat 21.6-26.1
   degrees against 11.7-17.8 for the two hierarchical arms.
2. **But it is not specific to explicit composition.**  V7 and
   additive are indistinguishable on both statistics, seed ranges
   fully overlapping.  The path-blind arm factorizes just as well.
   This is the same pattern as the CIFAR-10 defect tie.
3. **The CIFAR-100 defect separation is about magnitude, not
   direction.**  There, V7 (0.375) beats additive (0.398) with no seed
   overlap, yet the two share a subspace geometry.  So explicit
   closure training shrinks *how much* of the endpoint operator falls
   outside the path, without changing *where* the endpoint's dominant
   directions sit relative to it.
4. V7's CIFAR-10 angle spread (4.0 to 21.7 degrees across seeds) is
   too wide to support any claim at n=3 on that dataset.

## Verdict on the factorization prediction

Supported for hierarchical-versus-flat, **not supported** as a
property unique to path-supported composition.  Any paper claim should
be stated at the level the evidence reaches: training with an explicit
intermediate structure -- composed or additive -- makes endpoint
dependence markedly more path-factorizable than flat training does.

## Side note: the sigma-product red line, empirically

Top singular value of the naive per-edge sigma-product against the
true composed operator, CIFAR-10 seed 1: V7 2.045 vs 1.602 (1.28x
inflation), additive 2.468 vs 2.417 (1.02x), flat 1.122 vs 1.063
(1.05x).  The forbidden shortcut inflates most exactly where the
composition is actually trained, which is the concrete reason the red
line matters.
