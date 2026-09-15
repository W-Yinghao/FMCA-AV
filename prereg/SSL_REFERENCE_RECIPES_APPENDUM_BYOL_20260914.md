# Appendum — BYOL added to the reference-recipe wave

Frozen 2026-09-14 before the arm runs. Extends
`SSL_REFERENCE_RECIPES_PREREG_FROZEN_20260913`, which covered six methods
and left BYOL out because its only existing runs had collapsed.

**Disclosure:** BYOL has since been run with BatchNorm added to the OLD
template projector (512-512 -> 128) and reached layer 4 [73.46, 74.75]
on three seeds (`BYOL_COLLAPSE_DIAGNOSIS_20260913.md`). Those numbers
have been seen. This arm runs BYOL at its own projector/predictor
geometry (4096 -> 256 both, BatchNorm, EMA momentum 0.996), same budget
and optimizer as the other six reference runs, three seeds, milestones
at epochs 20 and 200.

P-B4. BYOL does not collapse (probe accuracy rises with depth on every
      seed) and its layer-4 range is at or above the old-template BN run's
      [73.46, 74.75]. REFUTED if it falls with depth on any seed or its
      range lies below 73.46.
P-B5. No prediction on its placement among the six.

Optimizer is not tuned, as for the other six; the same understatement
caveat applies.
