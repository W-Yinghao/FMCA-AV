# Frozen before compute — per-method learning rates for the external baselines

Frozen 2026-09-14. The reference-recipe wave held SGD lr 0.03 for all
six methods and declared that SimCLR, Barlow Twins and VICReg are
normally trained higher. This wave selects a learning rate per method
with the criterion fixed before any run.

## Grid and selection

Methods: simclr, barlow_twins, vicreg, moco_v2. Learning rates: 0.03
(the existing reference seed-1 run), 0.1, 0.3 — one seed (seed 1) per
new point, 200 epochs, everything else as the reference config.

**Criterion.** For each method, evaluate the epoch-200 checkpoint of each
grid point with the stagewise convex probe fitted on the training pool
minus a fixed 5,000-image holdout (seed-0 permutation) and scored on the
holdout (`--holdout 5000`; the record carries `score_set`). Select the
learning rate with the highest mean of stages 2–4 on the holdout; ties
go to the lower rate. **The test set is not consulted during
selection.**

Then run seeds 2 and 3 at the selected rate and report all three seeds
on the test set with the standard evaluator. If the selected rate is
0.03 the existing reference row stands and no new seeds are run.

## Predictions

P-L1. The selected rate exceeds 0.03 for simclr, barlow_twins and vicreg.
      REFUTED for any of the three whose holdout criterion is maximised
      at 0.03.
P-L2. No prediction for moco_v2.
P-L3. No prediction for how the re-selected rows compare with FMCA-AV.
      Whatever the outcome, the reference table is replaced only for the
      rows whose rate changed, and the caption states the selection
      protocol.

## Limits

Three points is a coarse grid; the selected rate is the best of three,
not a tuned optimum. Seed 1 alone drives selection, which is the usual
practice and is stated. FastSiam and SimSiam are left at 0.03, which is
near their published CIFAR settings.
