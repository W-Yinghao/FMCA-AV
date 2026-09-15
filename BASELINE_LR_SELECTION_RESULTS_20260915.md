# Per-method learning rates for the external baselines — results, 2026-09-15

Against `BASELINE_LR_SELECTION_PREREG_FROZEN_20260914`. Record only.

Grid: simclr, barlow_twins, vicreg, moco_v2 x lr {0.03 (the existing
reference seed-1 unit), 0.1, 0.3}, seed 1, 200 epochs, reference-recipe
configs (`configs/ssl_reference_lr/`): 8 new units. Selection on the
holdout criterion, then seeds 2 and 3 at the selected rate (8 units),
then 24 test-set stage profiles (epochs 20 and 200, three seeds;
`results/baseline_layerwise/reflr_*`). All complete, no failures.

## Selection (5,000-image holdout, seed-0 permutation; epoch 200)

Stages 1-4 on the holdout, then the stage 2-4 mean, per grid point.

    method        lr 0.03                           lr 0.1                            lr 0.3                            selected
    simclr        55.86/68.04/78.88/81.88   76.27   55.68/68.38/83.26/87.04   79.56   56.82/67.12/83.82/87.50   79.48   0.1
    barlow_twins  55.92/68.24/79.04/82.82   76.70   56.06/67.06/83.00/88.34   79.47   54.64/65.38/83.48/89.16   79.34   0.1
    vicreg        57.78/69.56/79.34/82.22   77.04   55.70/67.84/82.74/87.22   79.27   56.14/67.28/83.90/89.10   80.09   0.3
    moco_v2       54.78/65.20/74.76/77.82   72.59   54.22/65.06/78.04/82.28   75.13   55.42/64.70/79.20/84.34   76.08   0.3

Source: `results/baseline_layerwise/lrsel_*_holdout.json` (`score_set`
= holdout-5000). The test set was not consulted.

**P-L1 CONFIRMED**: simclr, barlow_twins and vicreg all select above
0.03. **P-L2** carried no prediction; moco_v2 selects 0.3. For simclr
and barlow_twins the 0.1-vs-0.3 decision is inside 0.15 point on the
criterion; the rule takes the higher mean (and the lower rate on a tie).

## Test-set rows at the selected rate (n = 3; same evaluator as every other row)

    epoch 200          stage 1          stage 2          stage 3          stage 4
    simclr 0.1         [54.77, 55.32]   [67.45, 68.45]   [81.81, 82.24]   [86.05, 86.36]
    barlow_twins 0.1   [55.30, 56.63]   [67.31, 68.25]   [82.43, 82.56]   [87.40, 87.72]
    vicreg 0.3         [55.67, 56.46]   [67.00, 67.86]   [82.84, 83.54]   [88.23, 88.61]
    moco_v2 0.3        [53.80, 55.40]   [64.60, 64.89]   [78.30, 78.45]   [83.42, 83.84]

    epoch 20
    simclr 0.1         [52.88, 54.55]   [62.70, 64.76]   [71.22, 71.65]   [73.96, 74.84]
    barlow_twins 0.1   [53.80, 54.41]   [64.06, 64.53]   [73.54, 73.97]   [76.06, 76.98]
    vicreg 0.3         [52.78, 54.32]   [63.82, 64.45]   [74.29, 74.93]   [77.45, 77.89]
    moco_v2 0.3        [49.99, 50.68]   [57.25, 57.62]   [62.15, 62.84]   [63.90, 64.92]

Against the lr-0.03 rows in `SSL_REFERENCE_RECIPES_RESULTS_20260914.md`:
at epoch 200, stage 4 rises for every method with disjoint ranges
(simclr 81.80-81.88 -> 86.05-86.36; barlow_twins 82.43-82.90 ->
87.40-87.72; vicreg 81.68-82.19 -> 88.23-88.61; moco_v2 77.68-78.45 ->
83.42-83.84; 4.2-6.0 points at the nearest edges) and stage 3 by
2.9-4.3; stages 1 and 2 are unchanged or lower (vicreg stage 2 is lower
by 0.40 with disjoint ranges; simclr and vicreg stage 1 are lower with
disjoint ranges).

## P-L3: rows replaced

All four rates changed, so all four rows are replaced in the reference
table (consolidated write-up §3), with the selection protocol stated in
the caption. simsiam, fastsiam and byol stay at lr 0.03 as the prereg
fixed. The lr-0.03 rows remain on record in
`SSL_REFERENCE_RECIPES_RESULTS_20260914.md`.

## The comparison with FMCA-AV after replacement (stagewise probe, epoch 200)

    FMCA-AV (ridge), n = 5       [57.23, 59.50]  [77.72, 80.69]  [83.96, 85.61]  [83.80, 85.58]
    best external upper edge      vicreg 57.65    simclr 68.45    vicreg 83.54    vicreg 88.61

Stage 2: FMCA-AV's lower edge exceeds the best external upper edge by
9.27 (previously 9.0). Stage 3: by 0.42 over vicreg (previously 4.4
over barlow_twins). Stage 4: FMCA-AV [83.80, 85.58] is BELOW vicreg
[88.23, 88.61], barlow_twins [87.40, 87.72] and simclr [86.05, 86.36]
with disjoint ranges (2.65, 1.82 and 0.47 at the nearest edges) and
overlaps moco_v2 [83.42, 83.84]. The earlier "0.90 above Barlow Twins
at stage 4" no longer holds. At epoch 20, FMCA-AV stage 4 [67.62, 70.83]
is below the three re-selected methods (73.96-77.89) and above moco_v2;
its stage 2 [61.77, 68.09] overlaps all three.

The truncated 45k arm (current multiview term) at stage 4 [79.55, 79.86]
is below all four re-selected methods; at stage 2 [77.96, 78.14] it is
above all of them.

## Declared limits, unchanged from the prereg

Three-point grid, seed-1 selection, holdout criterion: the best of three,
not a tuned optimum. FMCA-AV's own learning rate (0.1) was not
re-selected under the same protocol, so the comparison is between a fixed
FMCA-AV recipe and baselines given a three-point search.
