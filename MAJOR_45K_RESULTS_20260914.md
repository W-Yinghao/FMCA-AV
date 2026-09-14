# MAJOR at 45000 images, 200 epochs — results, 2026-09-14

Against `MAJOR_45K_800EPOCH_PREREG_FROZEN_20260913` as amended by
`MAJOR_45K_EPOCH_BUDGET_APPENDUM_20260913`. Record only.

Six units, all complete, no failures. `paper_composition` under the
truncated estimator, three disjoint seeds per dataset, 45000 training
images (n_calibration 2500, n_val 2500), 200 epochs, checkpoints kept
and evaluated at epochs 20 and 200. Config diff against the 30000-image
grid is exactly `n_calibration`, `n_val`, `checkpoint_milestones`.

## Endpoint accuracy against the 30000-image, 200-epoch grid

    dataset    instrument            30k/200ep (n=3)    45k/200ep (n=3)
    CIFAR-10   SGD probe, layer4     [82.43, 82.84]     [78.32, 78.47]
    CIFAR-100  SGD probe, layer4     [44.66, 45.91]     [42.48, 44.48]

**P-M1 REFUTED. P-M2 REFUTED.** Both predicted the 45000-image arm
would beat its 30000-image counterpart with disjoint ranges. On both
datasets it is lower, with disjoint ranges in the opposite direction:
4.1 points on CIFAR-10, 0.2 points on CIFAR-100 at the nearest edges.

The 45000-image training set is a strict superset of the 30000-image
one for the same seed (the split is `permutation[n_cal + n_val:]` on a
seed-fixed permutation), so the arm saw every image the earlier arm saw
plus 15000 more, and took 35000 optimizer steps against 23400.

## Per-tap convex probe, epoch 200

    CIFAR-10          30k              45k                CIFAR-100         30k              45k
    stem          [48.27, 50.74]  [47.98, 49.06]         stem          [21.93, 22.33]  [21.64, 22.71]
    layer2.1      [79.56, 80.29]  [77.96, 78.15]         layer2.1      [49.61, 49.94]  [50.10, 50.94]
    layer3.1      [82.63, 82.88]  [80.19, 80.94]         layer3.1      [50.40, 51.42]  [52.16, 52.45]
    layer4.0      [82.96, 83.55]  [80.15, 80.63]         layer4.0      [48.73, 50.31]  [50.33, 51.51]
    layer4.1      [82.36, 82.74]  [79.45, 79.86]         layer4.1      [47.44, 49.23]  [48.62, 50.32]

On CIFAR-100 the convex probe reads the 45k arm slightly HIGHER from
layer2.1 onward while the SGD probe reads it lower; the two instruments
disagree by about a point in opposite directions and neither gap is
large. On CIFAR-10 both instruments agree: the 45k arm is 2-4 points
lower at every tap from layer2.1 on.

## Epoch 20 (diagnostic, no prediction)

    CIFAR-10   ep20   layer2.1 [68.8, 71.0]  layer3.1 [70.4, 73.5]  layer4.0 [70.2, 72.6]
    CIFAR-100  ep20   layer2.1 [41.3, 41.4]  layer3.1 [41.0, 42.0]  layer4.0 [40.6, 41.1]

## The training state, which is where the two arms actually differ

`train/retained_min` is the number of directions (of 128) the truncated
whitener keeps at the most-truncated level, per epoch.

    CIFAR-10  30k  s1   34  41  43  49  65  106  128  128  128    (ep 0 2 5 10 25 50 100 150 199)
    CIFAR-10  30k  s2   41  50  47  56  77  115  128  128  128
    CIFAR-10  30k  s3   47  45  42  48  77  115  128  128  128
    CIFAR-10  45k  s1   34  18  20  20  20   20   20   20   20
    CIFAR-10  45k  s2   43  20  21  21  21   21   21   21   21
    CIFAR-10  45k  s3   48  17  18  18  18   18   18   18   18
    CIFAR-100 30k  s1-3 34-46 -> 17-19 by epoch 2, pinned at 19-20 to the end
    CIFAR-100 45k  s1-3 36-47 -> 17-19 by epoch 2, pinned at 18-20 to the end

Every run starts at 34-48. By epoch 2 the CIFAR-10 30k runs are at
41-50 and climb to 128 by epoch 100; every other run in this table is
at 17-20 by epoch 2 and never moves again. The CIFAR-10 45k trajectory
is indistinguishable from the CIFAR-100 trajectory at either split.

The rest of the training state follows the retained rank:

                          C10 30k              C10 45k
    final train/loss      [-219.1, -218.6]     [-41.0, -29.7]
    edge_trace_sum        [211.6, 211.8]       [77.5, 119.5]
    dir_trace             [94.2, 94.5]         [19.1, 21.7]
    leaf_trace            [98.5, 98.7]         [15.2, 16.9]
    closure_ratio         [0.13, 0.13]         [0.03, 0.05]

`dir_trace` on the 45k runs rises from 6 to 19-22 over 200 epochs; on
the 30k runs it rises from 6 to 94.

## Certificates at 2500 calibration roots (declared non-comparable to the 10000-root ones)

    CIFAR-10  10000-root, 30k arm   [128,128,128] x3      endpoint_norm 9.08-9.29   rce 0.160-0.169
    CIFAR-10  2500-root,  45k arm   [92,115,20] [99,107,21] [68,95,17]
                                                           endpoint_norm 3.64-4.14   rce 0.115-0.263
    CIFAR-10  2500-root,  ep20      [46,66,20] [46,66,21] [46,66,18]
                                                           endpoint_norm 3.62-3.85   rce 0.095-0.124
    CIFAR-100 10000-root, 30k arm   [90,106,19] [95,113,20] [92,110,19]
                                                           endpoint_norm 3.65-3.78   rce 0.133-0.184
    CIFAR-100 2500-root,  45k arm   [100,114,16] [104,121,20] [94,90,1]
                                                           endpoint_norm 3.58 / 3.94 / 0.08
                                                           rce 0.256 / 0.156 / 3.741

CIFAR-100 45k seed 3 certifies with an endpoint level of retained rank
1, endpoint norm 0.08 and relative closure error 3.74. Its SGD probe is
42.94 and its convex profile is in range with the other two seeds, so
the representation decodes; what its certificate measures at 2500 roots
is a different question and is recorded, not resolved, here.

The epoch-200 milestone certificates match the last-checkpoint
certificates on retained ranks and closure error to four decimals on
every unit, which validates the milestone pipeline.

## Standing

The frozen grid for a refuted P-M1 reads: "MAJOR does not benefit from
the longer schedule or the extra data on that dataset. Reported as such.
It does NOT license re-running with a different learning rate or
schedule to find a number that does improve." No re-run is proposed
here.

What this wave adds beyond the refutation is a fact about the corpus:
across every MAJOR configuration now on disk -- two datasets, two
splits, six arms, two robustness axes, a star control -- the only runs
that ever reach retained rank 128 of 128 are `paper_composition` and
`paper_beta0` on CIFAR-10 at the 30000-image split. Every other run pins
at 17-21 within the first two to five epochs. That bifurcation, and
which branch a run lands on, is not something the frozen prereg
predicted or explained, and it is not explained here.
