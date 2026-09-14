# External baselines at their own recipes — results, 2026-09-14

Against `SSL_REFERENCE_RECIPES_PREREG_FROZEN_20260913`. Record only.

Eighteen units, all complete, no failures. Six methods, three disjoint
seeds, CIFAR-10, 45000 images, 200 epochs, checkpoints at 20 and 200,
each profiled per stage with the same convex probe every MAJOR arm is
scored with. Optimizer unchanged from `ssl_matched` (SGD, lr 0.03) by
the prereg's own rule.

## Per-stage convex probe, epoch 200 (n=3)

    method            layer1          layer2          layer3          layer4
    barlow_twins  [55.60,56.85]  [68.04,68.29]  [79.14,79.57]  [82.43,82.90]
    vicreg        [57.21,57.65]  [68.26,68.70]  [78.64,79.15]  [81.68,82.19]
    simclr        [55.77,56.33]  [67.31,67.86]  [78.50,78.79]  [81.80,81.88]
    moco_v2       [54.71,54.89]  [64.55,65.28]  [73.73,74.03]  [77.68,78.45]
    fastsiam      [55.49,55.58]  [64.62,65.42]  [72.62,72.78]  [75.35,75.58]
    simsiam       [55.19,55.44]  [64.22,64.87]  [71.73,72.55]  [74.11,74.75]

    MAJOR 45k     [58.22,58.84]  [77.96,78.14]  [80.21,80.94]  [79.55,79.86]
    (stage-end taps of the block profile; same data, same epochs, same probe)

## Epoch 20 (diagnostic)

    method            layer1          layer2          layer3          layer4
    barlow_twins  [52.35,53.61]  [60.78,61.31]  [68.20,68.33]  [70.36,70.95]
    vicreg        [53.29,54.65]  [62.27,62.72]  [68.59,69.20]  [70.90,71.73]
    simclr        [53.63,54.74]  [61.78,62.26]  [68.10,69.35]  [69.98,70.30]
    moco_v2       [48.95,50.96]  [55.78,55.88]  [60.11,60.35]  [62.45,63.02]
    fastsiam      [51.23,51.51]  [56.38,57.23]  [59.11,59.78]  [58.46,59.84]
    simsiam       [50.65,51.57]  [55.59,56.45]  [57.92,58.36]  [56.32,57.38]
    MAJOR 45k     [55.97,56.57]  [68.75,70.99]  [70.40,73.49]  [69.40,71.96]

## Predictions

P-R1 CONFIRMED. simsiam and fastsiam do not collapse on any seed:
probe accuracy rises from layer1 to layer4 on all six runs (+18.7 to
+20.1 points), and validation cosine settles at 0.816-0.818 (simsiam)
and 0.877-0.879 (fastsiam) against BYOL-without-BN's 0.98-0.99. Every
other method rises with depth on every seed as well.

P-R2 MIXED, two of four.

    method        ssl_matched layer4   reference layer4   verdict
    barlow_twins  [79.84, 80.81]       [82.43, 82.90]     higher, disjoint
    vicreg        [81.44, 81.66]       [81.68, 82.19]     higher, disjoint
    simclr        [81.90, 82.28]       [81.80, 81.88]     LOWER by 0.02-0.4
    moco_v2       [78.65, 79.30]       [77.68, 78.45]     LOWER by 0.2-0.9

The reference projector (2048-wide, BatchNorm, method-specified
constants) helped the two decorrelation methods and did not help the
two contrastive ones at this learning rate. For simclr the change is
inside a point and the ranges nearly touch; for moco_v2 the reference
recipe with momentum 0.999 is worse than the copy-pasted one with the
0.996 fallback by up to 0.9 points. Reported as refuted for both; not
re-tuned.

P-R3 carried no prediction. On matched data and epochs, MAJOR is
[79.55, 79.86] at layer4 and sits below barlow_twins, vicreg and
simclr, above moco_v2, fastsiam and simsiam. At layer2 it is
[77.96, 78.14] against the best baseline's [68.26, 68.70].

## Declared limits, unchanged

The optimizer was not tuned per method. lr 0.03 is near the published
SimSiam and MoCo CIFAR settings and below what SimCLR, Barlow Twins and
VICReg are normally run at; the P-R2 outcome for simclr and moco_v2 is
consistent with a learning-rate interaction and is not evidence about
the methods. A MoCo v2 at 83.8 on CIFAR-10 exists in the literature and
in this user's own prior runs; this wave's 77.7-78.5 is a statement
about this recipe, this queue size and this learning rate, not about
MoCo v2.

---

## Addendum, 14 September — BYOL at the reference recipe

Against `SSL_REFERENCE_RECIPES_APPENDUM_BYOL_20260914`. Three seeds,
4096 → 256 projector and predictor with BatchNorm, EMA momentum 0.996,
same budget and optimizer, milestones 20 and 200.

    epoch 200   layer1 [54.65,55.24]  layer2 [63.15,63.94]  layer3 [69.89,70.23]  layer4 [72.91,73.95]
    epoch 20    layer1 [48.94,50.40]  layer2 [53.64,53.99]  layer3 [55.33,56.18]  layer4 [54.41,55.86]
    validation cosine 0.8295 / 0.8348 / 0.8339

P-B4: no collapse on any seed (accuracy rises 18.3–19.3 points from
layer 1 to layer 4) — confirmed on that clause. On level, the range
[72.91, 73.95] overlaps the old-template BN run's [73.46, 74.75] and its
lower edge is 0.55 below; the wider reference projector did not improve
BYOL at lr 0.03, consistent with the SimCLR and MoCo v2 outcome under
P-R2. P-B5 carried no prediction: BYOL is seventh of seven at every
stage.
