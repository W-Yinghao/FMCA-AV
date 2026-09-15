# Ridge β = 0, replicated with epoch-20/200 checkpoints — results, 2026-09-15

Against `RIDGE_BETA0_MILESTONE_PREREG_FROZEN_20260914`. Record only.
Six units, all complete, no failures, one run each (no resume).
`product_endpoint` with `loss.beta = 0`, ridge estimator, EMA endpoint
target, 45,000 training images, 200 epochs, three seeds per dataset;
config diff against the archived β = 0 templates exactly
`checkpoint_milestones: [20, 200]`; against the same-day full-model
replication (`gate1_20260914_ridge_ms*`) exactly `loss.beta: 128 → 0`.
Stage values are the stage-end taps of the block profile
(layer1.1 / 2.1 / 3.1 / 4.1), same evaluator as every other row.

## P-C1 — epoch-200 ranges overlap the archived β = 0 ranges: CONFIRMED

    dataset   measure     new (n=3)         archived (n=3)     
    CIFAR-10  SGD probe   [83.42, 83.58]    [83.55, 83.74]     overlap
    CIFAR-10  stage 2     [79.55, 79.92]    [79.91, 80.10]     overlap
    CIFAR-10  stage 4     [83.19, 83.53]    [83.41, 83.63]     overlap
    CIFAR-100 SGD probe   [54.79, 55.55]    [54.79, 55.55]     overlap
    CIFAR-100 stage 2     [53.75, 53.98]    [53.38, 53.98]     overlap
    CIFAR-100 stage 4     [54.90, 55.30]    [54.95, 55.30]     overlap

All six overlap. Independence of the samples, checkpoint to checkpoint
against the archived unit of the same seed:

    unit      archived GPU   new GPU     max |dw|   SGD archived / new
    C10 s1    H100 NVL       L40S        3.9        83.74 / 83.58
    C10 s2    A100-PCIE      L40S        4.7        83.55 / 83.42
    C10 s3    A100-PCIE      L40S        5.8        83.74 / 83.42
    C100 s1   H100 NVL       A100-PCIE   4.1        55.38 / 55.21
    C100 s2   A100-PCIE      A100-PCIE   9.96e-3    55.55 / 55.55
    C100 s3   A100-PCIE      A100-PCIE   9.83e-3    54.79 / 54.79

Four of six are independent replications; CIFAR-100 seeds 2 and 3
landed on the archived GPU family and are near-copies (same pattern as
the full-model replication, see `RIDGE_MILESTONE_REPLICATION_RESULTS_20260914.md`,
note of 15 September). The CIFAR-100 SGD range being identical to the
archived one is those two copies plus seed 1 inside it.

## Per seed

    CIFAR-10   seed  epoch 20  st1/2/3/4                 epoch 200  st1/2/3/4                 SGD     kNN
               1     56.11 / 67.11 / 69.54 / 69.74        60.61 / 79.66 / 83.66 / 83.53        83.58   81.16
               2     56.87 / 67.81 / 70.10 / 69.89        60.03 / 79.55 / 83.83 / 83.35        83.42   81.20
               3     56.30 / 67.35 / 68.83 / 68.15        59.78 / 79.92 / 83.81 / 83.19        83.42   81.04
    CIFAR-100  1     30.29 / 40.73 / 41.78 / 40.77        34.22 / 53.97 / 57.66 / 54.90        55.21   49.33
               2     30.71 / 41.70 / 44.14 / 43.12        33.22 / 53.98 / 58.70 / 55.30        55.55   50.73
               3     29.84 / 41.22 / 42.58 / 41.70        33.52 / 53.75 / 58.06 / 54.97        54.79   49.78

## The matched pair from the same day: full (β = 128) vs β = 0, both with milestones

Ranges (n = 3) and same-seed paired differences full − β = 0 (mean ± SD).

    CIFAR-10   epoch 20    full  st1 [52.25, 54.34]  st2 [61.77, 68.09]  st3 [67.05, 70.88]  st4 [67.62, 70.83]
                           β=0   st1 [56.11, 56.87]  st2 [67.11, 67.81]  st3 [68.83, 70.10]  st4 [68.15, 69.89]
                           paired      −3.46 ± 1.49        −3.38 ± 3.81        −0.96 ± 2.02        −0.45 ± 1.70
               epoch 200   full  st1 [57.46, 60.00]  st2 [77.72, 80.90]  st3 [84.20, 85.61]  st4 [83.65, 85.54]
                           β=0   st1 [59.78, 60.61]  st2 [79.55, 79.92]  st3 [83.66, 83.83]  st4 [83.19, 83.53]
                           paired      −1.73 ± 0.97        −0.66 ± 1.75        +1.21 ± 0.63        +1.46 ± 1.16     SGD +1.34 ± 0.87
    CIFAR-100  epoch 20    full  st1 [29.70, 30.14]  st2 [42.89, 43.87]  st3 [45.15, 46.46]  st4 [44.03, 44.82]
                           β=0   st1 [29.84, 30.71]  st2 [40.73, 41.70]  st3 [41.78, 44.14]  st4 [40.77, 43.12]
                           paired      −0.32 ± 0.66        +2.27 ± 0.95        +3.14 ± 1.87        +2.69 ± 1.61
               epoch 200   full  st1 [33.09, 33.86]  st2 [54.82, 55.35]  st3 [58.55, 59.57]  st4 [55.72, 56.26]
                           β=0   st1 [33.22, 34.22]  st2 [53.75, 53.98]  st3 [57.66, 58.70]  st4 [54.90, 55.30]
                           paired      −0.11 ± 0.89        +1.20 ± 0.38        +0.89 ± 0.91        +1.00 ± 0.21     SGD +0.72 ± 0.54

P-C2 carried no prediction. At epoch 20 on CIFAR-10 the β = 0 arm is
above the full model at stages 1 and 2 (disjoint at stage 1; the full
model's stage-2 range is wide, 61.77–68.09) and the two overlap at
stages 3 and 4; at epoch 200 the ordering at stages 3 and 4 is the
archived one (full above by 0.5–2.2 paired). On CIFAR-100 at epoch 20
the full model is above β = 0 at stages 2, 3 and 4 with disjoint ranges
(paired +2.3, +3.1, +2.7), a larger separation than at epoch 200
(+1.2, +0.9, +1.0). Recorded, not interpreted; no convergence-speed
claim is made here.

Sources: `results/gate1/gate1_20260914_ridge_ms_beta0{,_c100}/units/product_endpoint__seed{1,2,3}/`
(`unit.json`, `blockwise_profile_epoch-0020.json`, `blockwise_profile_epoch-0200.json`);
full-model counterparts in `gate1_20260914_ridge_ms{,_c100}`. Training
started 2026-09-14 23:30–00:40 under commit `9f4580c`.
