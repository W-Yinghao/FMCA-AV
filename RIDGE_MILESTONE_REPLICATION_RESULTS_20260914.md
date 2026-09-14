# Ridge full model, replicated with epoch-20/200 checkpoints — results

Against `RIDGE_MILESTONE_REPLICATION_PREREG_FROZEN_20260914`. Record only.
Six units, all complete, no failures. `product_endpoint`, ridge estimator,
EMA endpoint target, 45,000 training images, 200 epochs, three seeds per
dataset, config diff against the archived templates exactly
`checkpoint_milestones: [20, 200]`. Stage values below are the stage-end
taps of the block profile (layer1.1 / 2.1 / 3.1 / 4.1), which are the
same objects the archived stage profiles read.

## P-R1 — replication of the archived ranges at epoch 200: CONFIRMED

    dataset  measure     new (n=3)         archived          
    CIFAR-10 SGD probe   [83.91, 85.30]    [83.63, 85.56] (n=5)   overlap
    CIFAR-10 stage 2     [77.72, 80.90]    [77.72, 80.69] (n=5)   overlap
    CIFAR-10 stage 4     [83.65, 85.54]    [83.80, 85.58] (n=5)   overlap
    CIFAR-100 SGD probe  [55.44, 56.19]    [55.44, 56.19] (n=5)   overlap
    CIFAR-100 stage 2    [54.82, 55.35]    [54.81, 55.48] (n=5)   overlap
    CIFAR-100 stage 4    [55.72, 56.26]    [55.75, 56.39] (n=5)   overlap

All six ranges overlap their archived counterparts. The August runs
reproduce under the current code, and the two collections may be pooled
or reported side by side.

## P-R2 — epoch 20 (diagnostic, no prediction)

    CIFAR-10   epoch 20   st1 [52.25, 54.34]  st2 [61.77, 68.09]  st3 [67.05, 70.88]  st4 [67.62, 70.83]
               epoch 200  st1 [57.46, 60.00]  st2 [77.72, 80.90]  st3 [84.20, 85.61]  st4 [83.65, 85.54]
    CIFAR-100  epoch 20   st1 [29.70, 30.14]  st2 [42.89, 43.87]  st3 [45.15, 46.46]  st4 [44.03, 44.82]
               epoch 200  st1 [33.09, 33.86]  st2 [54.82, 55.35]  st3 [58.55, 59.57]  st4 [55.72, 56.26]

For comparison at the same taps and epochs, the truncated 45k arm read
CIFAR-10 epoch 20 st2 [68.75, 70.99] / st4 [69.40, 71.96] and epoch 200
st2 [77.96, 78.14] / st4 [79.55, 79.86]; the reference-recipe baselines at
epoch 20 read st4 between 56.32 (SimSiam) and 71.73 (VICReg).

## CIFAR-100 archived collection, now n = 5

Seed 4, failed on 21 August at CUDA driver initialization, was rerun into
its archived root on 14 September under current code; its failed record
is preserved beside it as `unit.json.failed_20260821`. With five seeds:

    SGD  55.78 [55.44, 56.19]    kNN 50.04 [49.81, 50.38]
    stage 1  33.50 [33.08, 34.20]     stage 2  55.07 [54.81, 55.48]
    stage 3  58.91 [58.57, 59.59]     stage 4  56.06 [55.75, 56.39]

The seed-4 rerun (SGD 55.64) sits inside the range of the four August
seeds.
