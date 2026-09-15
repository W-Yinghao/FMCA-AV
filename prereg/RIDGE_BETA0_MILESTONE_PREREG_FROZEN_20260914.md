# Frozen before compute — ridge β = 0 replicated with epoch-20 checkpoints

Frozen 2026-09-14. Companion to `RIDGE_MILESTONE_REPLICATION_PREREG_FROZEN_20260914`.

The archived β = 0 ridge units (the matched composition ablation) have
only `last.ckpt`. This retrains the same configuration — config diff
against `gate_v8_beta0` / `gate_c100_beta0` is exactly
`checkpoint_milestones: [20, 200]` — three seeds per dataset, so the
ablation's early depth profile can sit beside the full model's.

P-C1. Epoch-200 ranges overlap the archived β = 0 ranges: CIFAR-10 SGD
      [83.55, 83.74], stage 2 [79.91, 80.10], stage 4 [83.41, 83.63];
      CIFAR-100 SGD [54.79, 55.55], stage 2 [53.38, 53.98], stage 4
      [54.95, 55.30]. REFUTED if any of the six is disjoint.
P-C2. Epoch 20: no prediction.
