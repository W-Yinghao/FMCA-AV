# Frozen before compute — what decides the epoch-2 branch in truncated training

Frozen 2026-09-14. Diagnostic wave. **No accuracy from any arm here is a
result**; the frozen grid for the 45k main run forbids re-running with
a different schedule to find a better number, and this wave does not do
that. It asks which factor decides the branch.

## The fact under study

Every truncated run starts at 34–48 retained directions. By epoch 2 a
run is at 41–50 and climbs to 128 (CIFAR-10, 30k split) or at 17–21 and
pinned (every other configuration). The 45k training set is a strict
superset of the 30k set; hyperparameters are identical; steps per epoch
are 176 vs 117, so the warm-up covers 1760 vs 1170 steps at the same
per-epoch learning-rate values.

## Arms

A1. `gate_bifurc_30k`, `gate_bifurc_45k`: the two CIFAR-10 splits,
    three seeds each, FIVE epochs, per-level retained ranks logged every
    step (`step/retained_l{0,1,2}`), checkpoints at epochs 1, 2, 3, 5.
    Config diff against the corresponding full run: max_epochs,
    milestones, the two logging flags.
A2. `gate_bifurc_45k_wu7`: the 45k split with `warmup_epochs` 10 → 7, so
    warm-up spans 1232 steps against the 30k schedule's 1170. Three
    seeds, 200 epochs, milestones 20/200. Nothing else changes.

## Predictions

P-D1. In every 45k seed the ENDPOINT level (`retained_l2`) falls below 30
      before the end of epoch 1 while levels 0 and 1 stay above 60 through
      epoch 5. Rationale: the 2500-root certificates read
      [92,115,20]-shaped ranks, so the collapse is at the endpoint level.
      REFUTED if any seed's l2 stays ≥ 30 through epoch 1, or if l0/l1
      also fall below 60.
P-D2. In every 30k seed `retained_l2` stays ≥ 35 through epoch 5.
      REFUTED if any seed falls below 35.
P-D3. With warm-up aligned by steps, at least two of three 45k seeds reach
      retained_min ≥ 100 by epoch 100.
      REFUTED if all three are ≤ 25 at epoch 100.

## Interpretation grid, committed now

    P-D3 holds     the number of optimizer steps taken during warm-up decides
                   the branch; the 45k main run's lower result is a schedule
                   artefact.  This does NOT promote the wu7 accuracy: a
                   schedule chosen after seeing the branch is a tuned schedule
                   and would need its own preregistered confirmation.
    P-D3 refuted   warm-up steps are not the cause; the remaining candidate is
                   the composition of the first epochs' data.  Reported as
                   unresolved.
    P-D1 refuted   the collapse is not endpoint-specific and the certificate
                   reading was misleading about training time.
