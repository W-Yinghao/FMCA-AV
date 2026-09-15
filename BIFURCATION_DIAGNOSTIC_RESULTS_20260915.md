# What decides the epoch-2 branch — results, 2026-09-15

Against `BIFURCATION_DIAGNOSTIC_PREREG_FROZEN_20260914`. Record only,
plus one check that was not preregistered and is labelled as such. No
accuracy in this file is a result; the prereg says so and nothing here
changes that.

## Arms as run

A1 complete: `gate_bifurc_30k` and `gate_bifurc_45k`, `paper_composition`,
three seeds each, five epochs, per-level retained rank logged every
step (`step/retained_l{0,1,2}`), checkpoints at epochs 1/2/3/5. Steps
per epoch at batch 256: 117 (30k) and 176 (45k).
A2 (`gate_bifurc_45k_wu7`) was withdrawn before any reading (commit
9f4580c); P-D3 is not adjudicated.

## Per-step endpoint rank (`step/retained_l2`, of 128), current code

    split  seed  epoch 1, all steps  last step >= 60  first step < 30          from step 250  final l0/l1/l2
    30k    1     80-96               135              147 (epoch 2, step 30)   18-19          38/57/18
    30k    2     83-97               137              153 (epoch 2, step 36)   18-20          45/62/20
    30k    3     56-83               132              149 (epoch 2, step 32)   16-17          39/58/17
    45k    1     78-97               192              206 (epoch 2, step 30)   16-19          38/61/19
    45k    2     81-102              193              209 (epoch 2, step 33)   18-21          42/62/21
    45k    3     57-85               187              203 (epoch 2, step 27)   17-19          40/62/19

Levels 0 and 1 over the whole run: l0 29-55, l1 37-93, and they do not
move at the endpoint crash (30k seed 1: l0/l1 = 35/63 at step 146,
36/57 at step 147, while l2 goes 30 -> 26 -> 19 -> 18).

Epoch means of `train/retained_min`, the quantity the 200-epoch records
carry (epochs 1-5):

    30k   s1  33 23 18 18 18     s2  40 26 18 19 19     s3  47 25 16 16 17
    45k   s1  34 21 17 18 19     s2  42 24 19 20 21     s3  47 23 18 18 19

## Adjudication

**P-D1 REFUTED**, on both clauses. The endpoint level does not fall
below 30 before the end of epoch 1 in any 45k seed: it is 78-102 through
epoch 1 and falls 27-33 steps into epoch 2. And levels 0 and 1 are never
both above 60: level 0 is 29-55 in every run, so the "stay above 60"
clause could not hold in any run on disk, including the archived ones
that reach 128 at the minimum level. The threshold was mis-set.

**P-D2 REFUTED.** Every 30k seed's endpoint level falls to 16-20 at
steps 147-153 (epoch 2) and stays there. Under the current code the 30k
split does not take the branch the archived 30k runs took.

**P-D3** not adjudicated (arm withdrawn).

On the committed interpretation for "P-D1 refuted" — "the collapse is
not endpoint-specific": the refuting clause is the level-0/1 threshold,
and the measurements show the opposite of the committed reading — only
the endpoint level moves at the crash. That is a post-hoc observation
and is labelled as one.

Where the crash sits. In all six units the endpoint level holds 56-102
retained directions through the first epoch and for the first 27-36
steps of the second, then loses 60-70 directions within about 25 steps
and never recovers. The learning-rate schedule steps per epoch
(`LinearLR` warm-up, start factor 0.01 over 10 epochs, then cosine), so
the first schedule step — lr x 10.9 — is at the epoch-1/epoch-2 boundary
on both splits, and the crash follows it by ~30 steps on both. Recorded;
not tested.

## Not preregistered: the same five epochs under the pre-02fa55e code

Because P-D2 failed, the code was diffed. Commit `02fa55e`
(2026-09-11 01:52) replaced the multiview (leaf) term:

    before   trace_score(estimate_moments(f, leaf_views, centered=True), ridge=1e-3)
    after    truncated whiteners (tau = 1e-3) on pair-specific Grams built from
             individual view vectors; an empty retained set in either Gram
             invalidates the batch        (commit message cites Supplement §1.3)

That is the only training-path change in the commit. A worktree at its
parent `5322cdb` (`~/wt_5322cdb`; configs `gate_bifurc_old`,
`gate_bifurc_old45k`; roots `gate1_20260914_bifurc_{30k,45k}_oldcode_5322cdb`)
ran the same six units for five epochs. The old code logs only the
epoch mean of `train/retained_min`.

    code            split  seed   ep1  ep2  ep3  ep4  ep5
    5322cdb (old)   30k    1      33   39   39   39   40
    5322cdb (old)   30k    2      40   46   40   40   41
    5322cdb (old)   30k    3      47   44   39   40   41
    5322cdb (old)   45k    1      34   38   37   37   40
    5322cdb (old)   45k    2      42   44   40   41   44
    5322cdb (old)   45k    3      48   43   40   42   43

    archived C10 30k composition seed 1 (started before 02fa55e, 200 epochs):
                                   33   40   41   39   41   then 60 (ep 21), 105 (ep 51), 127 (ep 81-200)

Under the pre-02fa55e term both splits hold 37-48 through epoch 5 and
match the archived CIFAR-10 30k runs at the same epochs. Under the
current term both splits fall to 16-21 in epoch 2. Epoch-1 means are the
same under both codes for the same seed (30k: 33/40/47 vs 33/40/47;
45k: 34/42/48 vs 34/42/47).

SGD probe after five epochs, diagnostic only (the prereg forbids reading
these as results): old 30k [56.17, 57.39], new 30k [53.26, 54.48];
old 45k [58.13, 58.97], new 45k [56.15, 56.39].

## Which archived units ran which term

The `variant.startswith("paper_")` discriminator does not separate the
two multiview terms. The job start time (`hparams.yaml` mtime) against
2026-09-11 01:52 does, and the retained-rank signature agrees with it
on every unit. The term is only used when `leaf_reward_weight > 0`.

    unit(s)                                    started            term          train/retained_min, final
    C10 paper_composition s1-3                 09-10 21:50-22:42  pre-02fa55e   128, 128, 128
    C10 paper_beta0 s1-3                       09-11 01:50        pre-02fa55e   128, 128, 128
    C10 paper_alpha0 s1-3                      09-11 03:43-06:16  current       20, 21, 20
    C10 paper_T2 s1-3                          09-11 06:33-07:15  current       17, 18, 16
    C10 paper_T4, paper_K256, all seeds        09-12              current       18-21
    C10 star control (parallel children)       09-12              current       18 (certificate endpoint rank)
    C100 paper_composition s1-3                09-11 14:04        current       19, 20, 19
    C100 paper_beta0 s1-3                      09-11 14:04        current       47, 33, 51  (11-14 to ep ~50, then climbs)
    C100 paper_alpha0, paper_T2 s1-3           09-11 19:09-19:37  current       15-21
    45k main run, both datasets, s1-3          09-13              current       18-21
    C10/C100 paper_lambda0, paper_endpoint_only   --              leaf weight 0: term unused

The two pre-02fa55e groups are the only truncated units on disk that
ever reach 128. Every unit that used the current term with leaf weight
> 0 pins at 15-21 within two epochs, except the CIFAR-100 beta0 group,
which pins at 11-14 for about fifty epochs and then climbs to 33-51.
`paper_lambda0` (no multiview term under either code) sits at 39-43
through epoch 20 and ends at 45-65 (CIFAR-10) and 104-119 (CIFAR-100).

## What this changes in earlier records (corrections applied 2026-09-15)

1. `MAJOR_45K_RESULTS_20260914.md`: the 45k arm and the 30k arm it is
   compared with ran different multiview terms. P-M1/P-M2 stay refuted
   as stated — they predicted the 45k arm as run — but "config diff is
   exactly n_calibration, n_val, checkpoint_milestones" is true of the
   configs and false of the code, and the "unexplained bifurcation" is
   explained: code version, not data size.
2. The CIFAR-10 truncated ablation table (`PAPER_ESTIMATOR_RESULTS_20260911`,
   `MAJOR_EXPERIMENTS_AND_RESULTS_20260912` §1a-1b, consolidated §7)
   mixes terms: full and beta0 pre-02fa55e; alpha0 and T2 current;
   lambda0 and endpoint_only term-free. P1 (full vs beta0) and the
   lambda0 comparison are within one term; P2 and P3 (full vs alpha0,
   full vs T2), and T4/K256/star against full, are confounded.
3. Every CIFAR-100 truncated number is under the current term.
4. The consolidated write-up's §6, §8 and §9 rows on the truncated arms
   are revised; `SSL_ADDITIONAL_FINDINGS` item 1 is superseded.

## Not decided here

Which term is "the paper's" multiview term is a manuscript decision:
`02fa55e` implements Supplement §1.3 as written; the pre-02fa55e term
is what produced every retained-128 run and the 82.43-82.84 CIFAR-10
numbers. No 200-epoch run of the pre-02fa55e term at 45k exists, and
none is proposed here: the 45k prereg forbids re-running to find a
better number, and a run of the other term would need its own
preregistration stating which term it tests and why.
