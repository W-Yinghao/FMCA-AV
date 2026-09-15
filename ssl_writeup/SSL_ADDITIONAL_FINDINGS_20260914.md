# Findings beyond the manuscript's frame

2026-09-14. Things the results wave established that the paper as
organized does not use, or uses only in passing, and that I judge worth
keeping in view. Each item names its evidence. Where an item is my
reading rather than a measurement, it says so.

## 1. The truncated formulation has a two-branch training dynamic, and only one configuration ever reached the good branch

Every truncated run starts with 34–48 of 128 projected directions
retained. By epoch 2 a run is either at 41–50 and climbing (reaching
128 by epoch 100) or at 17–21 and pinned there for the remaining 195
epochs. Across every truncated configuration on disk — two datasets,
two data splits, six ablation arms, T = 4, K = 256, the star control —
**only `paper_composition` and `paper_beta0` on CIFAR-10 at the
30,000-image split took the upper branch.** Adding 15,000 images of the
same distribution (a strict superset, same seeds, same hyperparameters)
moved the CIFAR-10 full model to the lower branch and cost 4 points at
the final stage (82.43–82.84 → 78.32–78.47).

**Superseded 2026-09-15.** The branch is decided by the code version
of the final-stage multiview term, not by the data. Commit `02fa55e`
(11 Sep 01:52) replaced that term (ridge-1e-3 trace score → truncated
whiteners on pair-specific Grams, Supplement §1.3). The CIFAR-10
composition and β = 0 units — the only ones that ever reached 128 —
started before it; every other truncated unit with a multiview term
started after it. Under per-step logging the current term drops the
endpoint level from 56–102 to 16–21 about 30 steps into epoch 2 on both
splits and all six seeds, with levels 0 and 1 unchanged; the parent
commit's code holds 37–48 through epoch 5 on both splits. The 45k arm's
4-point deficit is therefore an implementation difference. Which term
is the paper's is a manuscript decision; the 200-epoch result of the
pre-`02fa55e` term at 45k does not exist. The prereg's P-D1 and P-D2
were both refuted (P-D1's level-0/1 threshold was mis-set; the 30k
split under the current code also crashes).

Evidence: `BIFURCATION_DIAGNOSTIC_RESULTS_20260915.md`;
`MAJOR_45K_RESULTS_20260914.md` (corrected); `git show 02fa55e`.

## 2. The ridge recipe does not depend on the composition penalty for its intermediate-stage advantage, and the star control is not worse

Under ridge, β = 0 lands inside the full model's stage-2 range on
CIFAR-10 (79.91–80.10 vs 77.72–80.69) and α = 0 and the parallel-
children control are above it (80.55–81.11 and 81.77–82.07). On CIFAR-100
the full model is above β = 0 by 0.4–0.8 at each stage, but the parallel
control is above the full model at every stage.

Reading: what produces the 9-point stage-2 gap over external methods is
training an objective at the intermediate readouts at all, which every
ridge variant does. The Markov-nesting motivation in the main section
("nested children are sampled from the realized parent … this sampling
procedure defines the Markov chain") is a modelling choice; on these
measurements it is not an accuracy advantage in the ridge recipe. Under
the truncated recipe the star control did degrade accuracy and collapsed
its endpoint rank to 18 with endpoint norm 3.9 — but that recipe's
sensitivity (item 1) makes the attribution weak.

Evidence: `SSL_NUMBERS_VERIFICATION_20260914.md` §3;
`SSL_PAPER_RESULTS_CONSOLIDATED_20260914.md` §5.

## 3. Estimator and recipe interact; neither is "better" on its own

The 2×2 on CIFAR-10 (final-stage SGD probe):

    recipe \ estimator     ridge                 truncated
    ridge/V8 recipe        83.63–85.56 (n=5)     81.17–81.90 (n=3)
    paper recipe           80.07–81.34 (n=3)     82.43–82.84 (n=3)

The V8 recipe (EMA target, bootstrap composition) wins under ridge and
loses under truncation; the paper recipe does the opposite. The frozen
prediction that the gap was mostly recipe was refuted. Caveat, mine:
the ridge-recipe/truncated cell trained on 37,500 images (I matched the
calibration split but not the validation split), so that cell is not
data-matched to the other three; the direction of the interaction does
not depend on it, its magnitude might.

Evidence: `results/gate1/gate1_20260912_x2x2/`;
`MAJOR_COMPLETION_WAVE_PREREG_FROZEN_20260912.md`.

## 4. Effective rank is not a collapse test

The BYOL run that collapsed (accuracy falling with depth, cosine 0.99)
has the highest final-stage backbone effective rank of any baseline
(58.5, vs 44.3 SimCLR, 33.6 VICReg, 30.5 fixed BYOL). High entropy rank
is what noise looks like. Conversely `endpoint_only` (effective rank
1.04) and K = 256 (1.01–4.09) both decode monotonically with depth to
77–79 %. The discriminating measure is whether probe accuracy rises with
depth; effective rank and trace describe conditioning. Two of my own
earlier readings ("degenerate", "not a usable representation") were
withdrawn on this basis.

Evidence: `EFFECTIVE_RANK_IS_NOT_A_COLLAPSE_TEST_20260913.md`.

## 5. The baselines were never configured as themselves, and the cheapest fix exposed which methods it mattered for

One copy-pasted objective/model block served all six methods. Only BYOL
collapsed, because only BYOL's collapse avoidance *is* the missing
BatchNorm; the other five absorbed the same defect behind their loss
terms. With reference projectors, Barlow Twins and VICReg improved
(disjoint ranges) while SimCLR and MoCo v2 did not — MoCo v2 at its own
momentum of 0.999 is up to 0.9 lower than at the 0.996 fallback. The
optimizer remains a single lr 0.03 for all six. A MoCo v2 at ~84 % on
CIFAR-10 (the user's earlier run) is not reproduced here and should not
be expected to be at this learning rate and queue.

Reading: the external comparison is internally consistent (one evaluator,
one budget) and externally understated by an unknown amount for the
three high-learning-rate methods. A per-method learning-rate wave, with
its selection criterion frozen before looking, is the honest next step
if the comparison is to carry weight beyond "same ruler".

Evidence: `BYOL_COLLAPSE_DIAGNOSIS_20260913.md`;
`SSL_REFERENCE_RECIPES_RESULTS_20260914.md`.

**Update 2026-09-15.** The learning-rate wave ran (P-L1 confirmed:
SimCLR 0.1, Barlow Twins 0.1, VICReg 0.3; MoCo v2 0.3). Stage 4 rose
4.2–6.0 points for all four, and the understatement was large enough to
flip the final-stage ordering: FMCA-AV (ridge) is now below VICReg,
Barlow Twins and SimCLR at stage 4 with disjoint ranges, and above all
seven only at stages 2 and 3. MoCo v2 at 83.4–83.8 is now in line with
the user's earlier 83.8. Evidence: `BASELINE_LR_SELECTION_RESULTS_20260915.md`.

## 6. The manuscript's earlier "45,000 images" claim was false on our side

Every `gate_paper` (truncated) unit trained on 30,000 images because the
10,000 + 10,000 reserve was silently subtracted from 50,000, while the
baselines trained on 45,000. The 14 September reorganization already
corrects this; it is recorded here because the earlier results file
(`PAPER_ESTIMATOR_RESULTS_20260911.md`) still carries the wrong label in
its like-for-like table and should not be cited for that table.

## 7. Runs are not reproducible unit-by-unit

`deterministic="warn"` plus a launcher spanning A100/H100/L40S: the same
unit rerun landed on a different architecture and was bit-identical on
none of 55 overlapping epochs (H100 vs A100). Three-seed ranges are the
right unit of evidence here, and a "rerun" is a new sample, not a
replication. The ridge replication wave (six new seeds, all ranges
overlapping the archived ones) is the right kind of reproducibility
claim for this repo.

## 8. Certificates at 2,500 calibration roots are a different instrument

The 45,000-image split leaves 2,500 roots for Stage-B calibration. Those
certificates read retained ranks of 17–21 at the endpoint where the
10,000-root ones read 128, and one CIFAR-100 seed reads endpoint rank 1
with relative error 3.74 on a representation that decodes normally. Any
operator claim made at 45,000 images either needs its own calibration
design (e.g. calibration drawn from the pretraining images with disclosed
overlap) or should be made at the 30,000-image split only.

## 9. Preregistration status, so nothing is over-cited

    preregistered, adjudicated     truncated 5-arm ladder (P1 confirmed narrowly, P2 half refuted, P3 confirmed);
                                   MAJOR 45k main run (P-M1/P-M2 refuted); ridge replication (P-R1 confirmed);
                                   BYOL BatchNorm (P-B1 half, P-B2 confirmed); reference recipes (P-R1 confirmed,
                                   P-R2 two of four); 2×2 (P1 refuted); star under truncation (P3 precondition failed)
    frozen after completion,       endpoint_only (P-E1 refuted; comparator anisotropic but functional)
    before reading
    descriptive only               T = 4, K = 256 (aggregates seen before any reading rule);
                                   ridge α = 0, M_end = 4, parallel (no prereg; profiles only)
    added 15 September             bifurcation diagnostic (P-D1, P-D2 refuted; P-D3 arm withdrawn;
                                   old-code check NOT preregistered); baseline learning rates
                                   (P-L1 confirmed; P-L3 rows replaced); ridge β = 0 milestones
                                   (P-C1 confirmed, 15 Sep)

## 10. Infrastructure facts that bear on the numbers

- Checkpoints were written only at the end of `fit()` until 13
  September; a killed run left nothing. Every unit in this report either
  completed inside one allocation or was trained after the fix.
- `stage_features` in the profiler ran without `no_grad` and exceeded a
  40 GB card; the archived profiles came from larger cards. Values are
  unaffected (bit-identical recomputation).
- The divergence guard bounds whitened objective terms only; `collapsed`
  in unit records means test accuracy < 15 %. Neither detects a scale
  runaway or an anisotropic representation.

## 11. Open questions I would put compute against, in order

1. ~~What decides the epoch-2 branch~~ — done 15 September: the
   multiview-term code version (item 1).
2. ~~Per-method learning rates~~ — done 15 September (item 5).
3. ~~A ridge β = 0 with epoch-20 checkpoints~~ — done 15 September
   (`RIDGE_BETA0_MILESTONE_RESULTS_20260915.md`; P-C1 confirmed).
4. New, and a decision before compute: which multiview term the
   manuscript's truncated formulation means. If the pre-`02fa55e`
   term, its 45k/200-epoch result does not exist and would need its own
   preregistration; if the current term, the CIFAR-10 30k rows are not
   that formulation's result.
