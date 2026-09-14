# Server-side verification of the 14 September results draft

Checked against raw `unit.json` and `layerwise_profile.json` files on the
server, 2026-09-14. Companion to `SSL_RESULTS_SOURCES_20260914.md`; this
file records what verified, what did not, and what was added.

## 1. Verified exactly

| Draft number | Source | Status |
|---|---|---|
| Ridge C10 SGD 84.98 [83.63, 85.56], kNN 82.72 [81.40, 83.52] | v8 product_endpoint seeds 1-5 | matches |
| Ridge C10 stages 57.49 / 78.60 / 85.44 / 85.46, ranges as tabled | seeds 1-3 profiles | matches, **but see §2** |
| final_mview C10 SGD 88.97 [88.85, 89.19]; stage 2 68.53, stage 4 88.77 | seeds 1-5 / 1-3 | matches |
| Ridge C100 SGD 55.89 [55.44, 56.19], kNN 49.91, stages 33.40/54.97/58.73/56.06 | seeds 1-3 | matches **for seeds 1-3 only**, see §2 |
| final_mview C100 stage 2 43.84, stage 4 61.17 | seeds 1-3 | matches |
| All dagger values (six refreshed baselines, truncated 45k, rank trajectories, C100 seed-3 certificate) | `SSL_REFERENCE_RECIPES_RESULTS_20260914.md`, `MAJOR_45K_RESULTS_20260914.md` | match; raw files are on the server in `results/baseline_layerwise/ref_*` and `results/gate1/gate1_20260913_major45k*` |
| All 30k truncated tables (ablations, operators, features, configuration) | `MAJOR_EXPERIMENTS_AND_RESULTS_20260912.md`, `results/paper_certificate/` | match |
| Ridge configs: calibration 2500 + validation 2500, 200 epochs → 45,000 images | `hparams.yaml`, `configs/gate_v8` | confirmed |

**Evaluator identity.** The ridge stage profiles (23 Aug, `layerwise_profile_20260823_v1`) and the refreshed baseline profiles (14 Sep) use the same `convex_probe` (byte-identical across those commits), the same 50,000-image probe set, and the same pooled-after-each-stage taps. One archived ridge profile was recomputed today with the current code on the same checkpoint: all four stages **bit-identical** (57.23 / 79.55 / 85.60 / 85.58). The draft's "evaluator matching unresolved" note for Table `sslres:baselines` can be closed.

## 2. Discrepancies

**2a. Ridge C10 stage ranges are n = 3 in the draft; n = 5 is now available.** Seeds 4 and 5 had no profile; they do now. The stage-4 lower bound moves.

| Stage | Draft (n=3) | All five seeds |
|---|---|---|
| 1 | 57.49 [57.23, 57.78] | 57.96 [57.23, 59.50] |
| 2 | 78.60 [77.72, 79.55] | 79.35 [77.72, 80.69] |
| 3 | 85.44 [85.12, 85.61] | 85.14 [83.96, 85.61] |
| 4 | 85.46 [85.25, 85.58] | **85.05 [83.80, 85.58]** |

Against Barlow Twins' 82.43–82.90 at stage 4 the ranges remain disjoint, by 0.90 rather than 2.35. The draft text "little further change in the mean at stage 4" still holds (85.14 → 85.05). The sentence "the archived ridge stagewise collection does not cover every model included in its final-stage SGD summary" is no longer true and can be dropped.

**2b. Ridge C100 has four complete seeds, not three.** Seed 5 is a complete 200-epoch unit (SGD 55.57, kNN 50.38); seed 4 failed at CUDA driver initialization and is not a result. With n = 4: SGD 55.81 [55.44, 56.19] (range unchanged), kNN 50.03 [49.81, 50.38], stages 33.60 / 54.97 / 58.74 / 55.98.

**2c. final_mview C100 has five complete seeds.** The draft's 61.46 [61.16, 61.96] is seeds 1-3. All five: SGD 61.34 **[61.12, 61.96]**, kNN 57.70 [57.50, 58.24]; stages 30.84 / 43.91 / 60.35 / 61.14. The draft's lower bound 61.16 is wrong by 0.04.

**2d. final_mview C10 stage means at n = 5**: 55.91 / 68.56 / 84.78 / 88.78 (draft 68.53 / 88.77 at n = 3; rounding-level).

## 3. The matched ridge composition ablation exists

The ledger states the beta = 0 ridge ablation is "absent locally … a template is not a completed ablation". It is complete on the server, three seeds per dataset, config diff against the full model exactly `loss.beta: 128 → 0`, checkpoints present. Stage profiles were computed today.

| Arm | Stage 1 | Stage 2 | Stage 3 | Stage 4 | SGD |
|---|---|---|---|---|---|
| C10 full (β=128, n=5) | 57.23–59.50 | 77.72–80.69 | 83.96–85.61 | 83.80–85.58 | 83.63–85.56 |
| C10 β=0 (n=3) | 59.27–60.27 | 79.91–80.10 | 83.91–83.97 | 83.41–83.63 | 83.55–83.74 |
| C100 full (n=4) | 33.08–34.20 | 54.81–55.13 | 58.57–58.94 | 55.75–56.33 | 55.44–56.19 |
| C100 β=0 (n=3) | 33.22–33.91 | 53.38–53.98 | 58.06–58.70 | 54.95–55.30 | 54.79–55.55 |

On CIFAR-10 the stage-2 ranges overlap (β=0 sits inside the full model's range) and the full model is above β=0 at stage 4 by 0.17 at the nearest edges. On CIFAR-100 the full model is above β=0 at every stage from 2 on, by 0.83 at stage 2 and 0.45 at stage 4 at the nearest edges. The draft's claim–evidence row "composition alone causes the ridge shallow-feature advantage — not claimed" now has data behind it: on CIFAR-10 the intermediate-stage advantage over external methods is present without the composition penalty.

Also profiled today, same protocol (CIFAR-10, n = 3 each), single-factor against the full model:

| Arm | Diff from full | Stage 2 | Stage 4 | SGD |
|---|---|---|---|---|
| α=0 (no adjacent-stage objective) | `loss.alpha 0.2 → 0` | 80.55–81.11 | 83.64–84.17 | 83.44–84.14 |
| M_end = 4 | `endpoint_descendants 8 → 4` | 75.05–80.67 | 82.29–85.03 | 82.11–85.10 |
| parallel children (star) | `view_tree.mode` | 81.77–82.07 | 84.19–84.83 | 83.99–84.60 |
| C100 parallel children | same | 55.20–56.06 | 57.03–58.19 | 56.52–57.18 |

Under the ridge recipe the parallel-children control is not below the full model at any stage on either dataset. This is recorded, not interpreted.

## 4. Submitted today

Twenty stage-granularity profile jobs (the 19 above plus one recheck), all landed. Twenty of twenty first failed with CUDA out-of-memory: `stage_features` in `run_layerwise_profile.py` lacked `@torch.no_grad()`, so the autograd graph accumulated across the 50,000-image pass; the archived profiles had only ever run on larger cards. Fixed in commit `cf46a7c`; the recheck in §1 shows the fix changes no value.

## 5. Not run, and why

- **Epoch-20 checkpoints for the ridge method.** The archived ridge runs (Aug 20-22) saved only `last.ckpt`. The draft as written contains no epoch-20 ridge number, so none was fabricated and no retrain was started. A retrain with milestones is 3 seeds × 2 datasets × ~7 h ≈ 42 GPU-hours and would produce a new set of runs, not the archived ones.
- **C100 ridge seed 4** failed at driver initialization; a rerun (~7 h) would bring CIFAR-100 to n = 5. Not started.
- **Certificates for ridge arms**: the operator diagnostics in the draft are truncated-coordinate only, by design; nothing to add.

## 6. Added after §5 was written (same day)

- **Ridge epoch-20 now exists.** Six new units (3 seeds × 2 datasets),
  config diff `checkpoint_milestones` only. P-R1 confirmed: every
  epoch-200 range overlaps the archived one. Epoch-20 stage profiles are
  in `RIDGE_MILESTONE_REPLICATION_RESULTS_20260914.md`.
- **CIFAR-100 ridge is n = 5.** Seed 4 rerun complete (SGD 55.64).
  Replacement numbers: SGD 55.78 [55.44, 56.19]; kNN 50.04 [49.81, 50.38];
  stages 33.50 / 55.07 / 58.91 / 56.06 with ranges [33.08, 34.20],
  [54.81, 55.48], [58.57, 59.59], [55.75, 56.39]. The §2b numbers (n = 4)
  are superseded.
