# 服务器确认结果：对 `SERVER_CONFIRMATION_CHECKLIST.md` 的逐项核查

核查日期 2026-09-15（05:30–06:00 CEST）。基准提交 `ae02ffe`（清单指定）；核查过程未改动任何训练、评价或结果文件，只新增了本目录。所有原始值直接从服务器上的 `unit.json`、`layerwise_profile.json`、`blockwise_profile_epoch-0200.json`、`results/baseline_layerwise/*.json`、`results/validity/*.json`、`hparams.yaml` 与检查点读出。

导出文件（同目录）：

| 文件 | 内容 |
|---|---|
| `performance_final.csv` | 主表每个单元格：数据集、方法、目标、优化图像数、epochs、backbone、取特征层、投影配置、评价器、原始小数、模型标识、训练代码提交、评价代码版本、重复编号；每个最大值附比较组 n / 均值 / 样本 SD / 范围 |
| `performance_sources.json` | 每个比较组的逐次评价（文件路径、检查点、seed、四个 stage 原始小数、SGD 探针） |
| `ridge_beta0_matched.csv`、`ridge_beta0_matched_summary.csv` | 已有匹配 β=0 对照：逐 seed 的 Stage 1–4 / SGD / kNN，均值、SD、同 seed 成对差 |
| `gram_analysis_final.csv` | Gram 表六个三元组 + 两份 45,000 图像模型的扩展分析，含指纹复核、校准池、评价集、阈值、保留维数 |
| `split_indices.json` | bigcal seed-1 切分的全部图像索引（校准 / 验证 / 训练）及 sha256；45k 五个 seed 的切分哈希；SGD 探针验证集哈希；学习率选择 holdout 的 5,000 个索引 |
| `experiment_manifest.json`、`truncated_units_manifest.csv` | 工程标识：评价器定义、EMA 行为、level_role 问题、比较组成员、β=0 复现状态、基线运行目录与提交、69 个截断单元的数学实现归类 |

论文所需数值只在 `performance_final.csv`、`ridge_beta0_matched*.csv`、`gram_analysis_final.csv`；工程标识只在 manifest。

---

## 1. 性能表最佳值

**统一口径**：CIFAR-10，CIFAR-adapted ResNet-18（宽度 64），45,000 张优化图像（n_calibration = n_val = 2,500），200 epochs，标准化逐层线性评价（`convex_probe`：按坐标标准化，零初始化多项逻辑回归，L-BFGS ≤ 200 步，权重罚 1e-4，在全部 50,000 张训练图上拟合，10,000 张测试图上打分）。评价函数 `convex_probe` 的源码 md5 `fc9e0f8b…` 自首次提交 `8637c7e`（8 月 23 日）到 `ae02ffe` 未变；gate 单元的记录版本 `layerwise_profile_20260823_v1`，基线记录版本 `baseline_layerwise_v1`，后者直接 import 前者的函数。

### 1a. 清单重点单元格：全部核实

| 单元格 | 清单值 | 核实值（原始小数） | 来源 | 比较组 n / 均值 / SD / 范围 |
|---|---:|---|---|---|
| final-level multiview，Stage 2 | 69.04 | 0.69040 | v8 `final_mview__seed5` | 5 / 68.56 / 0.33 / [68.16, 69.04] |
| final-level multiview，Stage 3 | 84.93 | 0.84930 | v8 `final_mview__seed3` | 5 / 84.78 / 0.14 / [84.62, 84.93] |
| final-level multiview，Stage 4 | 88.90 | 0.88900 | v8 `final_mview__seed3` | 5 / 88.78 / 0.09 / [88.64, 88.90] |
| Hierarchical，Stage 2 | 80.90 | 0.80900 | `gate1_20260914_ridge_ms/product_endpoint__seed1`（9 月 14 日复现，L40S→A100，与归档 seed 1 是不同样本） | 8 / 79.24 / 1.29 / [77.72, 80.90]；仅归档 5 次：79.35 / [77.72, 80.69] |
| Hierarchical，Stage 3 | 85.61 | 0.85610 | v8 `product_endpoint__seed2`（复现 seed 2 同值，见 §5） | 8 / 85.08 / 0.65 / [83.96, 85.61] |
| Hierarchical，Stage 4 | 85.58 | 0.85580 | v8 `product_endpoint__seed1` | 8 / 84.96 / 0.78 / [83.65, 85.58] |
| BYOL，Stage 4 | 74.76 | 0.7475999593734741 | `byol_bn_seed1.json`（旧模板投影头 + BN，lr 0.03，`last.ckpt`） | 3 / 74.27 / 0.70 / [73.46, 74.76]。参考配方 BYOL 另一组：3 / 73.34 / 0.54 / [72.91, 73.95] |
| VICReg，Stage 4 | 88.61 | 0.88610 | `reflr_vicreg_lr0.3_seed2_epoch-0200` | 3 / 88.41 / 0.19 / [88.23, 88.61] |
| Barlow Twins，Stage 4 | 87.72 | 0.87720 | `reflr_barlow_twins_lr0.1_seed1_epoch-0200` | 3 / 87.51 / 0.18 / [87.40, 87.72] |
| SimCLR，Stage 4 | 86.36 | 0.86360 | `reflr_simclr_lr0.1_seed1_epoch-0200` | 3 / 86.20 / 0.16 / [86.05, 86.36] |
| MoCo v2，Stage 4 | 83.84 | 0.83840 | `reflr_moco_v2_lr0.3_seed3_epoch-0200` | 3 / 83.66 / 0.22 / [83.42, 83.84] |

"74.75" 确为截断写法；四舍五入应为 74.76。注意这一格取的是**旧模板投影头 + BN** 的 BYOL，不是参考配方 BYOL；两组都在 CSV 中，标题需说明取的是哪一组。

最终层 SGD 探针（独立表）：CIFAR-10 final_mview 89.19（v8 seed 2；5 次 88.97，[88.85, 89.19]）；CIFAR-100 final_mview 61.96（c100pilot seed 3；5 次 61.34，[61.12, 61.96]）；Hierarchical CIFAR-10 85.56（seed 1；84.98，[83.63, 85.56]），CIFAR-100 56.19（seed 2；55.78，[55.44, 56.19]）。评价器不同（SGD 100 epochs、裁剪翻转、45,000/5,000 拆分），不得与逐层表混排。

### 1b. 同预算、同评价器的全部已完成 FMCA-AV 配置（CIFAR-10，45,000 图，200 ep，逐列最大值）

| 配置（ridge 估计器） | n | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---:|---:|---:|---:|---:|
| Hierarchical 完整（归档 5 + 复现 3） | 8 | 60.00 | 80.90 | 85.61 | 85.58 |
| final-level multiview | 5 | 56.23 | 69.04 | 84.93 | **88.90** |
| parallel children（star 对照） | 3 | 59.80 | **82.07** | 84.83 | 84.83 |
| α = 0 | 3 | 60.07 | 81.11 | 84.41 | 84.17 |
| M_end = 4 | 3 | 60.44 | 80.67 | 85.28 | 85.03 |
| β = 0 | 3 | 60.27 | 80.10 | 83.97 | 83.63 |
| final_2view | 3 | 55.53 | 68.37 | 82.76 | 85.54 |
| amdim_cross | 3 | 60.45 | 76.88 | 79.28 | 78.79 |
| additive_mview | 3 | 59.67 | 76.45 | 78.56 | 78.74 |
| additive_2view | 3 | 60.94 | 75.90 | 76.90 | 76.77 |
| product_only | 3 | 47.44 | 64.70 | 68.71 | 67.31 |
| 截断估计器主跑（45k，当前多视图项） | 3 | 58.84 | 78.14 | 80.94 | 79.86 |

结论：Stage 3、4 没有比 Hierarchical 更高的同目标配置；Stage 4 的更高值只有 final-level multiview（已单列）。Stage 2 上 parallel children（82.07）和 α = 0（81.11）高于 Hierarchical 的 80.90，Stage 1 上 additive_2view / amdim / M4 高于 60.00——这些都是消融或对照配置，若进主表必须另列方法行。

### 1c. 原 17 页论文数字

CIFAR-10 90.3%、CIFAR-100 70.3%（800 ep）、ImageNet-100 61.7%：`results/` 下没有任何对应记录；服务器上只有 `code_remote/IMAGENET-Copy1.ipynb`、`IMAGENET_RESNET_152.ipynb` 和仓库根目录 `IMAGENET_FMCA_run_new_results.ipynb`（引用外部路径 `/home/UFAD/hubo/imagenet/imagenet-100/`），未找到这三个数值本身。本次未核实，不应纳入。

---

## 2. Ridge β=0

**已有匹配对照（本稿使用）**：全部核实，均值与清单一致（n = 3，seed 1–3）。

| | Stage 1 | Stage 2 | Stage 3 | Stage 4 | SGD | kNN |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 完整（v8 s1–3） | 57.49 ± 0.28 | 78.60 ± 0.92 | 85.44 ± 0.28 | 85.46 ± 0.18 | 85.36 ± 0.18 | 83.15 ± 0.33 |
| CIFAR-10 β=0（v8_beta0 s1–3） | 59.87 ± 0.53 | 80.00 ± 0.10 | 83.94 ± 0.03 | 83.53 ± 0.11 | 83.68 ± 0.11 | 81.22 ± 0.14 |
| 成对差 完整 − β=0 | −2.38 ± 0.62 | −1.40 ± 1.01 | +1.50 ± 0.25 | +1.93 ± 0.22 | +1.68 ± 0.18 | +1.94 ± 0.35 |
| CIFAR-100 完整（c100pilot s1–3） | 33.40 ± 0.30 | 54.97 ± 0.16 | 58.73 ± 0.19 | 56.06 ± 0.28 | 55.89 ± 0.40 | 49.91 ± 0.15 |
| CIFAR-100 β=0（c100_beta0 s1–3） | 33.55 ± 0.35 | 53.70 ± 0.30 | 58.30 ± 0.35 | 55.07 ± 0.20 | 55.24 ± 0.40 | 50.05 ± 0.59 |
| 成对差 | −0.15 ± 0.65 | +1.27 ± 0.46 | +0.43 ± 0.50 | +0.99 ± 0.14 | +0.65 ± 0.60 | −0.14 ± 0.68 |

配置差异核实：`configs/gate_v8` vs `gate_v8_beta0`、`gate_c100` vs `gate_c100_beta0`，扁平化后仅 `loss.beta: 128 → 0`（另有 experiment.name / status 文字）。训练提交：完整 C10 s1–3 `1c747ea`/`63071af`（8 月 20 日），β=0 s1–3 `b68d329`（8 月 24 日）；C100 完整 `648c54e`（8 月 21 日），β=0 `b68d329`。归档 β=0 的 GPU：seed 1 H100 NVL，seed 2、3 A100-PCIE。

**新里程碑复现（`gate1_20260914_ridge_ms_beta0`、`_c100`）**：核查时 **0/6 完成**，六个任务在跑（CIFAR-10 s1–3 epoch 173/174/182，CIFAR-100 s1–3 epoch 153/154/114，共 200；A100/L40S 混合节点）。配置差异核实：`gate_ridge_ms` vs `gate_ridge_ms_beta0`（及 c100）仅 `loss.beta`；两者相对归档配置只多 `checkpoint_milestones: [20, 200]`。配对对象是同批 `gate1_20260914_ridge_ms(_c100)` seed 1–3（已完成）。完成后 sweep 会自动出 epoch-20/200 的逐层评价，届时整组替换，不单挑单元格；本文档不含第 20 epoch 的任何结论。后台任务会在落盘后打印 P-C1 对照。

**须知（§5 详述）**：9 月 14 日的完整模型复现在同 GPU 家族上几乎复制了归档权重（4/6 单元 max|Δw| < 1.3e-2，探针值到小数点后四位相同），只有 2/6 是真正独立的新样本。β=0 复现若同样落在 A100 上，可能同样近似复制归档 β=0 的 seed 2、3。

---

## 3. Gram 表

**六个三元组全部核实**（`curve["20000"]["gram"]["projection"]`，绝对误差 / 端点范数 / 相对误差）：

| 数据集 | 分层 | 仅相邻目标 | 仅最终层多视图 |
|---|---|---|---|
| CIFAR-10 | 1.489282 / 9.249549 / 0.161011 | 1.515900 / 9.125442 / 0.166118 | 1.940832 / 4.019990 / 0.482795 |
| CIFAR-100 | 1.575668 / 9.386764 / 0.167861 | 1.571517 / 9.223890 / 0.170375 | 2.130958 / 4.396498 / 0.484694 |

补齐的对应关系（全部在 `gram_analysis_final.csv`）：

- **模型指纹**：六份 JSON 的 `backbone_fingerprint` 与 `results/gate1/gate1_20260824_{v8,c100}_bigcal/units/<variant>__seed1/checkpoints/last.ckpt` 的 backbone 参数绝对值和（不含 BN running 统计量）逐一相符（差 < 1e-3）。检查点写于 8 月 24–25 日，分析文件写于 8 月 25 日 16:46–17:58，之后检查点未变。
- **训练配置/版本**：`configs/gate_{v8,c100}_bigcal`，与 `gate_v8`/`gate_c100` 的差异仅 `data.n_calibration: 2500 → 10000`、`data.n_val: 2500 → 10000`（配置的 status 字段自述"训练图像从 45,000 降到 30,000，精度不可与 v8 同表"）。训练提交 `8c556cd`（seed-1 六个单元起始 8 月 24 日 15:29–20:14），200 epochs，每单元一次运行无恢复。
- **图像身份**：`GateDataModule` 用 `torch.randperm(50000, generator.manual_seed(1))`：前 10,000 校准，次 10,000 验证，余 30,000 训练；CIFAR-10 与 CIFAR-100 同一置换。校准池 = 校准 ∪ 验证 = 20,000（convergence 模式 `pool_val_into_calibration=True`），`calibration_samples` 记录 20000；完整索引与 sha256（`448b6a7723e9…`）在 `split_indices.json`。评价集 = CIFAR 测试集 10,000 张（链视图随机种子 seed+300000）。三者两两无交集（已计数）。
- **坐标估计**：按层在校准池特征上拟合（`fit_level_coordinates`，surrogate 用 ridge 1e-3；projection 用截断 τ_relative = 1e-3）；`retained_ranks = [128, 128, 128]`，维数 128，六份相同。投影头随模型联合训练并从检查点加载（`weights_loaded: true`）；指纹只覆盖 backbone。
- 收敛曲线 N = 625…20000 均在，10000→20000 的 projection 相对误差变化 0.026（C10 分层）。

**45,000 图像完整 ridge 模型的 Gram-corrected 分析**：存在，但协议不同。`results/validity/c10_gram_product_endpoint.json`、`c100_gram_product_endpoint.json`（mode `convergence_extended`，seed 1）：指纹与 `gate1_20260820_v8/product_endpoint__seed1`、`gate1_20260821_c100pilot/product_endpoint__seed1` 相符。Stage-B 池 = 校准 2,500 + 验证 2,500 + 测试集前半 5,000 = 10,000；评价在测试集后半 5,000。N = 10000 时：CIFAR-10 1.337879 / 4.202028 / 0.318389，**retained_ranks [21, 128, 128]**（根层只保留 21 维，与 bigcal 的 [128,128,128] 不可比）；CIFAR-100 2.450029 / 9.708451 / 0.252360，[128,128,128]。同批 `additive_mview`、`final_mview` 也有对应文件。**β=0 没有任何 Gram 分析文件**；补一份属冻结后评价（加载 `gate1_20260823_v8_beta0` 检查点跑 `run_validity_gate.py`，分钟级），本次未启动。

---

## 4. 验证数据、学习率选择与模型选择边界

- **EMA 在验证中更新**：`HierarchyCertificateModule._shared_step` 对 train 和 val 都调用 `_variant_loss`，EMA 端点缓冲的更新（`ema_c_dir.mul_(m).add_(c_dir, 1−m)`）没有 `self.training` 门控，因此每 10 个 epoch 的验证批次以及 Lightning 起始 sanity-check 批次都会更新它。该行为由 `c60f532`（8 月 19 日）引入，到 `ae02ffe` 未改；8 月 20–24 日归档 ridge 单元与 9 月 14 日复现行为一致。本稿"45,000 称为优化图像子集"的措辞与此相符。
- **基线 5,000 图像**有三种不同的 5,000，须分开写：
  1. 基线自身切分保留的 2,500 校准 + 2,500 验证（`train_result.json`: train 45000 / val 2500 / calibration 2500 / test 10000）；`best-*.ckpt` 按该 val 的 `val_score` 选，但**没有任何报告行使用 best 检查点**——参考配方与选中学习率的剖面全部读固定的 `epoch-0200.ckpt`（epoch-20 行读 `epoch-0020.ckpt`），旧模板与 byol_bn 读 `last.ckpt`。
  2. 学习率选择的 holdout：`torch.randperm(50000, seed 0)` 的前 5,000，探针在其余 45,000 上拟合、在 holdout 上打分（`score_set = holdout-5000`）。它与 10,000 测试集**完全分开**；与基线预训练的 45,000 张**不独立**（同一 50,000 池的另一 seed 切分），也不等于上面的 2,500+2,500。索引在 `split_indices.json`。
  3. FMCA 单元 SGD 探针的 5,000 验证：`randperm(50000, unit seed)` 前 5,000，仅记录 `val_accuracy`，无检查点、无选择；探针在其余 45,000 上拟合，测试集打分。
- **选择分数**：从 `lrsel_*_holdout.json` 重算，确为 epoch-200 holdout 上 Stage 2–4 的平均：simclr 76.27 / **79.56** / 79.48，barlow_twins 76.70 / **79.47** / 79.34，vicreg 77.04 / 79.27 / **80.09**，moco_v2 72.59 / 75.13 / **76.08**（lr 0.03 / 0.1 / 0.3）。测试集未参与选择。
- **口径**：主表是逐列最大值。"验证集选择的单个模型在测试集上的分数"目前**没有记录**：FMCA-AV 单元没有任何基于验证集的模型选择步骤，基线的 val_score 选择的检查点未被评价。要用该口径需另定冻结协议。

---

## 5. 重复数、层名、评价器字段

- **评价器字段**（manifest `evaluators`）：逐层——标准化（训练均值/标准差，下限 1e-6）、零初始化、L-BFGS max_iter 200 / history 10 / strong-Wolfe、权重罚 1e-4·‖W‖²、拟合 50,000、打分 10,000；SGD——冻结骨干 + 线性层，lr 0.1 余弦 100 epochs，动量 0.9，wd 0，拟合 45,000。
- **归属**：五次主结果均值 = `gate1_20260820_v8/product_endpoint__seed1–5`（C10；提交 `1c747ea`…`6b3ab67`）与 `gate1_20260821_c100pilot/product_endpoint__seed1–5`（C100；seed 4 于 9 月 14 日在 `70e5e7b` 下重跑）。三次匹配消融 = `v8_beta0 / v8_alpha0 / v8_m4 / v8_parallel`、`c100_beta0 / c100_parallel` 各 seed 1–3（8 月 22–24 日）。新增复现 = `gate1_20260914_ridge_ms(_c100)` seed 1–3（`99b904d`，9 月 14 日）。
- **level_role**：79 份 `layerwise_profile.json` 中 20 份的 `level_role` 整体错位一档（layer3 标为 root tap）：v8 4、c100pilot 4、v8_m4 3、v8_beta0 3、c100_beta0 3、v8_alpha0 2、c100_parallel 1。所有 79 份的 `backbone_layer` 正确。核实用到该字段的三处都按 `backbone_layer`/stage 顺序读：`make_report_figures.py` 的 `fig_layerwise` 按 `stages` 顺序取值、横轴标 layer1–4；`results_audit_20260912/verified_results.json` 的 `layerwise_summary` 以 stage "0"–"3" 键和 `backbone_layer` 记录；`verified_tables.tex` 标 Stage 1–4。`level_stages = [1, 2, 3]`（0 起）对应 layer2/3/4 = Stage 2–4，与本稿一致。
- **复现的近似恒等（新发现）**：9 月 14 日 `ridge_ms` 六个单元中，落在与归档同一 GPU 家族（A100）上的四个——C10 seed 2、3，C100 seed 1、2——最终权重与归档权重最大差 7.7e-3 … 1.3e-2，SGD 探针、kNN 和四个 stage 的值与归档**完全相同到小数点后四位**；落在不同家族的两个（C10 seed 1：L40S→A100；C100 seed 3：L40S→A100）权重完全不同（max|Δw| 7.1、4.6）。因此"复现区间与归档重叠"（P-R1）对 4/6 单元近乎必然；Hierarchical 的 Stage 2 最大值 80.90 来自真正独立的 C10 seed 1。`RIDGE_MILESTONE_REPLICATION_RESULTS_20260914.md` 已加注。

---

## 6. 截断结果按数学实现归类

69 个截断相关单元的归类在 `truncated_units_manifest.csv`（每行：链归一化、辅助多视图项、辅助 Gram 是否求导、矩阵目标、训练起始时间、提交、恢复次数）。汇总：

| 辅助多视图项 | 单元 | 说明 |
|---|---|---|
| 链截断 + 辅助 **ridge 1e-3 trace score**（`02fa55e` 之前） | CIFAR-10 `paper_composition` s1–3、`paper_beta0` s1–3；旧代码 worktree `5322cdb` 的 6 个 5-epoch 诊断 | 辅助 Gram **参与求导**（`trace_score(estimate_moments(f, leaf_views), ridge=1e-3)`，梯度经 R_f、R_g、P_fg） |
| 链截断 + 辅助 **截断分数**（`02fa55e` 起） | CIFAR-10 `alpha0`、`T2`、`T4`、`K256`、star；全部 CIFAR-100 `paper_*`（λ0、endpoint_only 除外）；两个 45k 主跑；新代码 bifurc 30k/45k；x2x2 | 辅助 Gram **不求导**（`gram_detach_metric` 默认 True，白化器 detach，梯度只经交叉项） |
| 无辅助项（leaf weight 0） | `paper_lambda0`、`paper_endpoint_only`（两数据集） | 与代码版本无关 |

链归一化：所有 `paper_*` 单元为共享对称截断逆根，τ_relative = 1e-3；矩阵目标为直接端点算子进入梯度，无 EMA、无 stop-grad（模块对 `paper_*` 断言）。每个单元的 `train_logs` 只有 `version_0`，无恢复。5-epoch 诊断单元（bifurc 30k/45k 新旧代码）的探针精度不作为性能数值；本文档与 CSV 只保留其保留维数观察。

---

## 7. 跨数据集外部基线

服务器上**没有** CIFAR-100、Tiny ImageNet 或 ImageNet-100 的外部基线运行：`experiment_runs/` 中所有基线目录（`sslmatched_*`、`sslref_*`、`sslreflr_*`）均为 CIFAR-10；`results/baseline_layerwise/` 亦然。存在但不构成可比基线的：Tiny ImageNet-200 的 FMCA-AV 变体（`gate1_20260825_tin200_v1`，7 个变体 × 3 seed，100 epochs，95,000 张优化图，无外部方法）；CIFAR-10 深度-4 链（`gate1_20260826_depth4_v1`，30,000 图）。ImageNet-100 只有 `code_remote/` 下的 notebook，无结果记录。本稿不填外部基线是正确的。

---

## 未在本次完成的事项

1. ridge β=0 里程碑复现（6 单元）仍在跑；完成后由后台任务给出 P-C1 对照，并整组更新 `ridge_beta0_matched*.csv`。
2. 45k 完整 ridge 与 β=0 在同一协议下的 Gram-corrected 分析：完整模型有 seed-1 扩展协议版本，β=0 没有；未启动新评价。
3. 原论文 90.3 / 70.3 / 61.7 三个数未核实，来源不在本服务器的结果目录中。
