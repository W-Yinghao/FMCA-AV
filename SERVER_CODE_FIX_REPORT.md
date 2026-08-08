# FMCA-AV 科学正确性修复报告

日期：2026-08-09

项目路径：`/home/infres/yinwang/FMCA-AV`

修复版本标记：`20260809_scientific_correctness_v1`

## 版本边界

- 本次只修复 relative ridge、held-out spectrum/TSD 和 E9 dependence map 三项问题，没有修改 CIFAR100/ViT、harness 配额或队列/watcher，也没有启动 E5–E9 正式补跑。
- 新产生的训练、校准、评估和 E9 localization 输出会写入上述修复版本标记。没有该精确标记的既有输出，以及在本次修复前启动但之后才完成的进程，均视为 **pre-fix**。
- 没有删除或覆盖任何旧 checkpoint、校准文件、评估文件、localization 结果或汇总表。
- pre-fix 与 post-fix 必须使用不同 run ID/输出目录；只有带精确修复版本标记的结果才能进入 post-fix 汇总，禁止与旧结果混表。

## 修改文件与公式

- `fmca_av/operators.py`
  - 新增统一 `relative_ridge_scale` 和 `regularized_covariance`。相对 ridge 尺度为协方差对角线绝对值的均值；只对完全零尺度使用当前 dtype 的 `tiny` 下限，并对非有限输入显式报错，不再使用 `clamp_min(1.0)`。
  - whitening、trace、logdet 由同一 regularization 路径计算，因此对非零特征整体缩放 `a` 时，ridge penalty 同步缩放为 `a²`。
  - 新增统一 held-out spectral evaluation：
    - `z_f = (f - mean_f) @ transform_f`
    - `z_g = (g - mean_g) @ transform_g`
    - `C_test = z_f.T @ mean_views(z_g) / number_of_parents`
    - `singular_values = svdvals(C_test)`，`eigenvalues = singular_values²`
  - 原始 singular values/eigenvalues 不截断；只在 logdet/TSD 计算时截断接近 1 的 eigenvalues，并记录截断 mode 数。
  - 新增 paired-canonical dependence 辅助函数。采用的局部依赖贡献定义为
    `D(p) = Σ_k singular_value_k * u_k(x) * v_{p,k}(x)`，其中 `u(x)=(f(x)-mean_f)@transform_f`，`v_p(x)=(g(local_feature_p)-mean_g)@transform_g`。
- `fmca_av/objectives.py`
  - trace 与 logdet/whitening 共用统一 relative ridge 实现。
- `fmca_av/cli.py`
  - `evaluate` 改为调用完整 `C_test` 的统一 SVD 实现；逐坐标量只保留为明确命名的 `test_diagonal_correlations` 诊断。
- `scripts/run_e7_tsd_calibration.py`
  - E7 TSD 改为调用同一个 held-out SVD 函数；保存原始 spectrum，并只在 TSD/logdet 阶段截断及记录截断数量。
- `scripts/run_dependence_localization.py`
  - `u(x)` 通过 checkpoint 配置对应的真实 backbone、parent aggregation 和 `f_head` 路径计算；局部 `v_p` 通过 `g_head` 与 `transform_g` 计算。
  - 输出 `signed_dependence`、默认非负图 `absolute_dependence`、`g_energy_baseline`，以及 activation、center、edge、random baselines。
  - localization、faithfulness、deletion/insertion 和 randomization 的主指标均使用 `absolute_dependence`，不再使用 g-only energy 冒充 dependence map。
- `scripts/render_e9_localization_assets.py`
  - 后处理优先读取新的 `absolute_dependence` 主图，并在 caption 中明确其 paired-canonical 定义与 post-fix 边界；缺少精确修复版本标记的旧文件会被排除，防止新旧结果混表。
- `fmca_av/__init__.py`
  - 导出新增的公共计算辅助函数。
- `tests/test_operators.py`
  - 新增 `1e-3/1/1e3` 缩放不变性、零方差/非有限输入、held-out 旋转与完整 SVD、logdet 截断边界测试，并保留原测试。
- `tests/test_localization.py`
  - 新增 f 侧敏感性、零 singular values、toy 最大贡献位置恢复测试。

## 最小 Slurm 验证

所有 Python 验证均经现有 Slurm harness 以 `--gpus 0` 提交；没有在登录节点运行 Python 测试，也没有提交 CUDA 任务。

| Slurm job ID | 验证 | 结果 |
|---|---|---|
| 930368 | `python -m unittest discover -s tests -v` | PASS：12 tests，exit 0 |
| 930369 | 修改入口的 `py_compile` | PASS：exit 0 |
| 930376 | 最终 `python -m unittest discover -s tests -v` | PASS：12 tests，exit 0 |
| 930377 | 最终修改入口（含 E9 renderer）的 `py_compile` | PASS：exit 0 |
| 930381 | E9 post-fix 汇总隔离修改后的 `py_compile` | PASS：exit 0 |

验收结论：relative-ridge 小/大尺度不变性 PASS；零方差与非有限输入处理 PASS；held-out rotation/full-SVD PASS；E7 共用完整矩阵 SVD PASS；E9 同时依赖 f、g、singular values 的三个构造测试 PASS。

## 现有结果影响审计

审计只读取现有 `runs/` 元数据，没有修改结果。以下是 2026-08-09 01:28 CEST 的快照（修复前启动的后台任务之后若产生无版本标记输出，也自动归入同一 pre-fix 范围）：

- 585 个已有、`train_result.json` 带 `spectral_convention` 但不带新修复版本标记的 FMCA source run，共 1,087 个 checkpoint 文件；其 `artifacts/checkpoints/` 下的 checkpoint 全部受 relative ridge 影响。这一明确筛选规则即完整 FMCA checkpoint 清单，避免在报告中展开 1,087 条路径。
- 已完成的 DCCA/VAMP2 checkpoint 也受影响，当前明确文件为：
  - `runs/20260807-104910_cifar10-dcca-5epoch-screening/artifacts/checkpoints/best-004-114.946465.ckpt`
  - `runs/20260807-104910_cifar10-dcca-5epoch-screening/artifacts/checkpoints/last.ckpt`
  - `runs/20260807-105411_cifar10-vamp2-5epoch-screening/artifacts/checkpoints/best-004-115.624214.ckpt`
  - `runs/20260807-105411_cifar10-vamp2-5epoch-screening/artifacts/checkpoints/last.ckpt`
- 561 个既有 `calibration.pt` 和 560 个既有 `evaluation.json` 属于 pre-fix spectral evaluation 路径。
- 62 个既有 E9 `localization.json`（CUB 40、VOC 11、ImageNet 11，其中 randomization control 28）均没有新 map 版本标记，使用旧 g-only spectral/energy map，必须全部重跑。
- 4 组既有 E7 TSD calibration 输出使用旧 held-out 对角实现：
  - `20260807-062549_e7-heldout-tsd-calibration-full`
  - `20260807-062817_e7-heldout-tsd-calibration-k4-diagnostic`
  - `20260807-115931_e7-tsd-calibration-reliability-20rep`
  - `20260807-120707_e7-tsd-calibration-high-resource-10rep`

### 必须重新训练

- 所有以 relative-ridge FMCA objective 训练的 source checkpoint，包括上述 585 个已完成 FMCA source run，以及修复前已经启动但审计时尚未产生 `train_result.json` 的任务。
- DCCA 与 VAMP2 baseline 也调用受影响的 whitening/trace 实现，必须重训。
- E5 中所有 FMCA/DCCA/VAMP2 confirmatory seed；E7 中以这些方法训练的 source encoder；依赖这些 encoder 的 E6/E9 下游结果应在新 source checkpoint 上重做。
- E2 gradient-variance、E3 numerical/ridge ablation，以及直接调用该 objective 的 complexity/FLOPs/score 分析需要重新计算，即使它们不产生 source checkpoint。

旧 FMCA/DCCA/VAMP2 checkpoint 不得续训后直接宣称与 post-fix 新配置完全一致；应从头训练并使用新 run ID。仅对旧 checkpoint 重新 calibration/evaluate 可以作为明确标注的 pre-fix 诊断，但不能替代 post-fix 重训。

### 只需重新 calibration/evaluate

- 不使用 relative-ridge 目标训练的现有 source checkpoint，例如 SimCLR、Barlow Twins、VICReg、spectral contrastive、FastSiam、BYOL、MoCo v2、DINO 与 supervised 模型，本次 issue 1 不要求仅因此重训。
- 但凡使用旧逐坐标 held-out spectrum/TSD 的这些模型，都必须重新 calibration/evaluate；旧的 561 个 calibration 和 560 个 evaluation 文件不能进入 post-fix 汇总。
- E7 的四组旧 TSD 输出必须重新生成。原始 spectrum 必须来自完整 `C_test` SVD，截断只用于 TSD/logdet。

### E5–E9 建议重跑范围

- **E5**：FMCA/DCCA/VAMP2 全部 confirmatory seeds 从头重训；不受 ridge 影响的方法可保留 source checkpoint，但其 spectral calibration/evaluation 要重做。
- **E6**：凡是依赖受影响 source encoder 的下游任务，在新 source checkpoint 上重跑；不受影响 source 的旧训练可保留，但相关 spectral calibration/evaluation 需更新。
- **E7**：受影响 source encoder 从头重训；全部 held-out calibration、TSD 和依赖该 spectrum 的 factor 分析重新计算。
- **E8**：纯解析或连续 Markov 理论表不受这三项修复影响；若具体脚本调用本次有限样本 objective/spectrum，则该部分单独重算。
- **E9**：上述 62 个旧 localization run 以及由其派生的 localization、faithfulness、deletion/insertion、randomization 表和图全部重跑，不论 source method 是否受 ridge 影响。

本任务没有启动任何完整 E5–E9 补跑。
