# 证书框架：测试设计与论文对比 Baseline 设计

**日期：** 2026-08-16
**代码：** `fmca_av/certificate/`（`20260816_path_supported_certificate_v1`），对齐《FMCA_AV_SERVER_HANDOFF_20260815.md》§2/§4/§5/§7
**状态：** 测试第 1 层已实现并全绿；第 2 层已预注册（FROZEN，待 go）；第 3 层与 baseline 缺口为 DRAFT，等用户 steer。

---

## 1. 测试设计（三层）

### 1.1 第 1 层：单元验收套件（已实现，51 个测试全绿，纯 CPU ~30s）

| 测试文件 | 覆盖的红线 / 冻结条目 |
|---|---|
| `tests/test_certificate_counterexamples.py` | 六反例 population 表逐值断言（§2.4）；完整 Stage-B/C 经验管线复现；证书只接受正例；极化恒等式；Weyl 界；N 收敛；M-children 方差下降 |
| `tests/test_certificate_controls.py` | 双侧 gauge 不变（正控）；单侧旋转恶化（负控）；layer-order shuffle；pairing shuffle 噪声底线（parent 与 endpoint）；centering-off 常数模式污染（σ₁>0.95 伪传输）；edge-wise 独立校准破坏共享界面；naive σ-乘积在错位模式与旋转界面下失败（§2.5/§2.6） |
| `tests/test_certificate_gaussian_chain.py` | Hermite 解析链：全接口精确闭合；内层截断 → 解析缺陷 (ρ₀ρ₁)³（Thm 1 数值版）；经验收敛；Hermite 正交归一 |
| `tests/test_certificate_objective.py` | 冻结 loss 代数：closed chain 闭合比小 / 幻觉链爆炸；零算子无奖励（反例 4 = endpoint 项不可省的理由）；whitening 罚敏感性；α 的 additive 符号语义；梯度流；closure stop-grad 消融语义 |
| `tests/test_view_tree.py` | 嵌套树 Markov-by-construction：child 框 ⊆ realized parent 框（真实像素坐标寻址）；chain 状态 = descendant 0；mask 层保框改像素；确定性；parallel（star）负控会逃出 parent 框；collate 形状 |
| `tests/test_hierarchy_module.py` | 7 变体全部构建/前向/有限；完整方法反传到 backbone+全部 level projector；每层单 projector 同时服务 chain 与 children（共享界面红线）；level_stages 单调校验 |

### 1.2 第 2 层：Wave 0 预注册扫描（FROZEN，`prereg/WAVE0_CERTIFICATE_PREREG_FROZEN_20260816.md`）

网格：6 反例 × N∈{1e3,1e4,1e5} × M∈{1,4,16} × 20 seeds（A 组）+ Hermite 全/截断链同网格（B 组）+ 离散链 naive 失败（C 组）。判据 G1–G6（证书选择性 / N、M 收敛 / 控制电池 / 经验 Weyl 覆盖率 / 解析缺陷 CI / 闭合恒等式）。产物回填论文 §VII。runner probe 已通过；全量在你 "go" 后经 sbatch 执行（纯 CPU）。

### 1.3 第 3 层：Gate 运行时协议（每次训练运行自带）

Stage-B（冻结坐标，calibration split）→ Stage-C（split-descendant / cross-fit，`crossfit_edge_and_endpoint`）→ 证书全套 + §4 控制电池逐项落盘。术语纪律：共享样本 = "separately estimated"，真 cross-fit 才写 "cross-fitted"。

---

## 2. 论文 Baseline 设计

### 2.1 Gate 内部结构变体（7 行，同 parents/同编码视图/同 backbone 预算；configs/gate/ 已生成 DRAFT）

| # | variant | 机制 | 控制什么 |
|---|---|---|---|
| 1 | `final_2view` | 末层 flat FMCA，2 views | 经典两视图基线 |
| 2 | `final_mview` | 末层 flat FMCA，M views | 多视图但无层级 |
| 3 | `additive_2view` | 逐边 −S(C_ℓ) 求和，2 views | 多层加法（无复合） |
| 4 | `additive_mview` | 逐边 −S(C_ℓ) 求和，M views | HAI/HFMCA 家族 |
| 5 | `amdim_cross` | 选定跨尺度对 −S(C_{i→j}) 求和 | AMDIM 式 cross-scale |
| 6 | `product_only` | −S(C^comp) | 纯账本，无端点闭合（幻觉风险行） |
| 7 | `product_endpoint` | 完整冻结 loss | 本文方法 |

必加消融（handoff §5）：bootstrap 端点 vs separately-estimated 端点；closure stop-grad / EMA / 交替更新；nested tree vs parallel trajectories（view_tree mode 开关现成）。
预注册决策规则照 §7 第 3 步：6/7 行须在 held-out 闭合缺陷显著优于 additive 行且下游不劣，否则论文降级 measurement-only。

### 2.2 外部 objective-matched baseline：现状与缺口

**仓库已有（faithful、已跑或在跑）：** FastSSL-Barlow-Twins（M=2/8）、FastSSL-VICReg（M=2/8）、FroSSL（M=2 官方；M=8 崩溃调查中，是论文的 intervention 案例素材）、HAI faithful（8 views/4 stages）、flat FMCA-AV（M=2/M=8 旧矩阵，标 growing-compute）。

**缺口（8/10 综述 §6.2 定为必跑，按优先级）：**
- P0：Geometric PVC（Poly-View, ICLR24，需按伪代码实现）；MV-DHEL + MV-InfoNCE（官方码 github.com/pakoromilas/Multi-View-CL）；SSOLE（官方码 github.com/husthuaan/ssole）。
- P1：SimCLR 2-view 与 pairwise/mean-of-rest（PWE/AVG）行（如旧 screening 行可复用则标注沿用）；DINO 或 SwAV multi-crop recipe 行（recipe 级，不进 objective-matched 主表）；SCFS（官方码）。
- P2：M3G（仅 M=3 小规模）、INTL（若正文强调 spectral SSL）。

**预算协议：** 每个对比双报告 fixed-parent 与 fixed-encoded-view；记录 unique parents、总编码视图、GFLOPs（沿用已有 profiler 管线）、wall-clock、峰值显存。

### 2.3 Probing-depth 研究的对照组（第 4 步 deliverable，零训练成本）

s^cert_{ℓ→L} 预测最佳 probing 层 / tunnel 起点，对照：CKA、effective rank、matrix entropy、direct endpoint CCA、layerwise linear probe、local dependence sum。模型库：ResNet-50/152、ViT-B 的 supervised / SimCLR / DINO / MAE 公开 checkpoints。

---

## 3. 悬而未决、需你决定的点

1. **α 项符号**：冻结公式字面是 `+α·Σ S(C_edge)`；additive 对照语义要求最大化逐边依赖，故实现取 `−α·Σ S`（α>0, β=0 即 HFMCA-additive 家族）。已在 objective.py docstring 标注，待确认。
2. **flat 变体（v1/v2）的 f 侧**：经审查修正后，v1/v2 的视图 = endpoint full-path descendants（对根视图的独立完整重采样，即经典 star p(Y|X₀) 语义），不再用最后一条边的 masked 兄弟节点。但 parent 特征仍取"投影视图均值"，其分数 = ‖Cov(视图均值)‖²，在零依赖下有 K/M² 的确定性下限（W-MSE 式泛函而非严格 FMCA score）。更忠实的替代是沿用现有 `VisionFMCAAV`（独立 f-head）作 v1/v2 行；建议后者，gate 预注册时定。
3. **层-stage 映射**：DRAFT 配置用 ResNet-18 stages [1,2,3]（level 0=global→layer2 … level 2=masked→layer4）。备选 [0,2,3] 或四层树。
4. 最高权威冻结稿《…CERTIFICATE_REFINEMENT_20260815 (1).md》**尚未同步到服务器**（handoff §9 清单项）；本实现按 handoff §2 的自包含定义对齐，同步后需 diff 一遍。
5. **已知 DDP gap**（审查确认，单 GPU gate 不受影响）：hierarchy_module 训练时的均值/二阶矩/算子均为 per-replica（无 all_gather），日志未 sync_dist，flat 变体存在 unused projectors（DDP 需 find_unused_parameters）。多 GPU 前必须补齐。
