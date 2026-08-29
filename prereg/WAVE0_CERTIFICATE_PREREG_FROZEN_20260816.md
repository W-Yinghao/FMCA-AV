# Wave 0 预注册：反例电池 + 解析链验收（FROZEN）

**日期：** 2026-08-16
**状态：** FROZEN（设计全部由《FMCA_AV_SERVER_HANDOFF_20260815.md》§2.4/§7 与冻结本体决定；本文件冻结后不得修改，追加分析走 APPENDUM）
**依赖代码：** `fmca_av/certificate/`（版本串 `20260816_path_supported_certificate_v1`）
**计算级别：** 纯 CPU（sbatch 提交），无 GPU
**执行前提：** 单元测试套件（tests/test_certificate_*.py, test_view_tree.py, test_hierarchy_module.py）全绿 = probe 通过；用户明确 "go"。

---

## 1. 问题

冻结三元组框架 (C^dir, C^comp, Δ^CK) 的实现是否：
(a) 精确复现六个冻结反例的 population 验收表；
(b) 经验估计随 N（parents）收敛、边估计误差随 M（每边 children 数）下降；
(c) 证书只接受正例（closed chain），拒绝全部五个反例；
(d) 全部控制电池按预注册方向反应；
(e) 在 Gaussian–Hermite 解析链上复现解析投影缺陷（Thm 1 数值版）；
(f) 在异质/非正规链上按预期展示 naive 奇异值相乘失败。

## 2. 单元定义（real-id：单元键 = 名称字符串，禁止 bench index）

**A 组（六反例）**：unit = (case_name, N, M, seed)
- case_name ∈ {nilpotent_interface, hallucinated_path, leaky_interface, zero_operator, isospectral_mismatch, closed_chain}
- N ∈ {1 000, 10 000, 100 000}；M ∈ {1, 4, 16}；seed ∈ {1..20}
- 每单元流程：Stage-B 校准样本 N_cal = max(N/4, 500)（独立种子派生），Stage-C 估计样本 N；endpoint descendants = M。

**B 组（Gaussian–Hermite 链）**：unit = (config_name, N, M, seed)
- config ∈ {full_orders（[1,2,3] 全层）, truncated（内层只留 [1,2]）}，ρ = (0.8, 0.6)
- 同 A 组网格。

**C 组（离散链 naive 失败，matrix-level，无采样）**：unit = (chain_name,)
- misaligned_diag（diag(0.9,0.4)×diag(0.4,0.9)）、rotated_misaligned（同上加界面正交旋转）、nonnormal_markov（markov.py nonnormal_chain 两条不同链的白化算子按序复合 vs 直接二步算子）。

## 3. 主要终点（每单元 JSON 记录）

1. `edge_error_max` / `dir_error_max`：估计算子对 population 的逐元素最大绝对误差；
2. 证书报告全套：s^end、s^path、δ_op、δ_F、s^cert、Δ_mass、A_op、极化恒等式残差、top-k 主角、η_path（次要，含 clip 标志）、per-edge Frobenius/σ₁；
3. `certificate_accepts` = [max_k s^cert_k > τ]，τ = 0.05（预注册阈值）；
4. 控制电池（每单元同批数据上执行）：
   - 双侧 gauge 旋转：报告量相对变化（预期 ≤ 1e-8）；
   - 单侧旋转：C^comp 变化量（仅当 ‖C^comp‖_F > 0.05 时评判；预期 > 0.01·‖C^comp‖_F；零算子反例上该负控空真，记录不评判）；
   - parent–child pairing shuffle（10 次）：floor 最大值（预期 < 0.25 × 真边范数，closed_chain 上）；
   - endpoint pairing shuffle：同上（endpoint 版）；
   - centering-off（特征整体平移 +3 后）：uncentered 复合 σ₁（预期 > 0.95）vs centered（预期 < 0.85，closed_chain 上）；
   - layer-order shuffle（维度可复合时）：复合矩阵变化（异质链预期非零）；
   - naive σ-乘积 vs σ(完整复合)：误差（C 组预期 > 0.2）。
5. QC 哨兵：同 seed 重跑 bit-identical；所有量有限；Stage-B/C 样本 seed 域不相交。

## 4. 聚合规则（freeze-before-aggregate）

- 单元全部落盘后才允许聚合；聚合脚本只读 units 目录。
- 每 (case, N, M)：median + bootstrap 95% CI（seed 维，B = 2 000，聚合种子 20260816）。
- 收敛判定用中位数序：err(N=1e5) < err(N=1e4) < err(N=1e3)；M 判定同理在固定 N=1e4 上。

## 5. Go / No-Go（预承诺解释表）

| 判据 | 内容 | 通过 | 失败含义 |
|---|---|---|---|
| G1 | 证书判定：closed_chain 全部单元 accepts=true；五反例在 N ≥ 1e4 全部单元 accepts=false | 必须 | 证书公式或估计管线有错，**阻塞一切后续** |
| G2 | 收敛：A/B 组 median 误差随 N 单调降；M ∈ {1→16} 边误差 median 降 | 必须 | 估计器有偏或采样器不符合条件独立语义 |
| G3 | 控制电池：正控不变（≤1e-8 相对）；负控 ≥ 95% 单元按方向反应 | 必须 | 对应红线实现失效 |
| G4 | 经验 Weyl 覆盖：s^cert_k ≤ s^end_k + ε_tol 的违反率随 N 下降；N=1e5 时逐单元违反幅度 < 0.02 | 报告 | 违反幅度不减 → 需 Thm 2 的 ε_n 项先行（预期内，如实报告，不隐藏） |
| G5 | B 组 truncated：δ_op 的经验估计落在解析值 (0.48)³ 的 CI 内 | 必须 | Thm 1 数值化失败 |
| G6 | C 组：naive σ-乘积误差 > 0.2 且完整矩阵复合与直接算子一致（≤1e-10，全基无截断时） | 必须 | 复合实现走了禁用路径 |

**任一"必须"失败 → 停止、报告最差单元格、不启动 CIFAR Gate。全部通过 → 回填论文 §VII placeholder（两步提交：结果 commit 与解释 commit 分离），并等待用户对 Gate 预注册的 steer。**

## 6. 产物

- `results/wave0/20260816_path_supported_certificate_v1/units/<unit_key>.json`（append-only、skip-if-done）
- `results/wave0/.../WAVE0_RESULTS.md`（中性结果表 + CI + QC + sha256 manifest）
- 运行脚本：`scripts/run_wave0_certificate_suite.py`（--probe 单单元模式）
