# Gate 1 附录 3：detached 白化 + 0.1 训练 ridge（APPENDUM，v2 probe 已见，披露）

**日期：** 2026-08-17
**触发：** v2 QC probe（SLURM 945523）状态 complete 但 epoch-1 的 endpoint_score = 6.96，远超白化算子的总体上界 ~1。机制：白化矩阵对批统计可微时，优化器把特征推向 pooled 协方差**低估**子集方差的噪声方向；ridge 1e-3 下小特征值方向被放大至 ~31×，score 由估计缺口而非真实依赖构成（小批白化目标的已知失效；W-MSE 因此依赖大 batch）。v2 fleet 在其 8 单元起跑数分钟内全部截停，GPU 浪费可忽略；v2 目录只含该 probe，按档保留。

## 更正（v3）

1. **白化矩阵 detach**：W 由 batch 矩计算但不回传梯度——优化器无法定向塑形估计噪声。反塌缩压力保留（γ 条件项可微 + detached W 仍放大稀有方向）。
2. **训练 ridge 1e-3 → 0.1**（相对）：历史 FMCA notebook 即用 0.1 量级 ridge，这正是其稳定的来源；放大上限从 ~31× 降到 ~3.2×。
3. **哨兵升级为 per-term 上界**（endpoint/leaf ≤ 2.5、edge_sum ≤ 5、cross_sum ≤ 7.5、product ≤ 2.5、|loss| ≤ 50）：此类失效将直接 fail probe 作业，经 afterok 依赖自动阻断 fleet，无需人工识别。
4. Stage-B/C 测量协议不变（其大样本、无梯度、冻结坐标）；v3 = `gate1_20260817_v3`。

## 验证

64 个单元测试全绿（两处阈值随 0.1 ridge 收缩调整，语义不变）。E1/E2/E3、7×3 设计、200 epochs 照旧。

## 认识论备注

score 收缩（×~1/1.21）对所有 7 个变体同等作用，E1/E2 是变体间对比，不受共同收缩影响。
