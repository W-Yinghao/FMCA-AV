# Plug-in 研究预注册（DRAFT——v8 判定后冻结并启动）

**日期：** 2026-08-20　**问题：** EMA 闭合正则作为可移植外挂，装到标准 SSL 目标上，能否在不损失下游精度（±1pt）的同时显著改善 held-out 闭合缺陷？（TPAMI framework-lane 的 plug-in 实例化模板）

## 设计

- 3 方法（faithful 实现均在仓库，含 formal 参照数）：FastSSL-Barlow（off-diag 1/256）、FastSSL-VICReg（25/25）、FroSSL（inv 2.0）。
- 每方法 2 行 × 3 seeds = 18 单元：**base 行** = base loss + sg-backbone 的测量头训练（backbone 只见 base loss）；**plugin 行** = base loss + α·edges + β·EMA-closure + γ（闭合梯度进 backbone）。两行唯一到达 backbone 的差异 = 闭合正则。
- 数据/预算：v8 满配树、M=8 端点、200 epochs、lr 与优化器沿用各 base 的 formal 设置（首版统一 SGD 0.1 便于匹配——冻结前定夺此项）。
- 终点：E-P1 缺陷改善（plugin 行 held-out 归一化缺陷 < base 行，9 组配对全向 + 均值 ≤0.8×）；E-P2 精度非劣（probe/kNN ≥ base −1pt）；证书全套 + 控制电池随行。

## 判读

双过 → "closure 可移植正则"主张成立（论文 §VIII 的 plug-in 小节）；仅 E-P1 → 正则有效但有精度代价，如实报告 trade-off；均不过 → 该主张撤回，v7 结果独立成立。
