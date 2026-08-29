# Gate 1 附录：训练稳定性修复（APPENDUM，fleet 前，已披露）

**日期：** 2026-08-17
**触发：** QC probe 单元（SLURM 945112）按预注册 QC gate **失败**——loss 自 epoch 0 即 NaN。逐步诊断（CPU，batch 64）：未归一化 loss 在 K=128 时 whitening 罚项初值 ≈379（K² 项求和），SGD lr .03+momentum 在四次方罚面上正反馈，step 2 达 9e11，step 3 NaN。probe-gate 拦截成功，fleet 因 afterok 依赖零启动、零 GPU 浪费。

**除 NaN 外未观察到任何实验结果**；本附录在 fleet 产生任何数据之前冻结。

## 修复（实现层，冻结的 E1/E2/E3 对比与解释表不变）

1. **维度归一化**：score 项 → ‖C‖²_F / min(K_in,K_out)（每模式平均奇异质量）；whitening 项 → mean((R−I)²)（逐元素均值）。closure_ratio 本为比值，不变。这等价于重定标有效 β/γ/α 与学习率——超参数本属授权决断范围（Gate 预注册 §0）。
2. **梯度裁剪** gradient_clip_val=1.0。
3. **10-epoch 线性 warmup**（start_factor .01）+ cosine。

## 验证

- 45 个相关单元测试全绿（阈值随标度更新两处，语义不变）。
- CPU 4-step 检查：全部 loss 项 O(1)，梯度尖峰（50）被裁剪吸收，无爆炸。
- 权威验证 = 重新提交的 GPU probe-mode 单元（2 epochs 全管线）；fleet 依赖其成功。

## 判据不变声明

E1/E2/E3、7×3 设计、200 epochs、评估协议、塌缩定义全部照旧。本附录只改变优化器能否走到那一步。
