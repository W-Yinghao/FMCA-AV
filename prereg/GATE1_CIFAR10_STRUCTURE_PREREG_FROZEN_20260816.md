# Gate 1 预注册：CIFAR-10 结构 Gate（FROZEN）

**日期：** 2026-08-16　**状态：** FROZEN（冻结后不得修改；追加分析走 APPENDUM）
**权威依据：** 《FMCA_AV_SERVER_HANDOFF_20260815.md》§7 第 3 步；用户 2026-08-16 授予全部决断权（"你来推进，你有全部的决断权力，我要看到图像实验的结果"）。
**前置：** Wave 0 整体通过（G1–G5 冻结判据 + G6 相对判据附录，附录接受权由上述授权行使，记录于此）。

## 0. 授权决断记录（本次冻结前裁决的悬置项）

1. **G6 附录接受**：Wave 0 视为整体通过，论文 §VII 回填解锁（回填在 Gate 结果出来后一并做）。
2. **α 项符号**：采用最大化符号（−α·Σ S(C_edge)；α>0, β=0 = HFMCA/HAI additive 家族）。handoff §5 字面 "+α" 判定为笔误：惩罚逐边依赖不对应任何已声明设计。Gate 主行 α=0，不受影响。
3. **flat 行 (v1/v2)**：split-view 忠实实现——f = 前半 endpoint 视图的条件均值、g = 后半视图（视图噪声不共享，无 I/M 恒等下限），与其他行共用同一数据管线。
4. **层-stage 映射**：[1,2,3]（level 0 global→layer2 出口、level 1 local→layer3、level 2 masked→layer4）。
5. **预算匹配方式**：所有变体每 parent 编码完全相同的 15 个视图（chain 3 + children 2×4 + endpoint 4），loss 装配是唯一差异 ⇒ parents/encoded views/FLOPs 构造性恒等。

## 1. 设计

- **单元**：7 变体 × 3 paired seeds {1,2,3} = 21 单元。变体：v1 final_2view、v2 final_mview、v3 additive_2view、v4 additive_mview（HAI/HFMCA 类）、v5 amdim_cross、v6 product_only、v7 product_endpoint（完整方法）。
- **固定量**：ResNet-18 (CIFAR, width 64)；三层嵌套树 global(scale .5–1)→local(.3–.8)→mask(35%)；M=4 children/边；endpoint descendants=4；batch 256；SGD lr .03 momentum .9 wd 1e-4 cosine；**200 epochs**；单 GPU（A100/H100/L40S，排除 node51）；v7 权重 β=1, γ=1, α=0, ε=1e-6，无 stop-grad。
- **评估（每单元）**：Stage-B 坐标 = calibration split (2500)；Stage-C = test split（10k，separately-estimated 点估计 + 2-fold cross-fit）；证书全套 + gauge/pairing 控制；frozen linear probe（100 epochs, SGD .1）；weighted 20-NN；backbone 测试特征 effective rank / top-eigenvalue share。塌缩判定：eff-rank < 5 或 probe < 15%。
- **QC gate（fleet 前）**：product_endpoint/seed1 的 2-epoch probe-mode 单元必须全管线通过（训练无 NaN、证书可计算、probe/kNN 输出、gauge ≤ 1e-8）。

## 2. 主要终点与预注册对比

- **E1（主，闭合缺陷）**：held-out 归一化闭合缺陷 D = δ̂_F/‖Ĉ^dir‖_F（test split 点估计；cross-fit 两折作稳健性）。**判据**：v7 的 D 在 3 个 paired seeds 上逐一小于 v3、v4、v5 各自同 seed 的 D（9 组对比全部同向），且 mean(D_v7) < 0.5 × mean(D_additive_best)。v6 的 D 一并报告（无端点项时缺陷定义仍可评，但其 Ĉ^dir 无训练激励，预期解释困难——如实报告）。
- **E2（下游不劣）**：mean linear probe(v7) ≥ max(mean probe(v1..v5)) − 1.0 pt，且 mean kNN(v7) ≥ max(mean kNN(v1..v5)) − 1.0 pt。
- **E3（塌缩记录）**：任何单元塌缩必须报告，不从均值中静默剔除；某变体 ≥2/3 seeds 塌缩 → 该行标记 collapsed 并在解释中按行披露。

## 3. 预承诺解释表（handoff §7 第 3 步决策规则）

| 结局 | 解释 |
|---|---|
| E1 ✓ 且 E2 ✓ | 完整 compositional 方法主张成立；论文走"训练原则 + 测量框架"双支柱 |
| E1 ✓、E2 ✗ | 训练 claim 收窄为"闭合可施加、下游增益有限"；论文降级 measurement-only（旗舰不变） |
| E1 ✗ | 撤训练 claim；诊断哪一环（估计噪声 vs 优化动力学）；measurement-only 论文照常成立 |
| v7 塌缩而 additive 不塌 | 如实报告为完整方法的失效模式，触发 FroSSL-M8 类机制分析 |

## 4. 产物

`results/gate1/gate1_20260816_v1/units/<variant>__seed<k>/unit.json` + checkpoints + CSV 训练日志；聚合与主表在全部 21 单元完成后单独出（freeze-before-aggregate）。

## 5. 本 Gate 明确不含（后续波次）

bootstrap-vs-independent 端点消融、closure stop-grad/EMA 消融、nested-vs-parallel 树负控训练、M/L scaling、CIFAR-100/ImageNet-100、外部 objective-matched baseline 行（PVC/MV-DHEL/SSOLE）。列于 TEST_AND_BASELINE_DESIGN_20260816.md。
