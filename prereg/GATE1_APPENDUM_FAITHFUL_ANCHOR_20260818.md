# Gate 1 附录 5：可信锚工作点（gate v5，与 v3/v4 并列报告）

**日期：** 2026-08-18
**触发：** v4 中期结果显示光度通道只解释了小部分差距（flat 行 ~40% vs 本仓库历史 formal flat FMCA-AV 85.4/89.4%）。用户 steer（2026-08-18）：v1/v2 与历史差距过大则 gate 缺乏说服力，做下一版。经逐项比对历史 formal 配方（configs/ssl/cifar10_reference.json），残余差距定位于四项：单层 RRC 0.08–1.0 vs 三重复合裁剪+35% mask；f 侧 = 全视图均值 + 独立 f_head vs split-half 无 f_head；trace 目标（可微白化 ridge 1e-3）vs detached 白化 ridge 0.1；lr 0.1 vs 0.03。

## v5 改动（判据 E1/E2/E3、7×3 设计、200 epochs、预算匹配不变）

1. **树**：weak 根（crop 0.95–1.0 + flip）→ edge0 global crop **0.2–1.0** + color jitter(0.8/0.5) → edge1 local crop **0.4–1.0** + grayscale(0.2)。叶复合面积 0.08–1.0 = 经典单层 RRC 范围。**mask 通道从主树移除**（信息破坏嫌疑，降级为后续消融）；注意 v5 树反而更贴合冻结本体 §2.1 的原型（"原图/弱增强 → global crop → 内部 local crop"）。原 gate 预注册写的是 global→local→masked——此为披露的树修订。
2. **flat 行 (v1/v2) 换 faithful_trace 配方**：逐项复刻历史 formal 估计器——f = f_head(投影视图均值)（parent_feature_source=g, mean）、`estimate_moments` + `trace_score`（可微白化，ridge 1e-3）、无 γ 项。它们的角色从"同族对照"升级为**可信锚**：若 v1/v2 在 v5 树上逼近 85%（历史 M=2 的水平），则训练机制与树都被验证；若仍显著偏低，残余差距被隔离到树本身——两种结局都有诊断价值。
3. **全行 lr 0.03 → 0.1**（历史值；所有行共享，保持匹配）。warmup 10ep 保留（历史无 warmup，记录为偏差）。
4. 哨兵：faithful trace 的界是 K=128 非 1，新增 flat_trace_score ≤ 200、train/loss ≤ 300（层级行的 per-term 界不变）。
5. 层级行 (v3–v7) 机制不变（v3 gate 已验证其稳定与判据通过）。

## 风险记录

faithful_trace 的可微白化在弱信号下有估计器钻空风险（v2 教训）；在经典强度视图下历史证据（85–90%，6 个 formal runs）表明稳定。probe QC + 哨兵值守。

## 验证

33 项相关测试全绿（faithful 分支：f_head 构建/梯度、与 formal 估计器逐值一致、层级行不分配 flat 头）。
