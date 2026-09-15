# Gate 1 附录 9：终局 gate v8 规格

**日期：** 2026-08-20　**性质：** 出版主表 fleet。7 变体 × 3 paired seeds，判据 E1/E2/E3 原样（冻结稿 + 附录链）。

## 规格

- **树（满配，全行相同）**：weak 根（0.95–1.0，无 flip）→ 边0 crop(0.2–1.0)+jitter(含 hue)+逐 child flip(0.5) → 边1 crop(0.4–1.0)+gray(0.2)；**全行 M=8 端点视图**；每图 3+8+8=19 编码视图，预算构造性恒等。该树的 flat 锚已验证 89.5%（M=8 单边树）/满配三层树待本 fleet 定。
- **各行家族忠实配方**（v6 结论沿用）：flat=faithful trace；additive/amdim=逐算子 faithful trace；product_only=共享可微白化；**v7=ema 配方**（faithful bootstrap + leaf 1.0 + EMA 0.99 + α0.2 + β128），由 82.8/86.2/82.6 三种子与满配树 83.9 单种子选定。
- 200 epochs、lr 0.1、双 QC probe（v7 + flat）先行，fleet 挂 afterok。
- ema@400（在跑）作补充行，不入主表判定。

## 判定沿用

E1（9 配对 + 减半；测量地板 0.22 与选择膨胀的告诫按 ridge 标定报告随附）、E2（±1pt 非劣，probe 与 kNN）、E3（塌缩披露）。预承诺解释表不变。
