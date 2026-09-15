# Gate 1 附录 6：层级行切换可微白化（gate v6）

**日期：** 2026-08-18
**触发：** v5 完整结果——faithful flat 行 81.2/84.2%（管线与树验证通过），而同树同视图的全部层级行停在 41–48%。~36pt 差距定位于附录 3 的稳定性机制（detached 白化 + ridge 0.1）：它以学习信号换稳定。v5 flat 行证明**强视图下**可微白化（ridge 1e-3）稳定且高效；v2 时代的估计器钻空（score 6.96）发生在弱信号几何树上，条件已变。

## v6 改动（仅层级行 v3–v7）

- `loss.whitening_mode = differentiable`（白化矩阵不再 detach，梯度经由 Cholesky/三角求解回传——历史 formal 配方的行为）
- `loss.ridge = 1e-3`（历史值）
- 护栏保留：per-term score 上界（endpoint/leaf ≤ 2.5 等）+ 发散哨兵——若估计器钻空复发，probe 作业按设计失败并阻断 fleet。
- flat 行 (v1/v2) 配置等效不变（faithful 路径不读这两个键）；其 v5 单元**带出处标记导入** v6 目录（imported_from 字段），不重复消耗 6×3h GPU。

## 判据

E1/E2/E3 原样。v6 的中心问题：层级行（尤其 v7）在可微白化下能否显著逼近 flat 锚（84.2%），以及 v7 的排序优势（三个 gate 版本一致）是否延续到可信工作点。

## 验证

新增测试：可微模式梯度确实经过白化器（与 detached 模式梯度不同且有限）；population-white 输入下 score 仍在总体界附近。全部相关测试绿。
