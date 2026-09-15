# Gate 1 附录 4：add-one 光度层级树（gate v4，与 v3 并行报告）

**日期：** 2026-08-17
**触发：** v3 中期结果（15/21 单元）显示全变体工作点过低（flat 行 37–42% vs 本仓库同 200-epoch、同 probe 协议的历史 flat FMCA-AV formal 行 **85.4%（M=2）/ 89.4%（M=8）**）。定位主因：v3 视图树为纯几何（嵌套 crop+flip+mask），无光度增强 → 颜色直方图捷径。**v3 fleet 照常完成并完整报告**（其 E1/E2 同族对比在低工作点仍有效）；v4 是同一 7×3 设计在修正树上的重跑，两者并列报告，不覆盖。

## v4 树（唯一改动；HAI 式 add-one 层级，Markov-by-construction 不变）

- root（X₀）：global crop scale **0.2–1.0**（v3 为 0.5–1.0）+ flip；无光度
- edge 0（X₀→X₁ local crop 0.3–0.8）：**+ 逐 child color jitter（p=0.8, strength 0.5；亮度/对比度/饱和度，tensor 原生实现，不含 hue——已记录偏差）**
- edge 1（X₁→X₂ mask 35%）：**+ 逐 child grayscale（p=0.2）**；blur 关闭（CIFAR 惯例，与历史 formal 配置一致）

光度细化是条件于 realized parent 的合法随机通道；"增强层级"与"深度层级"自此语义对齐（HAI/HFMCA 的 hierarchical augmentation 被链语义收编）。判据、预算匹配（全变体同编码 15 视图）、E1/E2/E3、200 epochs、评估协议全部照旧；QC probe 先行，fleet 挂 afterok。

## 锚点行

历史 flat FMCA-AV formal 结果（85.4/89.4%，SimCLR 全套增强 + 单层 RRC 0.08–1.0）作为 recipe-level 锚点行进入 gate 主表，注明增强族差异，不参与 E1/E2 判定。

## 验证

view_tree 光度测试 3 项新增（灰度通道相等、兄弟 child 光度独立、盒子/确定性不受光度影响），14 项全绿。
