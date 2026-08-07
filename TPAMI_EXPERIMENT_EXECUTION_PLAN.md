# FMCA-AV → IEEE TPAMI 实验执行计划

**版本：** v1.2（服务器单人 / 最多 4 GPU / full-first）  
**日期：** 2026-08-07  
**对象：** *Functional Maximal Correlation with Auxiliary Variables for Single-Modality Dependence Analysis*  
**状态：** 规划完成；实验只在服务器执行，当前本地目录仅保存论文与计划

---

## 0. 执行摘要

这篇论文的 TPAMI 主线应冻结为：

\[
\boxed{
\text{辅助通道定义可保留的信息}
\rightarrow
\text{FMCA-AV 恢复该算子谱}
\rightarrow
\text{多条件采样改善估计}
\rightarrow
\text{谱用于表征学习与因素排序}
}
\]

当前执行原则是 **full first, select later**：先把能回答论文科学问题的实验组、数据集和负结果尽量做全，建立完整结果资产库；等结果稳定后再决定 TPAMI 主文、补充材料和删除项。实验仍应建立四个连续的证据门：

1. **真值恢复：** 在解析 Gaussian 和精确离散通道上验证谱、子空间和密度比恢复；
2. **机制隔离：** 在固定 parent 与固定总 view 两种预算下证明条件均值/多条件采样的独立贡献；
3. **公平 SSL：** 匹配 views、FLOPs/GPU 时间、head、backbone 与评估器后比较表征质量；
4. **谱语义：** 用 random/bottom/PCA/random rotation 和已知生成因素验证谱排序的含义。

TSD 以 **retained-dependence diagnostic** 为待检验假设，不预设它与下游准确率单调正相关。Markov/dependence map、ImageNet-1K、迁移、鲁棒性和扩展性全部纳入实验池；质量门只决定最终能写多强，不用于提前删除实验组。

TPAMI 属于 pattern analysis / machine intelligence 范围；实验应突出“可证伪的统计学习结论”，而不是仅追求 SSL SOTA。按 IEEE Computer Society regular Transactions 的 12 formatted-page 基线规划正文，完整 sweep、更多热图和工程细节放 supplemental；同时准备可运行代码、数据清单与环境锁定，以符合 IEEE 对可复现研究的建议。

---

## 1. 当前状态与硬阻塞

### 1.1 当前可用资产

- `ssl_paper.pdf`：17 页主稿，含 Appendix A–G；
- `FMCA_AV_EXPERIMENT_HANDOFF_20260806.md`：已有的理论/实验审计；
- 没有代码、README、配置、requirements、数据版本、checkpoint、训练日志或原始结果。
- 本地硬件与 Python 环境不作为实验环境；服务器最多同时使用 4 张 GPU，由单人 file-based harness 统一调度。

因此，当前 **0 个实验可以直接复跑**。第一阶段不是跑新表，而是拿到原实现或建立最小可复现实验仓库。

### 1.2 实验前必须冻结的数学定义

以下问题不解决，后续数字无法形成可靠证据：

1. 显式移除常数模态 \(\lambda_1=1\)，或在零均值空间分解 \(\rho-1\)；否则 log-det TSD 发散；
2. 用条件期望算子的 Hilbert–Schmidt/operator SVD 表述替代一般情形下不成立的 Mercer PSD-kernel 表述；
3. 用 Ky Fan/min–max/operator-SVD 原理修复 top-\(K\) 最优性证明；
4. 重复谱只能识别子空间，不能逐个比较 eigenfunction；
5. Markov 谱幂律必须写清可逆、自伴或 normal 等条件；
6. 明确 \(R_G\) 使用 mean of outer products，而不是 outer product of conditional mean；
7. 统一理论中的 raw-parent 与 SSL 实现中的 concat/mean-set/DeepSets parent。

### 1.3 当前最危险的实验主张

- Nyström 只是有限样本近似，不能作为 “exact/unbiased ground truth”；
- 9-view FMCA-AV 对多数 2-view baseline、128 head 对 2048 head、只按 epoch 比较，不能支持“更快/更省计算”；
- top-\(k\) 只对比未排序坐标的前 \(k\) 维，不能证明谱排序优于随机或 PCA；
- 第 5 epoch 的训练 TSD 不是 held-out dependence estimate；现有 crop 结果已显示 TSD 与 accuracy 非单调；
- dependence maps 目前只有视觉热图，不足以支持 explainability。

---

## 2. Claim → Experiment → Gate 总表

| ID | 论文主张 | 决定性实验 | 主指标 | GO 条件 | FAIL 时的论文措辞 |
|---|---|---|---|---|---|
| C1 | FMCA-AV 恢复辅助通道的真实谱 | 1D/2D Gaussian、有限离散通道 | top-\(K\) 谱误差、projector error、principal angles、held-out reconstruction/TSD | 误差随 \(N,M\) 稳定下降；高资源设置达到预注册精度；重复谱子空间稳定 | “approximates the spectrum empirically”，删除 exact/unbiased |
| C2 | 条件均值降低估计噪声 | fixed-parent 与 fixed-total-view 的 \(M\) 扫描 | \(P_{FG}\)/\(R_G\)/loss/gradient bias–variance | fixed-parent 下方差显著下降；fixed-budget 下存在清楚的 bias–variance/Pareto 规律 | “multi-view Monte Carlo approximation”，不声称降方差或效率 |
| C3 | 公平预算下 SSL 有竞争力 | matched views/FLOPs/time/head/backbone | linear probe、k-NN、compute-AUC、time-to-threshold | paired CI 支持优势或非劣，且不是仅由更多 views/head 造成 | 只声称按 epoch 收敛更快，或将 SSL 降为应用示例 |
| C4 | 谱按通道可预测性排序因素 | top/random/bottom/PCA/rotation；factor datasets | accuracy–dimension AUC、95%-性能最小维数、factor probe、跨 seed 子空间稳定性 | eigen-top 在预注册 \(k\) 区间稳定优于主要对照；因素顺序随通道变化符合预测 | “post-hoc ordered coordinates”，不声称语义重要性 |
| C5 | TSD 衡量保留的依赖 | analytic calibration、held-out severity、data-processing chain | calibration slope、\(R^2\)、Spearman、reliability、train–test gap | held-out 估计可校准且对破坏强度稳定响应 | 仅保留为训练目标，不作诊断量 |
| C6 | Markov 局部谱可组合 | 可逆/非可逆链 direct vs composed | Chapman–Kolmogorov residual、谱/子空间误差 | 适用假设内 direct 与 composed 一致，反例边界被正确识别 | 删除一般幂律，只保留受限命题或 direct-lag 方法 |
| C7 | dependence map 可解释模型 | 定位、deletion/insertion、randomization | Pointing Game、MaxBoxAcc/IoU、faithfulness AUC、sanity sensitivity | 同时胜过简单基线并通过随机化 | 改称 class-agnostic visualization，移补充材料 |

这些 gate 是 **解释门和组文门**，不是当前的实验裁剪门。除 Gate 0 会阻止明显无效的计算外，即使某项结果失败，也继续运行相应的诊断、边界条件和负对照，直到能解释失败来源。最终组文时再依据：C1 + C2 是否通过、C3/C4 的证据强度、C5–C7 的边界，决定正文叙事。

### 2.1 全量实验组总览

| 组 | 名称 | 主要问题 | 数据/对象 | 预期产物 |
|---|---|---|---|---|
| E0 | Estimator Integrity | 定义、矩阵估计、whitening、常数模态是否正确 | 解析矩阵、极小离散通道 | 单元测试、数值验收报告 |
| E1 | Exact Operator Recovery | 是否恢复真实谱、函数/子空间和密度比 | 1D/2D/高维 Gaussian、精确离散通道 | 谱误差、子空间误差、样本复杂度 |
| E2 | Conditional Sampling | 多个 \(Y\mid X\) 是否降低估计/梯度噪声 | Gaussian、离散通道、CIFAR10 | fixed-parent/fixed-budget bias–variance 图 |
| E3 | Objective & Numerics | trace/log-det、centering、ridge、batch/\(K\) 如何影响解 | synthetic + CIFAR10 | 全套数值与目标消融 |
| E4 | Architecture Alignment | raw/mean/DeepSets/concat 谁真正对应模型且有效 | CIFAR10/100、ImageNet100 | 聚合器、weight sharing、head 消融 |
| E5 | SSL Representation | 公平预算下表征学习是否有竞争力 | CIFAR10/100、STL10、TinyImageNet、ImageNet100/1K | linear/k-NN/compute 主表与曲线 |
| E6 | Generalization & Robustness | 表征是否可迁移、抗 corruption、低标签有效 | CIFAR-C、ImageNet-C/R/A、VOC、COCO | robustness、low-label、transfer 结果 |
| E7 | Spectral Semantics & TSD | 谱排序什么因素，TSD 测量什么 | controlled shapes、dSprites、Shapes3D、SmallNORB/MPI3D | factor probes、top-\(k\)、TSD calibration |
| E8 | Markov Dynamics | 局部谱何时可组合、何时失效 | reversible/non-reversible chains、OU、double-well | direct/composed/lag 验证与反例 |
| E9 | Dependence Maps | maps 是否定位且忠实于模型 | CUB、VOC、ImageNet localization；VGG/ResNet/ConvNeXt | localization、faithfulness、sanity checks |
| E10 | Scaling & Reproducibility | 样本、模型、维数、batch、views 的扩展规律 | synthetic 到 ImageNet-1K | scaling law、内存/吞吐、复现包 |

### 2.2 数据集池与用途

| 类别 | 数据集/生成器 | 主要可控因素 | 用于实验组 |
|---|---|---|---|
| 解析连续 | 1D Gaussian、2D isotropic/anisotropic Gaussian、10–100D correlated Gaussian | 真谱、重谱、condition number、维数 | E0/E1/E2/E3/E10 |
| 精确离散 | binary/q-ary symmetric、erasure、block、near-identity、near-independent、asymmetric cycle | 精确 SVD、非对称性、退化 | E0/E1/E2/E8 |
| 非线性 toy | two moons、circles、spiral、GMM、Swiss roll | 多模态、拓扑、非线性函数 | E1/E3/E10 |
| 可控视觉因素 | 自建 colored shapes | shape/color/position/scale/orientation/texture | E7 |
| 标准因素数据 | dSprites、Shapes3D、SmallNORB、MPI3D（至少前三个完整运行） | 位置、尺度、旋转、光照、相机、物体 | E7 |
| 小型 SSL | CIFAR10、CIFAR100、STL10 | 快速训练、细粒度、无标签数据量 | E2–E7/E10 |
| 中型 SSL | TinyImageNet、ImageNet100 | 224 分辨率、类别扩展 | E4–E7/E10 |
| 大型 SSL | ImageNet-1K | 标准规模与可比性 | E5/E6/E10 |
| Corruption/OOD | CIFAR10-C/100-C、ImageNet-C、ImageNet-R、ImageNet-A | 噪声、模糊、天气、风格、自然 OOD | E6/E7 |
| 迁移 | VOC2007 classification、VOC07+12 detection、COCO detection/segmentation | frozen/linear、低标签、密集预测 | E6 |
| 定位解释 | CUB-200-2011 boxes、PASCAL VOC segmentation、ImageNet localization | box/mask、foreground | E9 |
| 动力系统 | reversible discrete chain、directed cycle、OU、double-well Langevin | lag、可逆性、metastability | E8 |

所有外部数据在下载前记录来源、版本、license、类别列表和实际 split 文件。ImageNet100 必须固定并保存 100 类清单，不能只写“widely used subset”。不要求生成任何文件 hash/SHA。

### 2.3 全量因子轴

每个实验配置从以下轴中选取，形成统一 experiment manifest：

- channel：noise/crop/color/blur/grayscale/rotation/patch/Markov transition；
- conditional samples：\(M\)；parent batch：\(B_X\)；总 view budget：\(B_XM\)；
- output dimension：\(K\)；ridge：\(\epsilon\)；objective：trace/log-det；
- covariance：centered/uncentered；whitening：none/single/dual；
- parent aggregator：raw/mean/DeepSets/concat；
- backbone：MLP/ResNet-18/ResNet-50/VGG/ConvNeXt-T/ViT-S；
- training budget：epochs/encoded views/FLOPs/GPU-hours；
- evaluation：linear/k-NN/MLP/low-label/fine-tune/transfer/OOD；
- seed、数据 split、augmentation severity。

不做所有轴的笛卡尔积，但每个主效应必须完整覆盖；二阶交互采用预先设计的 fractional factorial/selected interaction matrix。Pilot 只用于定位稳定超参和估计运行时间，不用于删除整个实验组或数据集家族。

---

## 3. 全局冻结协议

### 3.1 数据与选择分工

所有实验必须使用四类互斥用途：

- `train`：优化网络参数；
- `calibration`：估计 \(R_F,R_G,P_{FG}\)，选择 ridge，执行 whitening/SVD；
- `validation`：仅用于训练轮数或少量预注册超参选择；
- `test`：一次性报告 held-out spectrum、TSD、谱误差和下游性能。

禁止在训练集最大化谱后，直接把训练目标或训练谱当作无偏依赖估计。

### 3.2 估计器定义

统一实现并记录：

\[
R_F^\epsilon=R_F+\epsilon I,\qquad
R_G^\epsilon=R_G+\epsilon I.
\]

- 主设置：\(\epsilon=10^{-3}\)；
- 数值敏感性：\(10^{-2},10^{-3},10^{-4},10^{-5}\)；
- 明确 centered covariance 或 uncentered second moment，二者不得混用；
- 明确常数模态移除时点；
- 统一 eigenvalue 是奇异值还是其平方；全文和代码只保留一种记号；
- 报告 condition number、numerical rank、最小特征值、NaN/分解失败率；
- 重谱评价 projection matrix/principal angles，不做逐 eigenfunction 强对齐。

### 3.3 统计规范

- 解析/toy：确认性结果 10–20 seeds；
- CIFAR10/100：确认性结果 5 seeds；
- ImageNet100：3 seeds；若方差较大则追加到 5；
- ImageNet-1K：至少 3 seeds，仅在资源允许且前序 gate 通过后运行；
- 主比较使用相同 seed/data order 的 paired differences；
- 报 mean、std、95% CI、effect size；随机子集或样本级指标使用层级 bootstrap；
- 同一 claim family 多次检验使用 Holm 校正；
- 固定 final epoch，或在运行前预注册 validation-selection；禁止挑 test 最优 epoch；
- pilot 用于确定可行范围，不与 confirmatory runs 混入最终显著性检验。

### 3.4 公平计算协议

SSL 同时使用三种预算横轴：

1. epoch；
2. total encoded views：\(E\times |D|\times V\)；
3. 实测 GPU-hours、吞吐与峰值显存；可行时补充估算 FLOPs。

每次 run 记录：backbone forward 次数、parent/view 数、head FLOPs、总参数、训练图片数、wall-clock、GPU 型号、功耗/显存（若可得）。

### 3.5 统一 SSL 评估器

- **Primary：** frozen backbone + linear probe；
- **Secondary：** weighted k-NN；
- **Supplementary：** 三层 MLP、1%/10% label fine-tuning；
- 主文明确 probe 输入是 backbone representation，projector representation 只作额外分析；
- linear probe 的 optimizer、epochs、augmentation、weight decay 和 model selection 对所有方法完全一致。

### 3.6 结果记录最小字段

每个 run 至少保存：

```yaml
run_id: <unique-id>
claim_id: C1|C2|...
config: <path>
dataset_name: <name>
dataset_version: <version-or-description>
splits: {train: <file-or-name>, calibration: <file-or-name>, validation: <file-or-name>, test: <file-or-name>}
seed: <int>
hardware: <gpu/cpu>
software_lock: <environment file>
status: success|failed|preempted
runtime_seconds: <float>
encoded_views: <int>
peak_memory_mb: <float>
metrics_file: <path>
checkpoint: <path>
failure_reason: <nullable>
```

---

## 4. Phase 0 — 获取代码、修定义、建立可复现骨架（第 1–2 周）

### 4.1 必做任务

1. 在服务器确认当前 FMCA-AV、旧 FMCA、HFMCA 的代码目录或压缩包及其人工版本标签；若无法获取，则按论文公式重建并明确“重实现”；
2. 建立独立项目目录；单人执行不要求 Git、commit、SHA 或文件 hash；
3. 固定环境：Python/PyTorch/CUDA/Lightly 版本、GPU 驱动、依赖 lockfile；
4. 建立统一 CLI：`train`、`calibrate`、`evaluate`、`aggregate`；
5. 建立配置继承，禁止在训练脚本里硬编码实验条件；
6. 实现 structured logging、checkpoint、自动重启与失败状态；
7. 为 constant mode、trace/log-det、whitening/SVD、\(R_G\)、离散真值写单元测试；
8. 找出 Table I “accuracy” 的真实评估协议，尝试复现一个 CIFAR10 数字；
9. 冻结 prior-work delta table：FMCA、HFMCA、FMCA-AV 每个公式、架构和实验的新增项。

### 4.2 建议目录

```text
fmca_av/
  configs/{toy,ssl,markov}/
  src/{models,objectives,operators,data,evaluation}/
  scripts/{train,calibrate,evaluate,aggregate}/
  tests/{unit,analytic,smoke}/
  manifests/{datasets,baselines}/
  outputs/<claim>/<experiment>/<run_id>/
  reports/<claim>/
  environment/
```

### 4.3 Gate 0

只有满足以下条件才进入大规模实验：

- 解析/离散 smoke tests 全部通过；
- log-det 不含常数模态，所有矩阵量定义一致；
- 同一个 checkpoint 可独立完成 calibration 与 test；
- 一个 CIFAR10 小规模 run 可从空环境复现；
- 日志能给出 seed、config、数据 split、GPU 与失败原因；
- 旧结果若无法复现，明确标记为 legacy/unverified，不进入新主表。

---

## 5. Phase 1 — 精确谱恢复（第 2–3 周）

### 5.1 1D Gaussian 解析真值

设置：

\[
X\sim\mathcal N(0,1),\quad
Y=X+\sqrt{\sigma}\varepsilon,\quad
r=(1+\sigma)^{-1/2},\quad
\lambda_{n+1}=r^{2n}.
\]

采用 staged design，避免 \(9\times5\times6\times4\) 的无意义全因子爆炸：

- reference：\(N=20k,M=8,K=16,\sigma=0.5\)；
- 噪声 sweep：\(\sigma\in\{0.01,0.05,0.1,0.25,0.5,1,2,5,10\}\)；
- 样本 sweep：\(N\in\{500,1k,5k,20k,100k\}\)；
- 条件样本 sweep：\(M\in\{1,2,4,8,16,32\}\)；
- 维数 sweep：\(K\in\{4,8,16,32\}\)；
- 只对 pilot 显示显著交互的 8–12 个组合补交互实验。

指标：top-\(K\) absolute/relative error、Hermite mode correlation、projector error、density-ratio reconstruction error、held-out TSD error、error-vs-\(N\)/\(M\) slope。

### 5.2 2D Gaussian 重谱

- isotropic：制造重复一阶谱，验证单个模式可旋转但 eigenspace 稳定；
- anisotropic：逐渐打破退化，验证 spectral block 分裂与排序；
- 主指标：principal angles、projection Frobenius error、blockwise eigenvalue error、seed-to-seed subspace overlap。

### 5.3 精确有限离散通道

构造 8–20 状态的四类通道，并直接对标准化联合矩阵做精确 SVD：

- near identity；
- block transition；
- high-noise near independence；
- asymmetric cycle。

扫描 3 个样本量、20 seeds。该实验是 C1 的最干净证据；Nyström、KICA、HSIC 只作估计器对照。

### 5.4 高维线性 Gaussian

将解析验证扩展到 \(d\in\{2,5,10,20,50,100\}\)，构造：

- 指定 canonical correlations 的对角通道；
- 低秩 signal + isotropic noise；
- condition number 从 \(10^1\) 到 \(10^6\) 的 ill-conditioned covariance；
- 有重复/近重复奇异值的 spectral blocks。

扫描样本维数比 \(N/d\)、\(M\)、\(K\)、ridge。报告 top-\(K\) 谱/子空间误差、数值失败率、内存和 wall-clock，用于回答方法是否只在二维图上成立。

### 5.5 非线性 toy 与可视化

two moons、circles、GMM、spiral、Swiss roll 全部运行。它们不单独承担 exact-ground-truth 主证据，但用于比较非线性几何、低密度区域、拓扑与网络容量。补充 eigenvalue error（相对高精度 numerical oracle）、aligned function/subspace correlation、density-ratio reconstruction 与 CI。

### 5.6 Baseline 与容量轴

- Nyström：网格/bin/kernel bandwidth/regularization 完整调参和敏感性；
- KICA/HSIC：不能只固定 \(\delta=0.1,\epsilon=0.1\)，用 validation 或 oracle-free criterion 选参；
- regular FMCA、DCCA、VAMP-2、一个 neural operator-SVD；
- MLP 宽度/深度、激活、训练步数与近似误差；
- 报告各方法参数量、训练时间、oracle 使用情况。

### 5.7 Gate 1（预注册建议）

- 高资源 Gaussian/离散设置 top-\(K\) 谱相对误差中位数不高于 5%（极小真值谱单独报告 absolute error）；
- projector error 不高于 0.10，且随 \(N\) 增大显著下降；
- 重复谱的 spectral-block overlap 稳定，不能以单个 eigenfunction 对齐失败判错；
- held-out TSD calibration \(R^2\ge 0.95\) 作为目标阈值；
- 若达不到，主张降为近似恢复，并报告失败区域。

以上数值是 confirmatory threshold 草案；只允许在 pilot 结束后、正式 seeds 启动前由作者组一次性冻结。

---

## 6. Phase 2 — 条件多采样机制（第 3–4 周）

### 6.1 固定 parent 数量

固定 \(B_X\)，扫描 \(M=1,2,4,8,16\)。测量：

- \(P_{FG}\)、\(R_G\) 的 bias/variance；
- trace/log-det 的 bias/variance；
- 梯度 MSE 与 cosine-to-reference；
- 最终谱误差、训练失败率和数值条件数。

目的：回答“同样数量的原始 \(X\) 下，多采样 \(Y\) 是否降低 conditional Monte Carlo noise”。

### 6.2 固定总 view 数

固定 \(B_XM=C\)：

| \(B_X\) | \(M\) |
|---:|---:|
| 1024 | 1 |
| 512 | 2 |
| 256 | 4 |
| 128 | 8 |
| 64 | 16 |

目的：回答固定总编码预算时，更多独立 parent 与更多 conditional samples 的最优折衷。

### 6.3 冻结网络参数的直接方差实验

- 用解析积分或超大 Monte Carlo 构造 reference loss/gradient；
- 每个条件重复 500–1000 个 minibatches；
- 报告 \(\operatorname{Bias}(\hat r)\)、\(\operatorname{Var}(\hat r)\)、\(\mathbb E\|\widehat{\nabla r}-\nabla r^\star\|^2\)；
- 分解 total variance 为 parent-sampling 与 conditional-sampling 两部分。

### 6.4 \(R_G\) 与聚合器消融

必须对照：

1. 正确的 \(\mathbb E[g(Y)g(Y)^\top\mid X]\)；
2. 错误的 \(\mathbb E[g(Y)\mid X]\mathbb E[g(Y)\mid X]^\top\)；
3. raw-parent；
4. mean-set；
5. DeepSets；
6. concat。

聚合器 pilot：CIFAR10、\(M\in\{2,8\}\)、3 seeds，匹配 backbone、view 数、head 参数量和 forward passes。确认性实验只保留 raw-parent、最佳 permutation-invariant 方案和当前 concat 三项。

### 6.5 Gate 2

- fixed-parent 下 \(M>1\) 的 loss/gradient variance ratio 的 95% CI 低于 1；
- 方差下降趋势与 conditional Monte Carlo 理论一致；
- fixed-budget 下给出完整 Pareto 曲线，不强求 \(M\) 单调越大越好；
- 若 concat 只因参数量或顺序信息获益，主方法改为 mean-set/DeepSets；
- \(R_G\) 错误实现必须被单元测试和实验同时识别。

### 6.6 E3 完整数值/目标消融

这组实验不因某个主配置胜出而取消，至少在 analytic Gaussian、finite channel、CIFAR10 上完整运行，并在 ImageNet100 对最佳/最差关键设置复核：

| 轴 | 设置 |
|---|---|
| Objective | trace、log-det |
| Constant handling | 显式常数模态、零均值空间、错误的未移除版本（只作 failure control） |
| Moment | centered covariance、uncentered second moment |
| Whitening | none、left only、right only、dual、dual + post-hoc SVD |
| Ridge | \(10^{-2},10^{-3},10^{-4},10^{-5}\)，加 adaptive shrinkage 对照 |
| Batch | \(64,128,256,512\) |
| Output dimension | \(32,64,128,256\) |
| Capacity | small/base/large MLP 或 projector |
| Weight sharing | shared、separate \(f/g\)、stop-gradient variants |
| Precision | FP32、AMP；关键矩阵分解强制 FP64/FP32 对照 |

主效应用 one-factor/reference 全覆盖；对 `objective × ridge × K × batch` 使用预注册的 16–32 个 fractional-factorial cells。输出 train/cal/test gap、effective/numerical rank、条件数、最小方差、梯度范数、SVD/Cholesky 失败率和总运行成本。

---

## 7. Phase 3 — 公平 SSL、迁移与鲁棒性（第 5–14 周）

### 7.1 数据与 backbone

| 执行波次 | 数据集 | Backbone | 作用 |
|---|---|---|---|
| A | CIFAR10 | ResNet-18 | 机制/架构/超参 screening |
| A | CIFAR10、CIFAR100 | ResNet-18 | 小数据完整 confirmatory 主矩阵 |
| A/B | STL10 | ResNet-18 | unlabeled split 与跨数据量泛化 |
| B | TinyImageNet | ResNet-18/50 | 分辨率和类别数中间尺度 |
| B | ImageNet100 | ResNet-50 | 224 分辨率完整公平 benchmark |
| C | ImageNet-1K | ResNet-50 | 标准大规模 benchmark |
| C | CIFAR100/ImageNet100 | ConvNeXt-T、ViT-S/16 | 跨 CNN/Transformer 架构泛化 |

所有数据集都进入结果资产库。Wave A 用来验证实现和效应，Wave B/C 在同一冻结协议上扩展，不因前面某个结果“不够漂亮”而取消；只有数据许可或算力不可得时记录为外部阻塞。

### 7.2 基线分层

**Novelty/机制必须有：**

- regular FMCA / FMCA-AV \(M=1\)；
- HFMCA 原实现；
- DCCA 或 VAMP-2 / neural operator-SVD；
- Spectral Contrastive Learning。

**同类 SSL 必须有：**

- SimCLR；
- MoCo v2 或 BYOL（至少一个 momentum/distillation family）；
- Barlow Twins 或 VICReg；
- FastSiam；
- SwAV 与 DINO 中至少一个在所有主数据集运行；在 ImageNet100/1K 尽量两者都运行，形成真正 multi-crop/multi-view 对照。

完整结果池允许超过 8 个方法；后续组文时再控制主表规模。2023–2026 新近方法先做机制相似性、可复现代码、backbone/预算可匹配性筛选，选择 1–2 个相关且能公平重训的方法加入完整池。无法公平重训的大模型可作为公开结果背景，不进入 superiority 检验。

### 7.3 三套公平协议

1. **Matched views：** 所有可支持方法统一 \(V=2\) 与 \(V=8\)；
2. **Matched compute：** 固定总 encoded views，并以实测 GPU-hours/FLOPs 二次校准；
3. **Matched architecture：** 同 backbone、相同 output dim、相近 projector 参数量；另附 native-head 结果。

所有方法共用 augmentation distribution、数据顺序、optimizer class、学习率搜索预算和评估器。若方法必须使用特殊 recipe，在 native-recipe 附表报告，但不能代替 matched 主表。

### 7.4 分两阶段运行

**Screening：** CIFAR10，3 seeds，短预算；用于排查 bug、确定每种方法一个稳定配置，不删除方法家族。  
**Confirmatory：** CIFAR10/100/STL10 各 5 seeds；TinyImageNet/ImageNet100 对全部可公平实现的基线跑 3 seeds，方差大时追加到 5；ImageNet-1K 对 FMCA-AV、直接前作和 3–4 个关键基线跑至少 3 seeds。

不得用 confirmatory test 结果继续调超参。若 screening 与 confirmatory 使用同一数据集，必须保留独立 validation split 或冻结配置后重跑新 seeds。

### 7.5 主指标与图表

- final linear-probe accuracy；
- k-NN accuracy；
- accuracy-vs-encoded-views AUC；
- accuracy-vs-GPU-hours AUC；
- time/views to 80%、90%、95% final performance；
- collapse/failure rate；
- effective rank、condition number、per-dimension variance；
- nonzero eigenvalue count 随 batch size \(B\in\{64,128,256,512\}\) 与 \(K\in\{32,64,128,256\}\) 的变化。

### 7.6 Gate 3

- “计算更高效”只能在 matched views 和 matched GPU/FLOPs 至少一项上由 paired CI 支持；
- “更稳定”必须由 seed 方差、失败率、gradient variance 或 conditioning 直接支持；
- “性能更好”以 frozen-backbone linear probe 为主，不以三层 MLP 代替；
- 若优势只在 matched epochs 出现，结论改为“用更多 views 换取更少 epochs”；
- 若只与旧基线相当，仍可保留为 C1/C2 驱动的统计方法论文，但删除 SSL SOTA 叙事。

### 7.7 E6 低标签与跨数据集迁移

对 FMCA-AV、FMCA/HFMCA 和 3–4 个关键 SSL 基线运行：

- 1%、10%、100% labels 的 linear probe；
- 1%、10% labels 的 end-to-end fine-tuning；
- ImageNet100/1K 预训练 → VOC2007 multi-label classification；
- ImageNet 预训练 → VOC07+12 detection；
- 资源允许时加入 COCO Mask R-CNN detection/instance segmentation。

统一 backbone、检测框架、schedule、augmentation 和随机种子。报告 AP/AP50/AP75、低标签 accuracy、收敛速度与方差，不只报告单个最终数字。

### 7.8 E6 corruption 与 OOD

- CIFAR10-C/100-C：15 corruption × 5 severity，报告 mCE、relative mCE、每类 corruption；
- ImageNet-C：主 corruption family 与全量 mCE；
- ImageNet-R/A：top-1、calibration error；
- 对输入增强 channel 与测试 corruption 的对应关系做分组分析，例如 blur-channel spectrum 是否预测 blur robustness；
- 报告 clean–corruption trade-off，不把 clean accuracy 差异误当 robustness。

### 7.9 复杂度和 scaling

系统测量：

- \(M\in\{1,2,4,8,16\}\) 对 images/s、GPU-hours、峰值显存的影响；
- \(K\in\{32,64,128,256,512\}\) 对矩阵构造、whitening/SVD、\(O(K^3)\) 部分的影响；
- ResNet-18/50、ConvNeXt-T、ViT-S 的参数、FLOPs、吞吐；
- 单卡与单节点 2/4-GPU DDP 的 scaling efficiency；不设计多节点实验；
- dependence calibration 与 map 生成的离线时间/内存。

输出单独的 complexity table 和 log-log scaling curves，为最终论文的效率结论提供完整依据。

---

## 8. Phase 4 — 谱排序、因素语义与 TSD（第 5–12 周，与 Phase 3 并行）

### 8.1 六类低维表示对照

对同一个完整表示、同一个 probe，比较：

- Eigen-top-\(k\)；
- Eigen-bottom-\(k\)；
- Random-\(k\)（每个 checkpoint 重复抽样 100 次）；
- PCA-top-\(k\)；
- Unranked-first-\(k\)；
- Random-rotation-first-\(k\)（重复 20–50 个正交旋转）。

\(k\in\{1,2,4,8,16,32,64,128\}\)。报告 accuracy–dimension curve/AUC、达到 full representation 95% 性能的最小维数、对随机对照的 paired difference。

### 8.2 已知生成因素

采用三层设计：

- 可控彩色形状数据：低成本、可精确指定 shape/color/position/scale/orientation；
- dSprites + Shapes3D：标准离散生成因素；
- SmallNORB + MPI3D：相机、光照、真实/仿真物体的外部验证。

逐个改变 auxiliary channel：color jitter、crop、rotation、blur、grayscale、additive noise，以及保持/破坏单个生成因素的定制通道。分别对各因素做 linear/MLP probe，并检验预注册方向：被通道破坏的因素应向低谱/低可预测性区域移动，被保留的因素应在高谱区域出现。

除分类/回归精度外，报告：

- factor-wise predictability vs eigenvalue rank；
- 每个 factor 达到 95% full-performance 的最小 spectral dimension；
- factor subspace 与 spectral block 的 canonical correlations；
- DCI/SAP/MIG 只作补充 disentanglement 指标，不代替通道可预测性检验；
- 同一数据上交换 auxiliary channel 后谱排序的可重复变化。

### 8.3 跨 seed 谱稳定性

- eigenvalue rank correlation；
- top-\(k\) eigenspace overlap；
- spectral-block stability；
- bootstrap CI。

不要求重谱块内部坐标逐维一致。

### 8.4 TSD 的三层实验

1. **解析 calibration：** exact Gaussian TSD vs held-out FMCA-AV/Nyström/其他 scalar dependence estimator；
2. **固定 encoder 的 channel diagnostic：** 在 test 上改变 severity，隔离“通道本身保留多少依赖”；
3. **重新训练后的 utility trade-off：** crop、color jitter、blur、rotation、grayscale 各 7 个强度、5 seeds 完整重训，画 utility–TSD；additive noise 再做一套解析/图像对应实验。

为控制成本，先在 CIFAR10 完成 `5 augmentations × 7 severities × 5 seeds` 全矩阵；CIFAR100 复跑全部 severity 但可用较短预算；ImageNet100 对每种 augmentation 选 low/medium/high 三档、3 seeds 复核。这里的“选三档”是跨规模复核设计，不是根据结果挑漂亮点，三档在 CIFAR 实验前预注册。

所有 TSD 均在 calibration/test 重新估计。报告 calibration slope、\(R^2\)、absolute error、Spearman、monotonicity violations、test–retest reliability、train–test gap。

### 8.5 Data-processing chain

构造 \(X\to Y_1\to Y_2\to Y_3\)，例如：

- 逐级加 Gaussian noise；
- crop → crop+color → crop+color+blur。

同时画 held-out TSD 与 downstream utility。预期允许倒 U/Pareto，不预注册“越大越好”。

### 8.6 Gate 4/5

- C4：Eigen-top 的 curve/AUC 在预注册低维区间稳定优于 random、bottom 和 PCA，且因素排序随通道改变；
- C5：held-out TSD 对解析真值可校准、test–retest 可靠，并对破坏强度给出稳定响应；
- 若 TSD 与 utility 呈倒 U，作为重要边界结果正面报告；
- 若 held-out TSD 本身不可靠，删除诊断量叙事，仅保留训练目标。

---

## 9. Phase 5 — Markov 与 dependence maps（第 9–14 周，完整实验轨）

### 9.1 先做可控 Markov

- reversible discrete chain：\(\tau\in\{1,2,4,8,16\}\)，比较 exact、direct lag-\(\tau\)、recursive/composed、理论幂律；
- non-reversible cycle：作为反例，界定幂律失败；
- 指标：spectrum/subspace error、density-ratio error、Chapman–Kolmogorov residual；
- OU、double-well Langevin、metastable multi-well 全部进入连续动力系统实验；比较 direct lag decomposition、VAMP 左右奇异函数与错误的简单幂律；
- 扫描轨迹长度、采样间隔、lag、噪声、状态离散化/网络容量和非平衡初始分布。

### 9.2 再做 CNN maps

1. plain CNN/VGG 先验证无 skip 的递归；
2. ResNet 明确按 DAG、identity/residual paths 或其他可证明方案处理 skip；
3. ConvNeXt 检查现代卷积架构；ViT patch tokens 作为扩展，明确它不是原 CNN Markov 假设的直接实例；
4. 监督、FMCA-AV、SimCLR/VICReg/DINO checkpoint 都运行，同一图像与统一色标比较；
5. 对早期层与最终层同时做 direct estimate 和 recursive composition；
6. 指标：map rank correlation、normalized L2、top-region IoU、时间/内存。

### 9.3 定位与 faithfulness

数据：CUB bounding boxes、PASCAL VOC segmentation、ImageNet localization 三套都运行；资源允许时加入 COCO segmentation 子集。  
定位：Pointing Game、MaxBoxAcc、pixel IoU/AUPRC、foreground energy ratio。  
基线：random、center Gaussian、edge/gradient、activation norm、class-agnostic PCA/Eigen-CAM。  
faithfulness：top/bottom/random/center deletion，报告 representation cosine drop、logit drop、insertion/deletion AUC。

### 9.4 Sanity checks

- layer-wise parameter randomization；
- fully random weights；
- random-label model。

### 9.5 Gate 6

完整实验轨不因中途失败停止。只有 direct-vs-recursive、定位、faithfulness、randomization 四项同时通过，未来组文时才使用 explainability；任一关键项失败，结果仍作为适用边界/负结果保留在资产库，最终改称 “class-agnostic local-to-global dependence visualization”。

---

## 10. 运行规模与算力规划

### 10.1 不做全因子暴力 sweep

所有工作包采用：

1. 单因素/reference screening；
2. 只补显著交互；
3. pilot seeds 与 confirmatory seeds 分离；
4. successive halving 只淘汰明显错误的超参数实例，不淘汰实验组、数据集家族或 baseline family；
5. 所有 confirmatory comparisons 使用完整预注册预算重跑；
6. 负结果、失败 run 和不稳定区域同样进入结果资产库。

### 10.2 建议资源档位

| 档位 | 服务器资源假设 | 可完成范围 | 预计墙钟时间 |
|---|---|---|---|
| 1 GPU | 单卡串行 | 全实验组串行；ImageNet-1K/COCO 排队最后 | 24–36 周 |
| 2 GPU | 两个单卡任务或一个 2-GPU DDP | toy/CIFAR 与 ImageNet 交替推进 | 18–26 周 |
| 4 GPU 上限 | 四个单卡任务、两个 2-GPU 任务或一个 4-GPU DDP | E0–E10 完整队列 | 12–18 周 |

这些是排程区间，不是 GPU-hour 报价。拿到代码和硬件后，先用 20/50 epoch profiling run 估计：

\[
\text{Total GPUh}=\sum_i (\text{runs}_i\times\text{measured GPUh per run}_i).
\]

第一周必须生成准确的 run-count × runtime 表，并给每个实验组建立队列；算力只决定启动时间和并行度，不提前决定是否保留 ImageNet-1K、Markov 或 Case 3。

### 10.3 初始运行量级（用于排程，不是最终预算）

| 实验池 | 预计 run/重复量级 | 计算特征 |
|---|---:|---|
| E0/E1 analytic + discrete + nonlinear | 800–1,200 short runs | 主要 CPU/单卡，适合大规模 seeds |
| E2/E3 mechanism/numerics | 200–400 training configs + frozen-batch repeats | CIFAR 前可完成多数 |
| E4/E5 CIFAR/STL/TinyImageNet | 150–250 GPU runs | 5-seed confirmatory 为主 |
| E5 ImageNet100 | 50–100 GPU runs | 3–5 seeds、matched budgets |
| E5/E6 ImageNet-1K/transfer/OOD | 20–50 pretraining runs + 多个 eval jobs | 主要算力来源 |
| E7 factor/TSD | 200–350 GPU runs | CIFAR severity 全矩阵 + factor datasets |
| E8 Markov | 200–500 short/medium runs | 多为 CPU/单卡 |
| E9 maps | 100–200 estimator/eval jobs | checkpoint 复用，训练成本较低 |

拿到代码后用真实 profiling 替换以上区间，并自动从 manifest 生成甘特图和 GPU-hour 预算。

### 10.4 执行波次（不是论文优先级）

- **Wave 0：** Phase 0、E0；确保后续结果有效；
- **Wave A：** E1–E4、CIFAR/STL、factor toy；
- **Wave B：** ImageNet100/TinyImageNet、完整 TSD、controlled Markov；
- **Wave C：** ImageNet-1K、transfer/OOD、多架构、quantitative maps；
- **Wave D：** 依据前面发现补边界条件、复跑高方差单元、生成最终结果资产。

算力不足时把 Wave C/D 延后或减少并行度，不从总计划删除。任何资源导致的未完成项都标记为 `BLOCKED_RESOURCE`，与科学上的 `FAIL/INCONCLUSIVE` 分开。

---

## 11. 单人执行时间线与 GPU 队列

| 周 | 估计器/理论轨 | SSL/扩展轨 | 语义/TSD/Markov 轨 | Maps/分析轨 | 里程碑 |
|---:|---|---|---|---|---|
| 1–2 | 常数模态、operator-SVD、\(R_G\)、unit tests | 获取代码、环境、logging、legacy smoke | 数据/license/manifest | 自动汇总模板 | G0 |
| 3–4 | 1D/2D/高维 Gaussian、finite channels | aggregator/\(M\) CIFAR pilot | controlled shapes、TSD oracle | figure/table QA | E0/E1 首批 |
| 5–6 | nonlinear toy、数值消融 | CIFAR10/100/STL complete matrix | factor datasets、fixed-parent variance | 失败区域分析 | G1/G2 |
| 7–8 | estimator scaling | TinyImageNet/ImageNet100 matched views | TSD CIFAR severity、reversible Markov | map direct-vs-recursive 原型 | Wave A 完成 |
| 9–10 | — | ImageNet100 matched compute、ConvNeXt/ViT | TSD ImageNet100、non-reversible/OU | VGG/ResNet map 定位 | G3/G4/G5 |
| 11–12 | 边界条件补跑 | ImageNet-1K 启动、low-label/transfer | double-well/MPI3D | CUB/VOC faithfulness/randomization | Wave B 完成 |
| 13–14 | 高方差单元追加 seeds | ImageNet-1K、COCO、OOD/corruption | Markov continuous confirmatory | ImageNet localization/ConvNeXt/ViT maps | G6 |
| 15–16 | 全部 confirmatory audit | 计算公平性复核 | 负结果/边界归档 | 全结果资产冻结 | Wave C/D |
| 17–18 | 必要复跑 | 必要复跑 | 必要复跑 | 组文候选表/图，但暂不删原始结果 | Experiment freeze |

时间线从“服务器代码目录可运行”之日开始计，不从本文档日期开始计。表中“并行”仅指 harness 在最多 4 张 GPU 上并行任务，不代表多人分工；实验设计、验收和组文均由一人完成。

---

## 12. 结果资产设计与未来组文接口

本阶段不按版面删除实验。下面只规定每个实验组必须生成哪些标准化表图；“主文/补充”标签是未来组文时的候选位置，实验冻结前不生效。

### 12.1 候选叙事顺序

1. **Exact Recovery of the Auxiliary-Variable Spectrum**  
   Gaussian + discrete truth；谱/子空间/reconstruction；
2. **Why Conditional Sampling Works**  
   fixed-parent、fixed-budget、loss/gradient variance、aggregator；
3. **Self-Supervised Representation Learning under Matched Budgets**  
   matched views/head/compute、linear probe、stability；
4. **Spectral Ranking and Auxiliary-Channel Semantics**  
   top/random/bottom/PCA、factor dataset；
5. **TSD as Retained Dependence**  
   calibration、severity、data-processing、utility trade-off；
6. **Markov Composition and Dependence Localization**  
   仅在 Gate 6 通过时保留。

### 12.2 必须生成的标准图表

- Fig. 1：analytic/discrete truth 的谱误差、子空间误差、样本复杂度；
- Fig. 2：\(M\) 对 matrix/loss/gradient variance 的 fixed-parent 与 fixed-budget 曲线；
- Table 1：matched-view/head 的 CIFAR10/100 linear probe；
- Table 2：ImageNet100 matched compute + GPU-hours/encoded views；
- Fig. 3：accuracy vs encoded views/GPU-hours；
- Fig. 4：top/random/bottom/PCA 的 accuracy–dimension curve 与 factor alignment；
- Fig. 5：held-out TSD calibration、severity 和 utility–TSD；
- Fig. 6：Markov direct/composed + map 定量结果，无论正负都生成；
- Table 3：low-label、transfer、corruption/OOD；
- Table 4：复杂度、吞吐、显存、FLOPs、GPU-hours；
- Fig. 7：数据规模/模型规模/\(M,K,B\) scaling；
- Failure atlas：定义失败、数值失败、负结果和适用边界汇总。

### 12.3 完整结果包

- 全部超参、环境与数据 split 文件/名称；
- 全 seeds/CI、失败 runs；
- ridge/batch/K/capacity 敏感性；
- native-head 与 native-recipe 结果；
- two moons/GMM/spiral 可视化；
- 更多 TSD 通道与所有 heatmaps；
- 定理假设、证明和完整 reproducibility checklist。

每张标准图表同时保存：原始逐 run 数据、聚合 CSV/Parquet、生成脚本、矢量图、caption 草案和 claim ID。未来即使不进入论文，也不从仓库删除。

---

## 13. 风险登记与应对

| 风险 | 严重度 | 早期信号 | 应对 |
|---|---:|---|---|
| 与 FMCA/HFMCA novelty 重合 | 致命 | 同公式/同架构/同数字已发表 | Week 1 完成逐项 delta；把贡献集中到 conditional sampling + exact recovery + fair evidence |
| 常数模态/log-det 定义不一致 | 致命 | objective/TSD 出现有限但理论应发散 | 先修定义和单测，旧 TSD 全部重算 |
| 无法取得旧代码/结果 | 高 | 无配置/checkpoint/日志 | 重实现并标注 legacy unverified；不把旧表当新证据 |
| 9-view 优势来自额外算力 | 高 | matched views 后优势消失 | 收窄为 sample-efficiency/epoch trade-off，不声称计算效率 |
| concat 与概率模型不一致 | 高 | permutation/order ablation 影响大 | 改为 mean-set/DeepSets，或重写模型定义 |
| 非零谱数量由 batch rank 决定 | 高 | effective rank 跟 \(B,K\) 同步 | 系统扫 \(B,K\)，限制可声称的有效维数 |
| TSD 与 utility 不单调 | 中 | crop 已出现反例 | 正面定位为 retained dependence；报告倒 U/Pareto |
| maps 只反映边缘/中心偏置 | 高 | random weights 仍有相同热图 | 降级为 visualization，移 supplement |
| baseline 工程质量不一致 | 高 | 某方法复现显著低于公开 recipe | 统一 codebase、报告 native recipe、做小规模复现验收 |
| confirmatory 后继续调参 | 高 | test 结果驱动 config 变化 | 锁 config 和 seed manifest；任何改动建立新实验批次并声明 exploratory |

---

## 14. 质量门、诊断动作与未来组文决策

- **G0 未通过：** 唯一会暂停下游计算的 gate；先解决代码、定义、数据和日志；
- **G1（恢复）失败：** 不隐藏也不立即停全项目；追加容量/优化/样本复杂度诊断，定位是估计器失败还是函数类不足；
- **G2（机制）失败：** 继续完成 fixed-budget、聚合器和 SSL，判断实际收益是否来自架构/多视图而非条件均值；
- **G3（SSL）混合：** 继续 transfer/OOD/语义实验，完整记录“在哪些预算/任务有益”；
- **G4（排序）失败：** 继续 factor channel 交换实验，判断是否只恢复子空间而不提供稳定坐标排序；
- **G5/TSD 失败：** 完成 severity 与 data-processing 矩阵，报告训练量、held-out 量和 utility 的分离；
- **G6/maps 失败：** 完成所有 sanity/negative controls，形成可信的失败边界。

所有实验完成后才进行组文决策：哪些结果支持主贡献、哪些作为局限/反例、哪些仅作为补充。Gate 不再承担当前“是否做该实验”的筛选功能。

---

## 15. 未来 72 小时的具体动作

第一步只执行 `INSTRUCTION_01_SERVER_HARNESS.md`，建立服务器 file-based harness 并通过验收；在它完成前不提交正式实验。

1. 在服务器准备当前 FMCA-AV、FMCA、HFMCA 代码目录/压缩包、checkpoint 和旧 logs；
2. 提供 GPU 数量/型号/预计可用时段，以及 ImageNet100/1K 数据位置与授权状态；
3. 确认 Table I accuracy 的真实评估器、训练 recipe、seed 和数据 split；
4. 确认当前实现是否移除常数模态、如何计算 log-det、\(R_G\) 和 ridge；
5. 建立服务器项目目录与环境 lock，跑一个 5–20 epoch CIFAR smoke test；
6. 同时实现 1D Gaussian 与有限离散通道 exact oracle/unit tests；
7. 生成第一版 `experiment_manifest.csv` 与 profiling-based GPU-hour 预算；
8. 将 E0–E10 全部录入队列，并依据 GPU/数据可用性标注启动波次；Case 3 与 ImageNet-1K 不提前删，只决定排程。

在以上信息未齐时，可以并行完成 exact-oracle、统计汇总器、日志框架和预注册模板；不应提前启动无法比较的大规模 SSL runs。

---

## 16. 每轮实验的结论模板

每个工作包完成后必须提交一页结论卡：

```text
Claim ID:
Question:
Primary estimand:
Pre-registered positive/negative criterion:
Code version label / config / data split / seeds:
Compute budget:
Main result with 95% CI:
Numerical stability and failed runs:
Decision: PASS / FAIL / INCONCLUSIVE
Narrowest supported statement:
Unsupported stronger statement:
Required manuscript change:
Next experiment (if any):
```

论文中的每一句强结论都必须能反向链接到一张通过 gate 的结论卡。

---

## 17. 完成定义

实验阶段只有在以下条件同时满足时才算完成：

- E0–E10 均达到 `COMPLETED`，或因无法获得数据/算力/许可被明确标记为 `BLOCKED_EXTERNAL`；不能因结果不显著而标记完成前删除；
- C1/C2 的 confirmatory gates 已按预注册标准完成评估，并对 PASS/FAIL/INCONCLUSIVE 给出诊断；
- 所有主表来自冻结配置、独立 test、规定 seeds；
- matched-view/head/compute 的公平性表完成；
- spectral ranking 至少有 random、bottom、PCA 三类强对照；
- 所有 TSD 均为去常数模态后的 held-out 估计；
- 每张主图可由单一命令从原始 run artifacts 重建；
- 代码、环境、数据 split、checkpoint、失败 run 清单可审计；
- 主文强度与 gate 结果一致；失败结果不被隐藏；
- Case 3 无论是否过 gate，都已完成 direct/composed、定位、faithfulness、sanity 和负对照；组文时再决定措辞与位置；
- ImageNet-1K、transfer/OOD、多架构结果均已运行或有可审计的外部阻塞记录；
- 完整结果索引能检索所有正结果、负结果、失败 run 和资源阻塞项。

这套完成定义优先保证“实验资产尽量完整、每个主张与边界都有可复现证据”。论文结构和版面筛选发生在 experiment freeze 之后。
