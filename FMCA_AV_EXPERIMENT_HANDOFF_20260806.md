# FMCA-AV 手稿与实验补全项目 Handoff

**日期：** 2026-08-06  
**源文件：** `ssl_paper.pdf`  
**手稿标题：** *Functional Maximal Correlation with Auxiliary Variables for Single-Modality Dependence Analysis*  
**当前任务：** 完成并重构实验部分，使核心主张逐项获得可证伪、可复现、计算公平的证据。当前阶段不应优先扩写正文，而应先冻结定义、补齐决定性实验并校正可声称的结论。

---

## 0. 一句话项目定义

这篇工作提出 **FMCA-AV**：当只有单一数据源 \(X\) 时，主动设计一个随机辅助通道

\[
Y\sim p(Y\mid X),
\]

再学习该通道中能够从 \(X\) 稳定传递到 \(Y\) 的正交函数模式。每个模式由左右函数 \(\phi_k(X),\psi_k(Y)\) 和谱值 \(\lambda_k\) 描述；谱值越大，表示该信息模式在所选辅助通道下越稳定、越可预测。

手稿给出三种辅助变量构造：

1. 加性高斯噪声；
2. 图像随机增强；
3. 多尺度图块形成的 Markov 过程。

对应三类用途：

1. 验证谱恢复；
2. 自监督表征学习；
3. 预训练网络内部特征的 dependence-map 分析。

---

## 1. 当前手稿状态

### 1.1 结构状态

手稿并非字面意义上的“只写了一半”。现有 17 页已经包含：

- Abstract、Introduction；
- density-ratio decomposition 定义；
- trace 与 log-determinant 目标；
- Markov 扩展；
- 三个应用案例；
- CIFAR10、CIFAR100、ImageNet100 实验；
- 多个理论和实验附录。

### 1.2 真正未完成之处

科学闭环仍未完成，主要缺口是：

1. **关键数学定义和命题仍有漏洞。**
2. **FMCA-AV 相对旧 FMCA/HFMCA 的真正增量没有隔离。**
3. **多视图条件均值的机制贡献没有直接实验。**
4. **SSL 比较没有匹配 views、FLOPs、head 维数和评估协议。**
5. **谱排序、TSD、dependence map 的强结论均缺少决定性对照。**
6. **Case 3 目前主要是定性图，不足以称为 explainability。**

因此，项目应被视为：**方法框架和初步现象已经存在，但实验论证尚未完成。**

---

## 2. 对方法的统一技术理解

### 2.1 密度比

方法分解：

\[
\rho(x,y)=\frac{p(x,y)}{p(x)p(y)}.
\]

- \(\rho=1\)：局部上与独立机会水平一致；
- \(\rho>1\)：该对样本比独立情况下更常共同出现；
- \(\rho<1\)：共同出现得更少。

互信息只把 \(\log\rho\) 平均成一个标量；FMCA-AV 试图进一步把依赖拆成一组正交函数模式。

### 2.2 正确的算子解释

最干净的数学对象是条件期望算子：

\[
(Tg)(x)=\mathbb E[g(Y)\mid X=x].
\]

其奇异函数满足：

\[
T\psi_k=\sqrt{\lambda_k}\phi_k,
\qquad
T^\ast\phi_k=\sqrt{\lambda_k}\psi_k.
\]

因此：

- \(\phi_k\)：原始变量一侧的函数模式；
- \(\psi_k\)：辅助变量一侧的对应模式；
- \(\sqrt{\lambda_k}\)：条件期望算子的奇异值；
- \(\lambda_k\)：其平方，可理解为该模式穿过辅助通道后保留的强度。

这更接近 **HGR maximal correlation / principal inertia components / nonlinear CCA / operator SVD**，而不是普通意义上的 Mercer kernel eigen-decomposition。

### 2.3 神经网络目标

网络输出：

\[
f_\theta(X)\in\mathbb R^K,
\qquad
g_\omega(Y)\in\mathbb R^K.
\]

矩阵：

\[
R_F=\mathbb E[f(X)f(X)^\top],
\quad
R_G=\mathbb E[g(Y)g(Y)^\top],
\quad
P_{FG}=\mathbb E[f(X)g(Y)^\top].
\]

FMCA-AV 的核心计算变化是：

\[
P_{FG}
=
\mathbb E_X\left[f(X)\,\mathbb E[g(Y)\mid X]^\top\right].
\]

也就是对同一个 \(X\) 采样多个 \(Y\)，用 Monte Carlo 平均近似条件均值。

白化交叉矩阵：

\[
C=R_F^{-1/2}P_{FG}R_G^{-1/2}.
\]

Trace 目标：

\[
r_T=\|C\|_F^2=\sum_k\lambda_k.
\]

Log-det 目标：

\[
r_L=-\log\det(I-CC^\top)
=\sum_k-\log(1-\lambda_k).
\]

训练后再通过 whitening 与 SVD，把网络学到的子空间旋转为有序谱坐标。

### 2.4 重要解释纪律

谱值并不表示“绝对语义重要性”。更准确的表述是：

> \(\lambda_k\) 排序的是某个函数因素在当前辅助通道 \(p(Y\mid X)\) 下的可预测性或稳定性。

改变增强机制，谱排序也会改变。

---

## 3. 当前手稿的五个主要主张及证据强度

### 主张 A：FMCA-AV 能恢复真实依赖谱

**当前证据：** two moons、GMM、spiral 上与离散 Nyström 结果视觉相似。  
**问题：** Nyström 本身是有限样本、网格和直方图近似，不能支撑“unbiased”或“exactly matches ground truth”。

### 主张 B：条件均值使训练更快、更稳定

**当前证据：** 9-view FMCA-AV 按 epoch 比多数 2-view SSL 方法更快。  
**问题：** 没有固定总 views、FLOPs、GPU 时间，也没有梯度或矩阵估计方差实验；无法区分目标函数收益和“看了更多图像”的收益。

### 主张 C：谱值能够给有用特征排序

**当前证据：** 排序后的前 \(k\) 维优于未排序表示的前 \(k\) 维。  
**问题：** 原始坐标本来没有顺序；缺少 random、bottom、PCA、random rotation 等强对照。

### 主张 D：TSD 能衡量增强强度，并与表示质量相关

**当前证据：** 增强越强，TSD 越低。  
**关键反例：** crop 从 0 增至中等强度时，TSD 下降，但分类准确率大幅上升。  
**正确结论：** TSD 更像“保留了多少依赖”，不是“表示质量分数”。下游效用可能与 TSD 呈倒 U 型或 Pareto 关系。

### 主张 E：多尺度 dependence map 能解释网络

**当前证据：** 多张定性热图，看起来覆盖对象；监督、自监督、不同 ResNet 之间图形相似。  
**问题：** 无定位指标、faithfulness、随机化 sanity check、直接估计对照。当前最多能称为 qualitative visualization。

---

## 4. 理论和定义层面的主要阻塞项

### 4.1 常数模式必须显式移除

第一模式恒为：

\[
\lambda_1=1,
\qquad\phi_1=\psi_1=1.
\]

若把它放入：

\[
-\log(1-\lambda_1),
\]

目标会发散。因此所有非平凡 TSD 应定义为：

\[
\mathrm{TSD}_{\mathrm{trace}}
=
\sum_{k=2}^{K}\lambda_k,
\]

\[
\mathrm{TSD}_{\log}
=
-\sum_{k=2}^{K}\log(1-\lambda_k).
\]

或者直接在零均值函数空间中工作，分解 \(\rho-1\)。

### 4.2 Mercer theorem 表述不适合一般跨空间问题

\(X\) 与 \(Y\) 可能位于不同空间，\(\rho(x,y)\) 也不是普通的同域对称 PSD kernel。应改用 Hilbert–Schmidt 条件期望算子的 SVD。

### 4.3 Proposition 3 的证明不足

现有 Cauchy–Schwarz + induction 不足以证明前 \(K\) 个有序模式。应使用：

- Ky Fan variational principle；
- Schmidt–Eckart–Young；
- von Neumann trace inequality；
- operator-SVD 的 min–max 形式。

同时必须承认：

- 符号不唯一；
- 重复谱值时只能识别整个 eigenspace；
- 受限网络函数类只能得到近似最优子空间。

### 4.4 Markov 谱幂公式缺少假设

一般非对称 Markov 算子的奇异值不满足简单幂律。要保留该结论，需要增加可逆、自伴或 normal operator 等条件；否则应采用 VAMP 式左右奇异函数或直接估计不同 lag。

### 4.5 多变量扩展目前是猜想性构造

附录中把乘积改为负平方指数和并加入任意系数 \(\alpha\)，没有从原始密度比分解推出。该部分不宜作为正式定理，建议删除、降级为未来工作或重建证明。

### 4.6 SSL 具体实现与概率模型不一致

理论写的是：

- \(f_\theta\) 输入原始 \(X\)；
- \(g_\omega\) 输入增强视图 \(Y\)。

实际架构却把多个增强的 backbone 输出 concat 后，再通过 head 形成 parent-side representation。必须通过架构 ablation 明确：真正的 parent 端到底是 raw image、mean-set、DeepSets 还是 concat。

### 4.7 CNN 确定性层级与“确定性依赖退化”存在张力

相邻 CNN 层是确定性映射。若 Case 3 的随机变量实际是图块，而层特征只是 projector，必须明确区分；否则其建模逻辑和前文动机冲突。

---

## 5. 相关工作与 novelty 风险

### 5.1 方法本体最接近的经典/现代线

- Rényi/Hirschfeld–Gebelein maximal correlation；
- ACE；
- principal inertia components；
- DCCA；
- function-space feature learning；
- neural operator SVD；
- VAMP/VAMPnet；
- spectral contrastive learning；
- Barlow Twins、VICReg 等 redundancy-reduction SSL。

### 5.2 作者既有工作的重合风险

讨论中识别到：

1. 旧 FMCA 已经包含 density-ratio decomposition、trace/log-det、双网络、whitening/SVD、谱特征等基础内容；
2. HFMCA 已经覆盖多视图图像增强、9-view 条件均值、CIFAR SSL、层级图像分析和 dependence maps，且部分核心数字与当前稿件高度重合；
3. 加高斯噪声处理确定性依赖退化，也已经出现在作者后续 auxiliary-variable 分析线中。

因此，当前稿件最合理的定位是：

> 对 FMCA、HFMCA 和 auxiliary-noise dependence analysis 的统一、严格、journal-style 扩展。

必须清楚列出旧工作与当前扩展的差异，不能把旧结果重新包装成全新算法。

### 5.3 最值得保留的新主线

最清楚、最可能成立的核心增量是：

1. 将单模态特征学习统一为人为设计 \(p(Y\mid X)\) 的算子谱学习；
2. 显式利用同一 \(X\) 的多个条件样本估计 \(\mathbb E[g(Y)\mid X]\)；
3. 给出精确或可验证的 operator-recovery 实验；
4. 证明谱值排序的是辅助通道所保留的因素；
5. 在公平计算预算下评估其 SSL 效用。

---

## 6. 实验必须回答的六个科学问题

| 科学问题 | 当前证据 | 决定性实验 |
|---|---|---|
| 是否恢复正确谱？ | Nyström 图像相似 | 解析真值、谱误差、子空间误差、样本复杂度 |
| 条件均值是否有独立贡献？ | 文字声称降方差 | \(M\) 扫描、固定总 views、梯度与矩阵估计方差 |
| SSL 是否在公平预算下更优？ | 9-view 对多数 2-view | 同 views、FLOPs、head、评估协议和 seeds |
| 谱值是否有效排序因素？ | top-\(k\) 对原始前 \(k\) | top/random/bottom/PCA/random rotation；已知因素数据 |
| TSD 到底测什么？ | 增强强度表 | held-out calibration、data-processing、utility–TSD 关系 |
| Markov map 是否可信？ | 定性热图 | direct-vs-recursive、定位、faithfulness、sanity checks |

---

## 7. 所有实验前必须冻结的统一协议

### 7.1 数据分工

至少分为：

- `train`：优化网络；
- `calibration`：估计 \(R_F,R_G,P_{FG}\)，做 whitening/SVD；
- `test`：报告 held-out eigenvalues、TSD、谱误差和下游性能。

禁止在同一训练数据上最大化谱值后，再把该训练谱值当作无偏结果。

### 7.2 数值正则化

统一：

\[
R_F^\epsilon=R_F+\epsilon I,
\qquad
R_G^\epsilon=R_G+\epsilon I.
\]

主设置可固定 \(\epsilon=10^{-3}\)，附录扫描：

\[
10^{-2},10^{-3},10^{-4},10^{-5}.
\]

报告 condition number、numerical rank、最小特征值和分解失败率。

### 7.3 重复谱值的评价

重复或近重复谱值时，不比较单个函数，而比较：

- principal angles；
- projection-matrix error；
- spectral block 的子空间重合。

### 7.4 公平性维度

SSL 至少需要三种横轴：

1. epoch；
2. total encoded views；
3. GPU hours 或估计 FLOPs。

### 7.5 统计规范

- toy/解析实验：10–20 seeds；
- CIFAR：至少 5 seeds；
- ImageNet100/1K：至少 3 seeds；
- 报 mean、std、95% CI；
- 主结论优先依据 paired seed differences；
- 固定 final epoch 或预注册 validation selection，不挑 test 最优点。

---

## 8. 实验工作包

# WP-A：精确真值下的谱恢复

## A1. 一维联合高斯：解析 ground truth

设置：

\[
X\sim\mathcal N(0,1),
\qquad
Y=X+\sqrt\sigma\,\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,1).
\]

标准化相关系数：

\[
r=\frac{1}{\sqrt{1+\sigma}}.
\]

非平凡谱可由 Hermite 模式解析获得：

\[
\lambda_{n+1}=r^{2n}.
\]

扫描：

- \(\sigma\in\{0.01,0.05,0.1,0.25,0.5,1,2,5,10\}\)；
- \(N\in\{500,1000,5000,20000,100000\}\)；
- \(M\in\{1,2,4,8,16,32\}\)；
- \(K\in\{4,8,16,32\}\)。

指标：

- eigenvalue relative error；
- Hermite function correlation；
- density-ratio reconstruction error；
- held-out TSD error；
- 样本复杂度与 \(M\) 复杂度。

## A2. 二维高斯：重复谱值与子空间恢复

两个版本：

1. 各向同性：一阶模式重复，检验单个函数会旋转但子空间稳定；
2. 各向异性：打破退化，检验排序恢复。

主指标：projection error、principal angles、跨 seed 子空间稳定性。

## A3. 有限离散通道

构造 8–20 状态的已知 \(p(X)\) 和转移矩阵 \(P(Y\mid X)\)，直接精确 SVD。

至少包含：

- 接近 identity；
- block transition；
- 高噪声近独立；
- 非对称循环。

这组实验不含 Nyström/核估计误差，是最干净的真值验证。

## A4. 保留 two moons / GMM / spiral

改为辅助可视化，不再承担“无偏真值”主证据。补：

- eigenvalue relative error；
- aligned function correlation；
- subspace principal angles；
- 多 seed CI。

---

# WP-B：条件均值与多条件采样机制

## B1. 固定 parent 数量

固定 \(B_X\)，扫描 \(M\)。总计算随 \(M\) 增加。

测：

- \(P_{FG}\)、\(R_G\) 估计误差；
- trace/log-det bias 和 variance；
- gradient variance；
- 最终谱误差；
- 训练失败率。

回答：给定同样数量的原始 \(X\)，多采样 \(Y\) 是否降低 conditional Monte Carlo noise？

## B2. 固定总 view 数

固定：

\[
B_XM=C.
\]

例如：

| \(B_X\) | \(M\) |
|---:|---:|
| 1024 | 1 |
| 512 | 2 |
| 256 | 4 |
| 128 | 8 |
| 64 | 16 |

回答：固定算力时，应看更多 parent，还是对同一 parent 多采样？

## B3. 直接测损失和梯度 bias–variance

固定网络参数，用超大 Monte Carlo 或解析积分作为参考，重复 500–1000 个 minibatch。

报告：

\[
\operatorname{Bias}(\hat r),
\quad
\operatorname{Var}(\hat r),
\quad
\mathbb E\|\widehat{\nabla r}-\nabla r^\star\|^2.
\]

## B4. \(R_G\) 估计实现检查

比较：

1. 正确的 mean of outer products：
   \[
   \mathbb E[g(Y)g(Y)^\top\mid X];
   \]
2. 错误的 outer product of conditional mean：
   \[
   \mathbb E[g(Y)\mid X]\mathbb E[g(Y)\mid X]^\top.
   \]

证明当前实现没有丢失条件方差。

---

# WP-C：SSL 架构与概率模型对齐

比较四种 parent-side representation：

1. **Raw-parent：** \(f(X)\) 直接输入未增强原图；
2. **Mean-set：** 对多个 view backbone feature 求均值；
3. **DeepSets：** permutation-invariant set encoder；
4. **Concat：** 当前手稿方案。

必须匹配：

- backbone；
- output dimension；
- projector parameter count；
- view 数；
- total forward passes。

重点判断：concat 的收益是否只是来自更大参数量或 view 排列信息。主方法优先考虑 permutation-invariant 版本。

---

# WP-D：公平 SSL benchmark

## D1. 数据与模型

主建议：

| 数据集 | Backbone | 作用 |
|---|---|---|
| CIFAR10 | ResNet-18 | 快速 ablation |
| CIFAR100 | ResNet-18 | 细粒度语义 |
| ImageNet100 | ResNet-50 | 主扩展 |
| ImageNet-1K | ResNet-50 | 完整 journal，可选 |

## D2. 必须包含的近机制基线

- Regular FMCA / FMCA-AV \(M=1\)；
- DCCA 或 VAMP-2；
- Spectral Contrastive Learning；
- Barlow Twins 或 VICReg；
- FastSiam 或其他 multi-view baseline；
- SimCLR 作为标准对比学习锚点。

Toy/operator recovery 中可加入 neural operator-SVD 方法。

## D3. 三种公平协议

1. matched epochs；
2. matched total encoded views；
3. matched GPU hours/FLOPs。

若优势只在 matched epochs 下存在，只能声称“以更多条件视图换取更少 epoch”，不能声称计算效率更高。

## D4. matched views

至少统一跑：

\[
M=2,
\qquad M=8.
\]

使 FMCA-AV 与多视图基线在同 view 数下比较。

## D5. matched head 与 native head

两套结果：

- matched projector dimension，例如全部 \(K=128\) 或 256；
- 各方法原论文 native head。

主结论优先依据 matched architecture。

## D6. 评估协议

主指标：

1. frozen backbone + linear probe；
2. k-NN；
3. 三层 MLP 仅作副指标；
4. 可选 1%/10% label fine-tuning。

必须明确 probe 使用 backbone representation 还是 projector representation。

## D7. 稳定性指标

- seed mean/std；
- failure/collapse rate；
- covariance condition number；
- effective rank；
- per-dimension minimum variance；
- gradient norm/variance；
- nonzero eigenvalue count；
- batch-size sensitivity。

扫描：

\[
B\in\{64,128,256,512\},
\qquad
K\in\{32,64,128,256\}.
\]

当前“约一半 eigenvalues 非零”必须判断是否只是有限 batch rank 约束。

---

# WP-E：谱排序与因素语义

## E1. 六种 top-\(k\) 对照

对同一个完整表示比较：

- Eigen-top-\(k\)；
- Eigen-bottom-\(k\)；
- Random-\(k\)；
- PCA-top-\(k\)；
- Unranked-first-\(k\)；
- Random-rotation-first-\(k\)。

扫描：

\[
k\in\{1,2,4,8,16,32,64,128\}.
\]

报告：

- linear-probe curve；
- k-NN curve；
- accuracy–dimension AUC；
- 达到完整表示 95% 性能的最小维数。

## E2. 已知生成因素数据

建议使用 dSprites、Shapes3D、SmallNORB 或自建彩色形状数据。

分别设计辅助通道：

| 通道 | 预期被破坏因素 | 预期保留因素 |
|---|---|---|
| color jitter | color | shape |
| crop | position、部分 scale | object identity |
| rotation | orientation | identity/shape |
| blur | texture、高频 | 低频轮廓 |
| grayscale | color | structure |

对 shape、color、position、scale、orientation 分别线性 probe。

目标是把主张改成：

> 谱值按辅助通道中的因素可预测性排序。

## E3. 跨种子稳定性

报告：

- eigenvalue rank correlation；
- top-\(k\) eigenspace overlap；
- spectral-block stability；
- bootstrap CI。

---

# WP-F：TSD 的重新定义与校准

## F1. 解析 calibration

在高斯问题上比较：

- exact TSD；
- FMCA-AV held-out TSD；
- Nyström TSD；
- 其他 scalar dependence estimator。

报告 calibration slope、\(R^2\)、绝对误差。

## F2. 所有图像 TSD 必须 held-out

流程：

1. train 学 encoder；
2. calibration 做 whitening/SVD；
3. test 重新估计跨视图相关与 eigenvalues；
4. 报 train–test TSD gap。

禁止使用第 5 epoch training objective 直接充当最终依赖估计。

## F3. 增强强度单调性

对 crop、color jitter、blur、rotation、grayscale 各扫至少 7 个强度、5 seeds。

报告：

- held-out TSD mean/CI；
- Spearman correlation；
- monotonicity violations；
- test–retest reliability。

## F4. Data-processing 链

构造：

\[
X\to Y_1\to Y_2\to Y_3,
\]

例如逐级加噪，或 crop → crop+color → crop+color+blur。

同时画：

- TSD 随破坏程度变化；
- downstream utility vs TSD。

预期结论可能是倒 U 型，而非“越大越好”。

---

# WP-G：可控 Markov 谱验证

## G1. 可逆离散 Markov chain

构造满足 detailed balance 的链，精确计算一步和多步转移。

比较：

- direct lag-\(\tau\) estimate；
- recursive/composed estimate；
- 理论幂律；
- Chapman–Kolmogorov residual。

扫描：

\[
\tau\in\{1,2,4,8,16\}.
\]

## G2. 非可逆链

构造有方向循环链，测试简单谱幂公式是否失败。

目的不是隐藏失败，而是界定定理适用范围：

- 可逆/自伴时成立；
- 一般非可逆时改用 VAMP 左右奇异函数或直接 lag decomposition。

## G3. 连续动力系统，可选

OU process 或双势阱 Langevin，适合完整 journal 版本。

---

# WP-H：Dependence map 的定量验证

## H1. Direct vs recursive

对早期层和最终层：

- 直接估计 \(\rho(Z_s,Z_S)\)；
- 通过相邻层递归组合。

比较：

- map rank correlation；
- normalized L2 error；
- top-region IoU；
- 时间和内存。

## H2. Plain CNN vs ResNet

先在无 skip 的 VGG/plain CNN 上验证链式组合，再处理 ResNet。

对于 ResNet，必须明确：

- 忽略 skip；
- 将网络视为 DAG；
- 或分别传播 identity 与 residual path。

## H3. 有标注定位数据

建议：PASCAL VOC、CUB bounding boxes、ImageNet localization 或 segmentation subset。

指标：

- Pointing Game；
- MaxBoxAcc；
- pixel IoU/AUPRC；
- foreground energy ratio。

必须对比：

- random map；
- center Gaussian；
- edge/gradient map；
- activation norm；
- class-agnostic PCA/Eigen-CAM 类方法。

## H4. Faithfulness

删除 dependence map 的 top-\(p\%\) 区域，测最终表示变化：

\[
D_{\mathrm{rep}}(p)
=1-\cos(z(x),z(x_{\mathrm{del}})).
\]

对比：top、bottom、random、center deletion。

还可报告 classification logit drop、insertion/deletion AUC。

## H5. Sanity checks

至少：

1. layer-wise parameter randomization；
2. fully random weights；
3. random-label model。

若随机模型仍产生相同对象轮廓，说明 map 主要反映输入边缘、网络架构或中心偏置，而非已学表示。

## H6. 解释纪律

只有 localization、faithfulness、randomization 均通过，才能称 explainability。否则统一称：

> class-agnostic local-to-global dependence visualization。

---

## 9. 总体 ablation 清单

1. trace vs log-det；
2. no whitening / single-side / dual-side / dual-side + post-hoc SVD；
3. centered covariance vs uncentered second moment；
4. 显式 constant mode vs zero-mean space；
5. \(K\) 与 batch size；
6. matrix regularization \(\epsilon\)；
7. network capacity；
8. \(M\) 与 parent/view budget；
9. raw-parent / mean-set / DeepSets / concat；
10. train/cal/test 谱估计 gap。

---

## 10. 最小可发表版本

若资源有限，优先完成：

1. **精确 operator recovery**：一维高斯、二维重复谱、有限离散通道；
2. **条件采样机制**：固定 parent 和固定总 views 的 \(M\) ablation；
3. **公平 SSL**：matched views、head、FLOPs、linear probe、多 seeds；
4. **谱语义**：top/random/bottom/PCA + 已知生成因素；
5. **held-out TSD**：改成 retained-dependence diagnostic；
6. **Case 3 降级或补全**：没有定量定位与 sanity checks 就移到附录。

主叙事应冻结为：

\[
\boxed{
\text{Exact recovery}
\rightarrow
\text{conditional-sampling mechanism}
\rightarrow
\text{fair SSL utility}
\rightarrow
\text{spectral semantics}
}
\]

不要继续通过增加更多相似 CIFAR 表格来扩充论文。

---

## 11. 优先级

### A 级：投稿前不可缺

1. 去常数模式并修复 TSD/log-det；
2. 解析/精确真值下的谱恢复；
3. \(M\) 和条件均值的固定预算 ablation；
4. 公平 SSL 比较；
5. linear probe + 多 seeds；
6. top/random/bottom/PCA 排序；
7. held-out TSD。

### B 级：保留三-case 结构时不可缺

1. 可控 Markov direct-vs-composed；
2. dependence map 定量定位；
3. representation deletion/insertion；
4. randomization sanity checks；
5. plain CNN 与 ResNet 对照。

### C 级：完整 journal 扩展

1. ImageNet-1K；
2. low-label transfer；
3. 多因素数据集；
4. 非可逆连续 Markov process；
5. 更大 backbone。

---

## 12. Claim-to-gate 规则

| 论文主张 | GO 条件 | 不满足时的降级表述 |
|---|---|---|
| 条件均值降低估计噪声 | 固定 parent 下矩阵和梯度方差随 \(M\) 降低 | 仅称为多视图 Monte Carlo 近似 |
| 计算效率更高 | matched views/FLOPs/GPU time 下仍更快 | 仅称按 epoch 收敛快 |
| 谱值排序有用特征 | top-\(k\) 显著优于 random/PCA/bottom | 仅称排序辅助通道稳定性 |
| TSD 衡量增强强度 | held-out TSD 稳定单调 | 仅作训练目标，不作诊断 |
| TSD 预测表示质量 | 跨设置有稳定预测关系 | 明确 TSD 与 utility 不等价 |
| Markov composition 有效 | recursive 与 direct 高度一致 | 删除全局组合应用 |
| dependence map 解释模型 | 定位、faithfulness、sanity 全通过 | 改称 visualization |

---

## 13. 推荐的最终实验章节结构

### VII-A Exact Recovery of the Auxiliary-Variable Spectrum

- 解析高斯；
- 离散通道；
- eigenvalue/subspace/reconstruction error；
- two moons 等作为可视化。

### VII-B Why Conditional Sampling Works

- fixed-parent \(M\) sweep；
- fixed-view-budget \(M\) sweep；
- loss/gradient variance；
- parent-side aggregation ablation。

### VII-C Self-Supervised Representation Learning

- matched-view；
- matched-FLOP；
- matched-head；
- linear probe；
- stability/effective rank。

### VII-D Spectral Ranking and Auxiliary-Channel Semantics

- top/random/bottom/PCA；
- factor preservation；
- spectrum vs factor alignment。

### VII-E TSD as Retained Dependence

- analytic calibration；
- held-out augmentation severity；
- data-processing chain；
- TSD–utility trade-off。

### VII-F Markov Composition and Dependence Localization

仅在 direct-vs-recursive、定位、faithfulness 和 sanity check 全部完成后保留在主文；否则移附录。

---

## 14. 推荐执行顺序

### Phase 0：定义与代码审计

交付物：

- 去常数模式；
- train/cal/test 拆分；
- matrix regularization；
- trace/log-det 单元测试；
- \(R_G\) 正确估计检查；
- 固定配置、seed、日志格式。

### Phase 1：核心理论验证

交付物：

- 1D Gaussian exact；
- 2D repeated spectrum；
- finite discrete channel；
- fixed-parent / fixed-view-budget \(M\) ablation；
- bias–variance 报告。

只有 Phase 1 通过，才有资格保留“谱恢复”和“条件均值机制”主张。

### Phase 2：公平 SSL

交付物：

- CIFAR10/100 matched views/head；
- ImageNet100 matched views/FLOPs；
- linear probe、多 seeds、effective rank；
- raw/mean/DeepSets/concat。

### Phase 3：谱语义与 TSD

交付物：

- top/random/bottom/PCA；
- factor dataset；
- held-out TSD；
- augmentation severity 和 data-processing 曲线。

### Phase 4：Markov 与 maps

仅在前三阶段成功后执行。优先可控 Markov，再做图像解释。

---

## 15. 后续执行者报告模板

每一轮实验报告必须包含：

### 15.1 代码与复现状态

- branch；
- commit SHA；
- worktree clean/dirty；
- config 路径；
- 数据版本；
- 环境与 GPU；
- seeds。

### 15.2 实验问题

- 本轮只检验哪一个主张；
- primary estimand；
- 正向/负向判据；
- 禁止事后更换的超参数。

### 15.3 结果

- 主表；
- mean/std/CI；
- 计算预算；
- 失败任务及原因；
- 数值稳定性指标。

### 15.4 结论纪律

必须给出：

- `PASS` / `FAIL` / `INCONCLUSIVE`；
- 该结果支持的最窄结论；
- 不支持的强结论；
- 对论文措辞的影响；
- 下一轮实验计划。

禁止用“图看起来不错”代替定量判断。

---

## 16. 当前 PM 式结论

1. **现在不应扩展更多数据集或继续堆 SSL 基线。**
2. **第一优先级是精确谱恢复和条件采样机制。**
3. **FMCA-AV 的核心 novelty 必须与旧 FMCA/HFMCA 隔离。**
4. **TSD 应重新定位为 retained dependence，而不是表示质量。**
5. **谱值的最好表述是通道稳定性/可预测性排序。**
6. **Case 3 是当前最弱部分；未通过定量验证前，应降级。**
7. **主文最终应围绕一条单线展开，而不是三个松散 demo：**

\[
\text{辅助通道定义信息稳定性}
\rightarrow
\text{FMCA-AV 恢复其谱}
\rightarrow
\text{多条件采样改善估计}
\rightarrow
\text{谱可用于表征与因素排序}.
\]

---

## 17. 尚未解决、下一会话首先需要确认的信息

1. 目标投稿是 conference、IEEE journal 还是一般 ML journal；
2. 当前代码仓库、branch 和实现状态；
3. 旧 FMCA/HFMCA 代码及已发表结果能否直接复用；
4. 可用 GPU 数量、型号和时间预算；
5. 是否坚持保留 Case 3；
6. 是否需要 ImageNet-1K，还是 ImageNet100 足够；
7. 当前 SSL 表中的准确率究竟是 k-NN、linear probe、MLP 还是其他协议；
8. trace/log-det 当前实际实现中是否已经去除常数模式和加 ridge regularization。

在这些信息明确前，最安全的下一步是先做 **Phase 0 代码审计和 Phase 1 精确真值实验设计**。
