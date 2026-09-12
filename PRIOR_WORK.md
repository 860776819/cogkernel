# 先行者源码调查（2026-09-13，代理实查 GitHub 源码，路径已核实）

原则：各取一刀，不做复读机。以下每条都标注「偷什么 / 扔什么」。

## 1. explauto（Oudeyer 团队，IAC/RIAC 好奇心）— 学习进度的标准实现

repo: flowersteam/explauto（Python 2 时代，已停维护；实测目录是 `interest_model/` 而非 `interest/`）

源码机制（`explauto/interest_model/tree.py`、`competences.py`）：
- 能力度量：`competence = exp(-||goal-reached||)`（截断在 [-10, 0]）。
- 区域学习进度 `progress_idxs`：默认 `abs_deriv_smooth`——**滑动窗口内前半段与后半段能力均值之差的绝对值**。就这么简单，不需要 RL。
- 区域分裂：点数超上限（100）时，尝试 100 个随机分裂点，选使 `len(l)*len(g)*|progress(l)-progress(g)|` 最大的（`best_interest_diff`，Baranes & Oudeyer 2012）。
- 采样哪个区域：softmax(progress × 区域体积，温度 0.2)。源码注释里有个警告值得记住：窗口重叠配置不当会让一次低能力样本永远贡献高进度→永远盯着一个区域。
- **偷**：`abs_deriv_smooth` 公式（我们 v0 的 LP=慢EMA−快EMA 是它的连续化版本）；softmax×体积采样。
- **扔**：它要求显式目标空间+有界连续域；我们在潜在/误差空间在线聚类分区。代码太老，重写不引用。

## 2. pymdp（主动推理）— 认知价值=信息增益+实用价值

repo: infer-actively/pymdp，`pymdp/control.py`（现名 `compute_neg_efe_policy`；旧的 `calculate_G_pA` 已不在）：
- 状态信息增益：`H(预测观测熵) − Σ q(s)·H(A_s)`——预测观测的熵减去按后验加权的模型熵。
- 实用价值：预测观测分布与偏好向量 C 的点积。
- 参数信息增益（Dirichlet 超参上的 `wnorm`）可选加入。
- 每条策略 `neg_efe = info_gain + utility − param_info_gain`，softmax(γ·G, γ=16) 选择。
- **偷**：无奖励的 model-based 好奇心函数式（状态信息增益项本身就能当关切信号用，M1+ 接入）。
- **扔**：有限策略矩阵+逐步规划——正是我们说的"步进触发"架构，全盘不取。

## 3. DreamerV3 — 世界模型与想象的标准工程

repo: danijar/dreamerv3（`dreamerv3/agent.py`、`rssm.py`、`embodied/core/replay.py`）：
- RSSM：确定性 GRU + 分类隐变量；先验/后验对称 KL 训练，stop-gradient 分工（dyn 项训练先验、rep 项训练编码器），`free_nats=1.0` 截断。
- 想象：从后验状态用先验动力学滚动 15 步，actor-critic 全在模型内训练。
- 回放：chunk 存储、默认纯均匀采样；priority/recency 选择器实现了但默认关闭。
- **偷**：symmetrized-KL + free-nats 的先验/后验训练配方；「世界模型只在真实经验上训练」的纪律。
- **扔**：想象为奖励服务的整套 RL 机器（λ-return、retnorm、slow target）。我们的想象为预测/探究服务，无奖励流。

## 4. MicroPsi2 / Dörner PSI 理论 — 神念镶空壳，理论可自实现

repo: joschabach/micropsi2（2022 后停更，依赖已死的 Theano）。**实查警告**：源码里 grep 不到 Dörner 式的需求/驱力/调制参数（arousal/pleasure/resolution level）——它只是个 nodenet 图编辑器+运行时；动机模型只存在于更老的 motivation_machine demo (2016)。没有维护中的 PSI 实现。
- **偷**：Bach 发表的 PSI 设计文档当规格书（需求→驱力→调制参数），门控节点作为非 NN 连续基质。
- **扔**：代码本身。

## 5. Goyal & Bengio 全局工作台（ICLR 2023）— 竞争广播的最小实现

repo: anirudh9119/shared_workspace（已停更但完整）：
- 每模块一个标量"出价"logit（一个线性层）；`Sparse_attention(top_k)` 实现 hard top-k：阈值=第 k 大分数，`clamp(score − threshold, 0)` 后归一化——只有 top-k 模块的提案写入工作台，再广播给所有模块读。
- **偷**：「每模块标量出价 → 线性 top-k 掩码 → 单一广播」这个模式只有几行，每拍可跑，不需要任务奖励。M1 的关切竞争可以直接用这个实现。
- **扔**：RIM/transformer 外壳和监督训练；它的竞争是静态任务驱动的，不是自组织的学习进度驱动。

## 6. OpenWorm — 有虫无认知

openworm/OpenWorm + c302（活跃）：从连接组自底向上模拟 C. elegans 神经系统+3D 肌肉/身体物理。**不做学习、记忆、好奇心**——行为是硬连线+生物力学的涌现。引用时写清："虫，但无认知"。

## 对 v0 设计的直接回灌

1. 我们的 LP（慢EMA−快EMA）与 explauto 的 `abs_deriv_smooth` 同构——合法性确认，且我们的在线聚类分区避开了它对显式目标空间的依赖。
2. GWT top-k 竞争模式（出价→top-k→广播）确认可插进 M1 的关切层，替代 v0 的"仅观测"竞争。
3. Dreamer 的纪律确认了 v0 保守决定 5（想象不入训练）是对的工程直觉：连 Dreamer 都只让世界模型吃真实回放。
4. pymdp 的状态信息增益是 M1+「惊异关切」的现成公式，比逐点预测误差更有理论根基。
5. MicroPsi 的空壳警示：不要指望现成"认知架构"仓库，核心机制都得自己写+自己验。
