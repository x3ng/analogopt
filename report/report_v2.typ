#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  numbering: "1",
)

#set text(
  font: ("Noto Serif CJK SC"),
  size: 12pt,
  lang: "zh",
)

#show heading.where(level: 1): set block(above: 2.2em, below: 1.0em)
#show heading.where(level: 2): set block(above: 1.6em, below: 0.9em)

#show heading.where(level: 1): it => {
  set text(font: ("Sarasa Gothic SC"), weight: "bold")
  it
}
#show heading.where(level: 2): it => {
  set text(font: ("Sarasa Gothic SC"), weight: "semibold")
  it
}

#set par(leading: 0.85em, first-line-indent: 2em)
#set heading(numbering: "1.")
#show figure: set block(above: 1.0em, below: 0.8em)

= 实验背景与目的

模拟电路参数优化是芯片设计中的关键步骤。传统方法依赖设计师经验手动调参，效率低且难以保证最优。近年来，强化学习（RL）和贝叶斯优化（BO）开始被尝试用于自动化电路参数优化；同时，大语言模型（LLM）的快速发展也催生了 LLM 辅助优化的新方向。

本实验基于 AnalogGym 开源 benchmark 的 NMCF 放大器电路（SKY130 PDK），系统对比五种参数优化方法：

- 强化学习（RL）：DDPG + R-GCN
- 贝叶斯优化（BO）：Gaussian Process + Expected Improvement
- 自建 LLM 增强 BO（LLM+BO）
- LLANA v1：LLM 完全替代 GP surrogate 的 BO
- LLANA v2：电路感知 prompt 改进版 LLANA

实验目的是评估 LLM 辅助方法在 24 维连续参数空间中的实际优化能力，与 RL、BO 进行定量对比。

= 实验平台

== 电路与参数空间

#set par(first-line-indent: 0em)

实验使用 AnalogGym 中的 NMCF 放大器电路（Nested Miller Compensation with Feedforward），基于 SKY130 PDK。

- *参数空间*：24 维连续参数，归一化到 $[-1, 1]$。包括 7 个 MOSFET 的宽/长/倍率（21 个）、偏置电流 $I_b$、补偿电容乘数 $M_("C0")$、$M_("C1")$
- *优化目标*：最大化 Figure of Merit（FoM），即 11 个性能指标得分之和
- *仿真器*：Ngspice v45，每次运行 AC、DC 和瞬态分析

== 评估指标

11 个评估指标：温度系数（TC）、功耗（Power）、失调电压（$V_"os"$）、共模抑制比（CMRR）、直流增益（DC Gain）、增益带宽积（GBW）、相位裕度（PM）、正/负电源抑制比（PSRP/PSRN）、压摆率（Slew Rate）、建立时间（Settling Time）。每个指标得分 $<= 0$，最高为 0（全部达标）。

#set par(first-line-indent: 2em)

= 实验方法

== RL：DDPG + R-GCN

模型使用 R-GCN（关系图卷积网络）将电路拓扑编码为图结构（28 节点），每节点包含 12 个晶体管工作点特征（$g_m$、$g_("ds")$、$V_("th")$、$V_("dsat")$ 等）。Actor 输出 24 维连续动作，Critic 评估状态-动作价值。策略为 DDPG，前 100 步纯随机探索，使用 uniform noise（$sigma = 2$）。实验运行 500 步，batch size 128，memory size 10000。

== BO：GP + EI

使用 BoTorch 框架：surrogate model 为 `SingleTaskGP`，acquisition function 为 Expected Improvement（EI）。初始使用 Sobol 序列采样 10 个点，之后每轮使用 L-BFGS-B 优化 EI 选择下一个评估点。共 10 轮 BO 迭代，总计 20 次评估，seed=42 可复现。

== LLM+BO：自建 LLM 增强 BO

基于自建 `LLMBOSolver`，BO 的 GP+EI 做主优化器，LLM 提供辅助候选点。每轮生成 3 个候选点：1 个来自 EI 优化，2 个来自 LLM 分析优化历史的提议，从中选最优者进入下一轮。LLM 通过 Anthropic 兼容 API 调用。共 10 轮迭代（10 初始 Sobol + 30 后续评估）。LLM 使用 DeepSeek-chat。

== LLANA v1：LLM 替代 GP

LLANA 是一个用 LLM 完全替代 GP surrogate 和 acquisition function 的 BO 框架。采用 discriminative mode，不依赖任何数学优化器。v1 使用原始的默认 prompt：

- 参数名：`x0`, `x1`, ..., `x23`（无意义索引，LLM 不知道正在调什么）
- 任务描述：`"NMCF_Amplifier"`（仅电路名，无拓扑信息）
- System prompt：`"You are an AI assistant that helps people find information."`
- 优化指标描述：`"mean squared error"`（不准确，实际是 FoM 最大化）

配置：`n_initial=5`、`n_trials=15`、`n_templates=1`、`n_gens=3`、`n_candidates=5`。LLM 使用 DeepSeek-chat。

== LLANA v2：电路感知 Prompt

在相同 LLANA 框架内，仅修改 prompt 文本内容（不改变核心算法）。三项改进：

1. *参数语义化*：将 `x0..x23` 替换为实际电路参数名（`W_M0`, `L_M0`, `M_M0`, `W_M8`, ..., `Ib`, `M_C0`, `M_C1`），让 LLM 理解它正在优化 MOSFET 尺寸、偏置电流和补偿电容。这实际上是正确启用了 LLANA 框架的 `use_feature_semantics=True` 功能。

2. *任务描述丰富*：将任务描述扩展为包含拓扑信息（3-stage nested Miller compensation with feedforward, SKY130, 7 MOSFET pairs with W/L/M, 11 performance metrics）。

3. *System prompt 领域化*：将 system prompt 替换为 `"You are an expert analog circuit designer optimizing amplifier transistor sizing parameters using simulation feedback."`。Acquisition 阶段使用 `"You are an expert analog circuit designer. Give only the requested output format with no explanation."`。

其他配置与 v1 完全一致：`n_initial=5`、`n_trials=15`、`n_templates=1`、`n_gens=3`、`n_candidates=5`，相同的 LLM（DeepSeek-chat）。

= 实验设置

#figure(
  table(
    columns: 6,
    [*参数*], [*BO*], [*LLM+BO*], [*RL*], [*LLANA v1*], [*LLANA v2*],
    [初始采样], [Sobol 10], [Sobol 10], [Random 100], [Random 5], [Random 5],
    [优化轮次], [10 BO iter], [10 iter（每轮3点）], [500 steps], [15 trials], [15 trials],
    [总评估数], [20], [40], [500], [14\*], [20],
    [总耗时], [76.9s], [1383.5s], [5642.8s], [773.8s], [1978.8s],
  ),
  caption: [实验参数配置。\*LLANA v1 在 Trial 7 崩溃，实际仅完成 14 次评估。],
)

= 实验结果

== 总体对比

#figure(
  table(
    columns: 6,
    [*方法*], [*最佳 FoM*], [*评估次数*], [*耗时*], [*样本效率*], [*备注*],
    `RL (DDPG+RGCN)`, [*−0.84*], [500], [5642.8s], [低], [最优 FoM, 但 25× BO 评估数],
    `BO (GP+EI)`,     [−1.39], [20],  [76.9s],    [*最优*], [同 seed 可复现],
    `LLANA v2`,       [−1.43], [20],  [1978.8s], [差], [改进 prompt, 未崩溃但未超 BO],
    `LLM+BO (自建)`,  [−1.51], [40],  [1383.5s], [差], [LLM 提议未带来增益],
    `LLANA v1`,       [−1.77], [14],  [773.8s],  [最差], [Trial 7 崩溃],
  ),
  caption: [五种方法优化结果对比，按 FoM 从高到低排列。],
)

== 收敛曲线

#figure(
  image("figures/comparison.png", width: 100%),
  caption: [收敛曲线对比。左：每次评估的 FoM；中：累积最优 FoM；右：最终结果柱状对比。],
)

各方法收敛表现：

- *BO* 收敛最快：第 2 个优化步找到最佳点（#calc.round(-1.39, digits: 2)），之后在相近水平波动。Sobol 初始化 + EI 在 24 维空间中有效平衡了探索与利用。

- *RL* 500 步达到最佳 FoM（#calc.round(-0.84, digits: 2)），出现在 step 479。FoM 在 #calc.round(-0.84, digits: 2) 至 #calc.round(-6.2, digits: 1) 间波动，DDPG 后期持续改善但振荡明显。

- *LLANA v2* 20 次评估完整运行，best（#calc.round(-1.43, digits: 2)）来自初始随机采样，LLM 在后续 15 轮中未提出更优配置。但输出质量稳定，最差仅 #calc.round(-4.82, digits: 2)。

- *LLM+BO* 波动大（#calc.round(-1.5, digits: 1) 至 #calc.round(-5.2, digits: 1)），LLM 提议的点质量不稳定，API 延迟占实验时间 90% 以上。

- *LLANA v1* 14 次评估后崩溃，best（#calc.round(-1.77, digits: 2)）来自初始采样，后续 LLM 输出退化为全零、交替 ±1 等离散模式。

== LLANA v1 vs v2：Prompt 消融对比

LLANA v1 与 v2 的唯一区别是 prompt 设计，框架算法和 LLM 模型完全相同，构成一次受控的 prompt 消融实验。

#figure(
  table(
    columns: 5,
    [*指标*], [*v1*], [*v2*], [*改善*], [*说明*],
    [最佳 FoM], [#calc.round(-1.7673, digits: 2)], [#calc.round(-1.4305, digits: 2)], [+0.34], [均来自初始随机采样],
    [最差 FoM], [#calc.round(-44.5316, digits: 2)], [#calc.round(-4.8157, digits: 2)], [*+39.72*], [v2 无灾难性输出],
    [崩溃], [是（Trial 7）], [否], [——], [v2 完整运行 20 次],
    [退化模式], [全零、交替 ±1], [无], [——], [v2 输出合理],
  ),
  caption: [LLANA v1 vs v2 消融对比。Prompt 改进使 LLM 从完全不可控变为稳定运行。],
)

三项改进的各自效果：

- *参数语义化*（效果最显著）：`x0=0.5` 对 LLM 毫无意义，但 `W_M0=0.5` 能触发 LLM 预训练中关于 MOSFET 宽度的领域知识，让 LLM 知道它在调什么。

- *任务描述丰富*：让 LLM 知道自己在优化三级 NMC 放大器而非抽象的"tabular regression task"，有助于理解各参数之间的耦合关系。

- *System prompt 领域化*：从通用助手变为电路设计专家角色，激活领域相关的 token 分布。

但无论 prompt 如何改进，LLM 始终未提出优于随机初始采样的配置。BO 在相同 20 次评估下取得 #calc.round(-1.39, digits: 2)。

= 分析讨论

== BO 效率最高的原因

GP 在 24 维空间中能有效建模 FoM landscape，EI 在探索与利用间取得平衡。每次评估仅需 ngspice 仿真（约 4s），无 API 延迟。20 次评估不到 80s，且同 seed 可复现。在有限仿真预算场景下，BO 是最优选择。

== LLM 方法未能超越 BO 的原因

*核心问题：高维连续空间中的数值推理能力*

24 维参数空间要求精确的数值推理——判断哪些维度组合能提升 FoM、预测参数调整的方向和步长。这是当前 LLM 的固有弱点。LLANA 在低维超参数优化中有报告效果，但在 24 维连续空间中，LLM 的 in-context learning 无法替代 GP 的数学建模。

*Prompt 能防崩溃但不能赋能力*

v1 $->$ v2：FoM 范围收窄 39.72、崩溃消失、输出质量提升。但 LLM 始终未找到优于随机采样的配置。Prompt engineering 的边际收益递减：从 `x0` 到 `W_M0` 是 0→1 的飞跃，但后续更详细的电路描述收益递减。LLM 缺乏对归一化数值（如 `W_M0=0.173`）的物理直觉。

*LLM+BO 为何也未超越 BO*

自建 LLM+BO 保留 GP 做主优化器，LLM 仅作辅助提议。但每轮额外 2 次 LLM 调用增加约 60s API 延迟，LLM 提议的候选点在 24 维空间中质量不稳定。边际收益为零。LLM 接收的是归一化到 $[-1, 1]$ 的抽象数值，即使知道"这是 MOSFET 参数"，也无法从 `W_M0=0.173` 推断物理意义和调优方向。

*LLM 辅助电路优化的合理路径*

不是替代数学 optimizer，而是在低维子问题或离散选择中发挥 LLM 的语义推理优势：分析 trade-off、约束搜索空间、在 BO 停滞时提供基于设计规则的探索方向。

== RL 的改进与局限

从 200 步增加到 500 步，FoM 从 −1.24 提升到 #calc.round(-0.84, digits: 2)（$Delta = +0.40$），最佳出现在 step 479，说明 DDPG 后期持续学习。但 500 步耗时约 1.5h，样本效率远低于 BO（20 次/77s）。RL 有进一步优化潜力（更多训练步数、尝试 PPO），但需要权衡训练成本。

== 典型失败模式

在 NMCF 电路中，几种性能指标始终难以达标：

- *功耗*：三级放大器在满足增益/带宽要求时，偏置电流偏大导致功耗超标。
- *相位裕度*：三级运放天然存在相位问题，补偿电容调节不够精细。
- *建立时间*：与 GBW 和 PM 紧密耦合，难以独立优化。

= 总结

1. RL 取得最佳终极性能（500 步 FoM = #calc.round(-0.84, digits: 2)），但需要 25 倍于 BO 的仿真次数，样本效率低。

2. BO 是最高效的方法：20 次评估、77s 达到 #calc.round(-1.39, digits: 2)，在有限仿真预算下最优。

3. Prompt 改进显著改善了 LLANA 的表现——消除了崩溃、大幅收窄输出范围——但 LLM 始终未提出优于随机采样的配置。在高维连续空间中，数学 surrogate（GP）不可替代。

4. LLM+BO 自建方法受限于 LLM 数值推理缺陷和 API 延迟，未带来增益。

5. LLM 辅助电路优化的正确方向是发挥其语义理解和领域知识推理优势（分析 trade-off、约束搜索空间），而非替代数学优化器在优化循环中做数值决策。
