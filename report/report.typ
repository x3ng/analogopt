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

#show heading.where(level: 1): it => {
  set text(font: ("Sarasa Gothic SC"), weight: "bold")
  it
}
#show heading.where(level: 2): it => {
  set text(font: ("Sarasa Gothic SC"), weight: "semibold")
  it
}

#set par(leading: 0.65em, first-line-indent: 2em)
#set heading(numbering: "1.")

// --- helpers ---
#let tabcell(body) = table.cell(fill: gray.lighten(95pt), body)[#body]

// ============================================================
= LLM 辅助模拟电路参数优化实验报告

#set par(first-line-indent: 0em)

#text(size: 11pt, style: "italic")[

*摘要*：本实验基于 AnalogGym 开源 benchmark 的 NMCF 放大器电路（SKY130 PDK），系统对比了四种参数优化方法——强化学习（DDPG+R-GCN）、贝叶斯优化（GP+EI）、自建 LLM 增强 BO（LLM+BO）以及 LLANA（LLM 完全替代 GP 的 BO 框架）。在 24 维连续参数空间上，BO 以 20 次评估、FoM = −1.39 取得最优效率；RL 以 200 次评估、FoM = −1.24 取得最优终极性能；两种 LLM 辅助方法均未能超越纯 BO。LLANA 更因 LLM 在高维空间中探索退化，14 次评估后崩溃。实验表明，在高维连续参数空间中，数学 surrogate（GP）仍不可替代，LLM 更适合作为辅助角色而非主力 optimizer。

]

#set par(first-line-indent: 2em)

= 引言

模拟电路参数优化是芯片设计中的关键步骤。传统方法依赖设计师经验手动调参，效率低且难以保证最优。近年来，强化学习（RL）和贝叶斯优化（BO）等方法被尝试应用于自动化电路参数优化。与此同时，大语言模型（LLM）的快速发展催生了 LLM 辅助优化的新方向，例如 LLAMBO、LLANA 等框架尝试用 LLM 的上下文学习能力替代传统数学 surrogate model。

本实验以 AnalogGym 开源 benchmark @analoggym 的 NMCF 放大器电路为测试平台（SKY130 PDK），系统对比四种优化方法：

- *RL*：AnalogGym 自带的 R-GCN + DDPG 方法
- *BO*：基于 BoTorch 的 Gaussian Process + Expected Improvement
- *LLM+BO*：自建 LLM 增强的 BO，LLM 分析优化历史并提议候选参数
- *LLANA*：LLM 完全替代 GP surrogate 和 acquisition function（arXiv 2406.05250）@llana

= 方法

== 问题定义

- *电路*：NMCF 放大器（SKY130 PDK）
- *参数空间*：24 维连续参数，归一化到 $[-1, 1]$。包括 7 个 MOSFET 的 W/L/M（21 个）、偏置电流 Ib、补偿电容乘数 $M_("C0")$、$M_("C1")$
- *优化目标*：最大化 Figure of Merit（FoM），即 $Sigma$ 11 个性能指标得分之和，每个指标得分 $<= 0$，最高为 0（全部达标）
- *仿真器*：Ngspice v45，每次仿真运行 AC/DC 分析（`TB_Amplifier_ACDC.cir`）和瞬态分析（`TB_Amplifier_Tran.cir`）

11 个评估指标：温度系数（TC）、功耗（Power）、失调电压（$V_"os"$）、共模抑制比（CMRR）、直流增益（DC Gain）、增益带宽积（GBW）、相位裕度（PM）、正/负电源抑制比（PSRP/PSRN）、压摆率（Slew Rate）、建立时间（Settling Time）。

== RL 方法（RGNN + DDPG）

模型采用 R-GCN（关系图卷积网络）编码电路拓扑为图结构（28 节点），每节点包含 12 个晶体管工作点特征（$g_m$、$g_("ds")$、$V_("th")$、$V_("dsat")$ 等），由 `dev_params.py` 生成的 Spice 命令从 DCOP 分析提取。Actor 输出 24 维连续动作（参数变化方向），Critic 评估状态-动作价值。策略为 DDPG（深度确定性策略梯度），前 1000 步纯随机探索，使用 uniform noise（$sigma = 2$），batch size 128，memory size 10000。实验运行 200 步。

== Bayesian Optimization

使用 BoTorch @botorch 框架：Surrogate model 为 `SingleTaskGP`，Acquisition function 为 Expected Improvement（EI），初始采样使用 Sobol 序列 10 个点以均匀覆盖参数空间。每轮用 L-BFGS-B 优化 EI 选择下一个评估点。共 10 轮 BO 迭代。

== LLM+BO 方法（自建）

基于自建 `LLMBOSolver`。每轮生成 3 个候选点：1 个来自 BO 的 EI 优化结果，2 个来自 LLM 基于优化历史（参数 $->$ FoM）的提议。三个候选点中选 FoM 最优者进入下一轮。LLM 通过 Anthropic SDK 兼容接口调 DeepSeek（`deepseek-v4-pro`）。

== LLANA 方法

LLANA @llana 是 LLM-Enhanced BO 框架，完全用 LLM 替代 GP surrogate 和 acquisition function。采用 discriminative mode：

- *Acquisition*：LLM 基于历史观察的 few-shot 示例生成候选配置
- *Surrogate*：LLM 对候选配置预测性能分数，计算 EI 选择最优候选

不依赖任何数学优化器，纯粹靠 LLM 的 in-context learning。实验中设置 `n_templates=1`、`n_gens=3`、`n_candidates=5`（保守参数以避免 API 限流），`n_initial=5`、`n_trials=15`。LLM 使用 DeepSeek-chat（OpenAI 兼容接口；`deepseek-v4-pro` 为推理模型，长 prompt 返回空 content 无法使用）。

= 实验设置

#figure(
  table(
    columns: 5,
    [*参数*], [*BO*], [*LLM+BO*], [*RL*], [*LLANA*],
    [初始采样], [Sobol $10$], [Sobol $10$], [Random $100$], [Random $5$],
    [优化步数], [10 BO iter], [10 iter（每轮3点）], [200 steps], [9 trials（Trial 9 崩溃）],
    [总评估数], [20], [40], [200], [14],
    [总耗时], [83.8s], [1345.7s], [1903.9s], [1063.5s],
  ),
  caption: [实验参数配置],
)

= 结果

== 四方法对比

#figure(
  table(
    columns: 5,
    [*方法*], [*最佳 FoM*], [*评估次数*], [*耗时*], [*样本效率*],
    `BO (GP+EI)`,     [−1.3923], [20],  [83.8s],   [*最优*],
    `RL (DDPG+RGCN)`, [*−1.2446*], [200], [1903.9s], [低],
    `LLM+BO (自建)`,  [−1.5063], [40],  [1345.7s], [差],
    `LLANA`,          [−1.7673], [14],  [1063.5s], [最差（早停）],
  ),
  caption: [四种方法优化结果对比],
)

== 收敛分析

#figure(
  image("figures/comparison.png", width: 100%),
  caption: [收敛曲线对比。左：每次评估的 FoM；中：累积最优 FoM；右：最终结果柱状对比（FoM / 时间 / 评估次数）。],
)

- *BO*：收敛最快，第 2 个优化步即找到最佳点（−1.39），之后在相近水平波动。
- *LLM+BO*：波动大（−1.5 ∼ −4.1），LLM 提议的点质量不稳定，未超越 BO。
- *RL*：200 步中 FoM 在 −1.24 ∼ −4.0 间波动，最佳在 step 101（刚结束随机探索阶段），后续未见明显改善。
- *LLANA*：初始随机采样（5 点）最优仅 −1.67，后续 LLM 提议严重趋向退化配置，无法有效探索，Trial 9 因候选点重复触发早停。

== LLANA 的探索失败分析

LLANA 在第 9 轮崩溃（`LLM failed to generate candidate points`），根本原因是_LLM 在 24 维空间中探索能力严重不足_。

#figure(
  table(
    columns: 3,
    [*LLM 提议类型*], [*出现次数*], [*示例*],
    [全零向量], [多次], [`[0, 0, 0, ..., 0]`],
    [交替 $plus.minus 1$], [3 次], [`[0, −1, 1, −1, 0, ...]`],
    [交替 $plus.minus 0.5$], [1 次], [`[−0.5, 0.5, −0.5, 0.5, ...]`],
    [部分零 + 交替值], [多次], [`[0, 0, −1, 0, −0, 1, ...]`],
  ),
  caption: [LLM 输出的退化候选配置模式],
)

LLM 只输出整数或半整数的退化配置，未产生具有连续多样性的候选点。每轮 acquisition 需要 5 个有效候选，但实际只能产出约 3 个，5 次重试后仍无法凑齐，触发异常退出。这与 LLANA 原论文在低维超参数优化任务上的成功形成鲜明对比——NMCF 的 24 维连续参数空间超出了当前 LLM（DeepSeek-chat）的数值直觉范围。

== 典型失败模式

四种方法均在以下指标上持续丢分：

- *Power（功耗）*：NMCF 三级放大器在满足增益/带宽要求时，偏置电流偏大导致功耗超标。
- *Phase Margin（相位裕度）*：三级运放天然存在相位问题，补偿电容 $M_("C0")$、$M_("C1")$ 调节不够精细。
- *Settling Time（建立时间）*：与 GBW 和 PM 紧密耦合，难以独立优化。

= 讨论

== BO 为何效率最高

GP 模型在 24 维空间能有效建模 FoM landscape，EI 在探索与利用间取得良好平衡。每次评估的计算开销仅为 ngspice 仿真（约 4s），无额外 API 延迟。30 次评估（含初始 10 点）的总耗时不到 90s。

== LLM+BO 为何未达预期

- *Prompt 设计简单*：仅提供参数值和 FoM，缺乏电路领域知识注入。
- *API 延迟高*：每轮 2 次 LLM 调用增加约 60s，占实验时间的 90% 以上。
- *LLM 缺乏数值直觉*：在 24 维空间中提议候选点超出了 LLM 的强项范围。
- *候选策略低效*：每轮评估 3 个点（1 BO + 2 LLM），LLM 提议的点质量常不如 EI 优化结果。

== LLANA 为何更差

- *LLM surrogate 失效*：LLANA 用 LLM 替代 GP 做 surrogate model，要求 LLM 基于 5-10 个历史样本预测未见点的 FoM。在 24 维空间中，LLM 无法做出有意义预测，几乎退化为对历史值的机械复述。
- *探索退化*：LLM 在 acquisition 阶段反复输出离散退化配置（全零、交替 $plus.minus 1$），丧失了连续空间探索能力。DeepSeek-chat 对「建议一个 24 维 $[-1, 1]$ 内的配置」这类请求倾向于输出结构化而非数值化的回答。
- *样本效率极低*：14 次评估后即因候选点重复而崩溃，最优 FoM（−1.77）仅来自近似全零配置，不如任意一次随机初始采样。
- *与 LLM+BO 的本质区别*：自建 LLM+BO 保留 GP 做主 optimizer，LLM 仅作辅助提议；LLANA 完全放弃数学 surrogates，将整个优化过程交给 LLM 的上下文学习——在 24 维电路优化问题上，这个思路被证明不适用。

== RL 的潜力与局限

RL 达到最佳 FoM（−1.24），表明 DDPG+R-GCN 有潜力。但 200 步中最佳出现在 step 101（接近随机阶段结束），后续未见改善。可能原因：DDPG 超参数未针对此问题调优，或 GNN 对电路拓扑的编码不够有效。相比 BO，RL 的主要问题是样本效率低——需 10 倍评估次数才取得相近结果。

== 改进方向

- *LLM 辅助优化的适用边界*：LLANA 实验表明，纯 LLM 驱动的优化在 20+ 维连续参数空间中不适合。LLM 更适合低维（$<10$）或离散/分类参数空间，或作为数学 optimizer 的补充而非替代。
- *LLM+BO*：注入电路领域知识（器件物理约束、设计规则）到 prompt；LLM 仅在 BO 停滞时触发；尝试用更强大的模型或 domain-specific embedding 编码参数空间。
- *RL*：增加训练步数（1000+）观察学习效果；尝试 PPO 替代 DDPG。
- *实验扩展*：在更多 AnalogGym 电路上验证；引入多目标优化（Pareto front）。

= 结论

在 NMCF 电路参数优化的对比实验中：

1. *BO 是当前最实用的方法*：20 次评估内达 FoM = −1.39，效率显著高于所有其他方法。GP 是高维连续空间中最可靠的 surrogate model。

2. *RL 展示最佳终极 FoM（−1.24）*但需 200 次评估，样本效率低，适合仿真预算充足的场景。

3. *LLM+BO 未带来增益*：自建混合方法中，LLM 提议的候选点质量不稳定，API 延迟增加了 90% 以上的时间开销，FoM 反而退化（−1.51）。

4. *LLANA 不适合此类问题*：LLM 在 24 维空间中探索退化为离散模式，14 次评估后崩溃，最优仅 −1.77。纯 LLM-based surrogate 无法替代 GP 在高维连续空间中的建模能力。

5. 所有方法均受限于 NMCF 电路的物理瓶颈（功耗-速度-稳定性 trade-off）。

6. *核心启示*：LLM 辅助电路优化的有效路径不是替代数学 optimizer，而是在低维子问题、离散选择或领域知识注入上发挥作用。

#pagebreak()

#set par(first-line-indent: 0em)
#set text(size: 10pt)

= 参考文献

#bibliography(
  "refs.bib",
  title: none,
)

