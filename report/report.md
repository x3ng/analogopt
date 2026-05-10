# LLM辅助模拟电路参数优化 — 实验报告

## 1. 引言

模拟电路参数优化是芯片设计中的关键步骤。传统方法依赖设计师经验手动调参，效率低且难以保证最优。近年来，强化学习（RL）和贝叶斯优化（BO）等方法被应用于自动化电路参数优化。

本实验以 AnalogGym 开源 benchmark 的 NMCF 放大器电路为测试平台（SKY130 PDK），系统对比四种优化方法：
- **RL**：AnalogGym 自带的 RGNN + DDPG 方法
- **BO**：基于 BoTorch 的 Gaussian Process + Expected Improvement
- **LLM+BO**：自建 LLM 增强的 BO，利用大语言模型分析优化历史并提议候选参数
- **LLANA**：LLM-Enhanced BO 框架（arXiv 2406.05250），用 LLM 完全替代 GP surrogate 和 acquisition function

## 2. 方法

### 2.1 问题定义
- **电路**：NMCF 放大器（SKY130 PDK）
- **参数空间**：24 维连续参数（7 个 MOSFET 的 W/L/M、偏置电流 Ib、2 个电容乘数 M_C0、M_C1），归一化到 [-1, 1]
- **优化目标**：最大化 Figure of Merit（FoM，即 Reward）。FoM = Σ 11 个性能指标分数，每个 ≤ 0，最高为 0（全部达标）
- **仿真器**：Ngspice v45

11 个指标：TC（温度系数）、Power（功耗）、Vos（失调电压）、CMRR（共模抑制比）、DC Gain（直流增益）、GBW（增益带宽积）、Phase Margin（相位裕度）、PSRP/PSRN（电源抑制比±）、Slew Rate（压摆率）、Settling Time（建立时间）

### 2.2 RL 方法
- 模型：R-GCN（关系图卷积网络）+ DDPG（深度确定性策略梯度）
- 图结构编码电路拓扑（28 节点，每节点 12 特征 = 7 个晶体管 OP 参数 + 5 个电路参数）
- Actor 输出 24 维连续动作，Critic 评估动作价值

### 2.3 Bayesian Optimization
- Surrogate Model：Gaussian Process（BoTorch SingleTaskGP）
- Acquisition Function：Expected Improvement（EI）
- 初始化：Sobol 序列采样 10 个点
- 每轮选 EI 最大的点评估

### 2.4 LLM+BO 方法
- 基于自建 LLMBOSolver
- 每轮生成候选：1 个 BO 提议点 + 2 个 LLM 提议点
- LLM 接收优化历史（参数→FoM），生成有希望的候选参数
- 三个候选点中选 FoM 最优的进入下一轮
- LLM API：DeepSeek（deepseek-v4-pro[1m]，Anthropic 兼容接口）

### 2.5 LLANA 方法
- 完全用 LLM 替代 GP surrogate 和 acquisition function
- **Acquisition**：LLM 基于历史观察的 few-shot 示例生成候选配置（discriminative mode）
- **Surrogate**：LLM 对候选配置预测性能分数，计算 Expected Improvement 选择最优候选
- 不依赖任何数学优化器，纯粹靠 LLM 的上下文学习能力
- n_templates=1, n_gens=3, n_candidates=5（保守参数，避免 API 限流）
- LLM API：DeepSeek-chat（OpenAI 兼容接口，deepseek-v4-pro 返回空 content 无法使用）

## 3. 实验设置

| 参数 | BO | LLM+BO | RL | LLANA |
|------|----|--------|-----|-------|
| 初始采样 | Sobol 10 | Sobol 10 | Random 100 | Random 5 |
| 优化步数 | 10 BO iter | 10 iter (每轮3点) | 200 steps | 9 trials (Trial 9 崩溃) |
| 总评估数 | 20 | 40 | 200 | 14 |
| 总时间 | 83.8s | 1345.7s | 1903.9s | 1063.5s |

## 4. 结果

### 4.1 四方法对比

| 方法 | 最佳 FoM | 时间 | 评估次数 | 样本效率 (FoM/eval) |
|------|----------|------|----------|---------------------|
| **RL (DDPG+RGCN)** | **-1.2446** | 1903.9s | 200 | 低（大量随机探索） |
| **BO (GP+EI)** | -1.3923 | 83.8s | 20 | **最优** |
| LLM+BO (自建) | -1.5063 | 1345.7s | 40 | 差（LLM 耗时大） |
| LLANA | -1.7673 | 1063.5s | 14 | 最差（搜索早停） |

### 4.2 收敛分析

![comparison](figures/comparison.png)

- **BO**：收敛最快，第 2 个优化步即找到最佳点（-1.39），之后在相近水平波动
- **LLM+BO**：波动大（-1.5 ~ -4.1），LLM 提议的点质量不稳定，未超越 BO
- **RL**：200 步中 FoM 在 -1.24 ~ -4.0 间波动，最佳在 step 101（刚结束随机探索），未观察到明显的学习曲线
- **LLANA**：初始随机采样（5 点）最优仅 -1.67，后续 LLM 提议严重趋向退化配置（全零、交替 ±0.5、交替 ±1），无法有效探索，Trial 9 因候选点重复触发早停

### 4.3 LLANA 的探索失败分析

LLANA 在第 9 轮崩溃（`LLM failed to generate candidate points`），根本原因是 **LLM 在 24 维空间中探索能力不足**：

| LLM 提议类型 | 频率 | 示例 |
|-------------|------|------|
| 全零向量 | 多次 | [0, 0, 0, ..., 0] |
| 交替 ±1 | 3 次 | [0, -1, 1, -1, 0, -1, 1, 1, ...] |
| 交替 ±0.5 | 1 次 | [-0.5, 0.5, -0.5, 0.5, ...] |
| 部分零+交替 | 多次 | [0, 0, -1, 0, -0, 1, ...] |

LLM 只输出了整数或半整数退化配置，未产生具有连续多样性的候选点。每轮 acquisition 需要 5 个候选但实际只能产出 3 个有效点，5 次重试后仍无法凑齐，触发异常退出。

这与 LLANA 原论文在低维超参数优化任务上的成功形成对比——NMCF 的 24 维连续参数空间超出了 LLM 的数值直觉范围。

### 4.4 典型失败模式

四种方法都在以下指标丢分：
- **Power（功耗）**：NMCF 电路在满足增益/带宽要求时功耗偏高
- **Phase Margin（相位裕度）**：补偿电容 M_C0/M_C1 调节不够精细
- **Settling Time（建立时间）**：与 GBW 和 PM 紧密耦合，难独立优化

## 5. 讨论

### 5.1 BO 为何效率最高
- GP 模型在 24 维空间能有效建模 FoM landscape
- EI 采集函数在探索与利用间取得了良好平衡
- 每次评估的计算开销仅为 ngspice 仿真（~4s）

### 5.2 LLM+BO 为何未达预期
- **Prompt 设计简单**：仅提供参数值和 FoM，缺乏电路领域知识注入
- **API 延迟高**：每轮 2 次 LLM 调用增加 ~60s，占实验时间的 90% 以上
- **LLM 缺乏数值直觉**：在 24 维空间中提议候选点超出了 LLM 的强项范围
- **候选策略低效**：每轮评估 3 个点（1 BO + 2 LLM），但 LLM 提议的点常不如 BO 的 EI 优化结果

### 5.3 LLANA 为何更差
- **LLM surrogate 失效**：LLANA 用 LLM 替代 GP 做 surrogate model，要求 LLM 基于 5-10 个历史样本预测未知点的 FoM。在 24 维空间中，LLM 根本无法做出有意义的预测，几乎退化为对历史值的简单复述
- **探索退化**：LLM 在 acquisition 阶段反复输出离散退化配置（全零、交替 ±1），丧失了连续空间探索能力。DeepSeek-chat 对 "建议一个 24 维 [-1,1] 内的配置" 这种请求倾向于给出结构化而非数值化的回答
- **样本效率极低**：14 次评估后即因候选点重复而崩溃，最优 FoM（-1.77）仅来自近似全零配置，不如任何一次随机初始采样
- **与 LLM+BO 的本质区别**：自建 LLM+BO 保留了 GP 做主优化器，LLM 仅作辅助提议；LLANA 完全放弃数学 surrogates，将整个优化过程交给 LLM 的上下文学习——在 24 维电路优化问题上，这个思路被证明不适用

### 5.4 RL 的潜力与局限
- RL 达到最佳 FoM（-1.24），表明该方法有潜力
- 但 200 步中最佳在 step 101（接近随机阶段结束），后续未见改善
- 可能原因：DDPG 超参数未针对此问题调优，或 GNN 对电路拓扑的编码不够有效

### 5.5 改进方向
- **LLM+BO**：注入电路领域知识（器件物理约束、设计规则）到 prompt；LLM 仅在 BO 停滞时触发；尝试用更强大的模型或用 domain-specific embedding 编码参数空间
- **LLM 辅助优化的适用边界**：LLANA 实验表明，纯 LLM 驱动的优化在 20+ 维连续参数空间中不适合。LLM 更适合低维（<10）或离散/分类参数空间，或作为数学 optimizer 的补充而非替代
- **RL**：增加训练步数（1000+）观察学习效果；尝试 PPO 替代 DDPG
- **实验扩展**：在更多 AnalogGym 电路上验证；引入多目标优化（Pareto front）

## 6. 结论

在 NMCF 电路参数优化的对比实验中：

1. **BO 是当前最实用的方法**：20 次评估内达 FoM = -1.39，效率显著高于所有其他方法
2. **RL 展示最佳终极 FoM（-1.24）**但需 200 次评估，样本效率低，适合仿真预算充足的场景
3. **LLM+BO 未带来增益**：简单接入 LLM 额外增加 90% 时间开销但 FoM 反而退化（-1.51）
4. **LLANA 不适合此类问题**：LLM 在 24 维空间中探索退化，14 次评估后崩溃，最优仅 -1.77。纯 LLM-based surrogate 无法替代 GP 在高维连续空间中的建模能力
5. 所有方法均受限于 NMCF 电路的物理瓶颈（功耗-速度-稳定性 trade-off）
6. **核心启示**：LLM 辅助电路优化的有效路径不是替代数学 optimizer，而是在低维子问题或领域知识注入上发挥作用

## 参考

- AnalogGym: https://github.com/CODA-Team/AnalogGym
- BoTorch: https://botorch.org/
- LLAMBO: https://github.com/tennisonliu/LLAMBO
- LLANA: https://github.com/dekura/LLANA
