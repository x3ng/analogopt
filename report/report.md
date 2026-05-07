# LLM辅助模拟电路参数优化 — 实验报告

## 1. 引言

模拟电路参数优化是芯片设计中的关键步骤，传统上依赖人工调参或强化学习（RL）。本实验系统对比三种优化方法在 NMCF 放大器电路上的表现：RL（RGNN+DDPG）、贝叶斯优化（BO），以及 LLM 增强的贝叶斯优化（LLM+BO）。

## 2. 方法

### 2.1 问题定义
- **电路**：NMCF 放大器（SKY130 PDK）
- **参数空间**：24 维连续参数（7 个 MOSFET 的 W/L/M、偏置电流、2 个电容乘数），归一化到 [-1, 1]
- **优化目标**：最大化 Figure of Merit（FoM），FoM 为 11 个性能指标的分数之和（TC、Power、Vos、CMRR、DC Gain、GBW、Phase Margin、PSRP、PSRN、Slew Rate、Settling Time），最高为 0

### 2.2 RL 方法（Baseline）
- RGNN_RL：Graph Neural Network + DDPG
- 使用 AnalogGym 的 `main_AMP.py`
- GNN 编码电路拓扑，DDPG 学习最优参数

### 2.3 Bayesian Optimization
- Gaussian Process surrogate model (BoTorch)
- Expected Improvement 采集函数
- Sobol 序列初始化

### 2.4 LLM+BO 方法
- LLM 分析优化历史，提议候选参数点
- 与 BO 提议点混合（BO 提供系统性探索，LLM 提供领域知识）
- LLM 通过 Anthropic 兼容 API 调用（DeepSeek / Qwen）

## 3. 实验设置

| 参数 | 值 |
|------|-----|
| 电路 | NMCF (SKY130) |
| 参数维度 | 24 |
| 初始采样 | 10 (Sobol) |
| 优化迭代 | 20 (total 30 evaluations) |
| 仿真器 | Ngspice 45 |
| BO 框架 | BoTorch |
| LLM API | DeepSeek (deepseek-v4-pro[1m]) |

## 4. 结果

| 方法 | 最佳 FoM | 总时间 | 迭代数 |
|------|----------|--------|--------|
| BO | -1.3923 | 83.8s | 20 |
| LLM+BO | TBD | TBD | 20 |
| RL | TBD | TBD | TBD |

## 5. 讨论

（待填充）

## 6. 结论

（待填充）
