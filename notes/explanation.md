# AnalogGym 模拟电路优化 — 项目解析

## 整体目标

这是一个研究生作业项目，目的是**用 LLM 辅助模拟电路参数优化**，对比纯 RL、纯 BO、LLM+BO、LLANA 四种方法在 NMCF 电路的优化表现。

---

## 一、AnalogGym 是什么

**AnalogGym** 是一个开源 benchmark，提供了一组模拟电路的 **gym 环境**。每个电路被封装成一个 `gym.Env`：
- `.step(action)` → 根据参数跑一次 ngspice 仿真 → 返回 FoM（电路性能综合分数）
- 电路的 **参数空间** 就是 gym 的 action space

RL 方法直接调用 `.step()`，BO / LLM 方法也一样——底层都是 ngspice。不需要看懂 ngspice 的内部逻辑，只需要知道：**传参 → 仿真 → 返回分数**。

项目目录 `analoggym/RGNN_RL/` 包含：
- `AMP_NMCF.py` — NMCF 电路的 gym 环境实现（gym.Env 子类）
- `main_AMP.py` — RL 训练入口
- `ddpg.py` — DDPG agent 实现
- `models.py` — GNN 模型（RGCN/GCN/GAT/MLP）
- `ckt_graphs.py` — 电路拓扑转图结构
- `dev_params.py` — 生成 Spice 命令提取 BSIM4 器件参数

AnalogGym 内置了 14 种放大器电路（NMCF、DFCFC1/2、AZC、AFFC 等），本项目集中在 **NMCF**。

---

## 二、dev_params.py 是做什么的

### 核心功能

`dev_params.py` 生成一段 **ngspice 的 DCOP 分析脚本**，用来提取每个晶体管（MOSFET）的内部物理参数。

具体来说：
- 电路中每个晶体管（如 M0, M8, M10...）在 Spice 网表中都被实例化为一个 **subcircuit**（包含 SPICE 模型）
- ngspice 仿真后，可以读取晶体管内部的工作点参数：**跨导（gm）、输出电导（gds）、阈值电压（vth）、漏电流（id）、各端口电容（cgs, cgd, cdb）**等
- `dev_params.py` 生成 `let gm_M0=@m.xm0[...]` 这样的命令，告诉 ngspice 「把这些参数的值写出来」
- 运行它之后会生成 `simulations/AMP_NMCF_dev_params.spice` 文件

### 它和 RL 的关系

RL 需要 **状态表示（state/observation）**。如果只知道「你把 W、L、M 设成了多少」是不够的——RL agent 还需要知道**晶体管当前实际工作在什么状态**（Vth 多少、gm 多少、有没有进入饱和区等）。

```
                     dev_params.py 提取的器件参数
参数 (W/L/M) → ngspice 仿真 →  ───────────────────────→ GNN → RL state
                               FoM (reward)            (12 features/transistor)
```

所以 `dev_params.py` 是 **RL 状态表示的关键依赖**。一个典型的状态向量包含每个晶体管的 12 个特征（gm, gds, vth, vdsat, id, cgs, cgd 等），再加上 5 个全局电路参数。

### 为什么老师要单独提供一个替换版本

AnalogGym 自带的 `dev_params.py` 可能和 SKY130 PDK 的器件名称不完全匹配（不同 PDK 版本，MOS 的型号名可能不同）。老师提供了一个修正版，确保器件名称与当前 PDK 兼容。

### BO/LLM+BO/LLANA 需要它吗

**不需要**。BO 等方法的输入直接是参数向量 → 输出是 FoM，不需要中间状态表示。`dev_params.py` 只在 RL 的 GNN 状态提取中使用。

---

## 三、NMCF 电路

### 基本概念

NMCF（Nested Miller Compensation with Feedforward）是一种**嵌套密勒补偿**放大器拓扑，属于「三级运算放大器」的一种。

它有 **24 个可调参数**：
- **7 个 MOSFET**（M0, M8, M10, M11, M17, M21, M23），每个有 3 个参数（W 宽、L 长、M 叉指数）→ 21 个
- **1 个偏置电流**（Ib）
- **2 个补偿电容乘数**（M_C0, M_C1）

所有参数被归一化到 **[-1, 1]** 区间。

### 仿真做了什么

每次 ngspice 跑**两个仿真**（`TB_Amplifier_ACDC.cir` 和 `TB_Amplifier_Tran.cir`）：
1. **AC/DC 仿真**：测量增益、GBW、相位裕度、CMRR、PSRP/PSRN、功耗、失调电压、温度系数
2. **瞬态仿真**：测量压摆率（Slew Rate）、建立时间（Settling Time）

共 11 个指标，每个和设计规范对比后计算得分（≤ 0，0=完美），FoM 是所有得分的总和。

### 常见丢分项

NMCF 的典型设计 trade-off：
- **功耗高**（Power）：三级运放需要更多偏置电流
- **相位裕度差**（Phase Margin）：三级运放天然有相位问题
- **建立时间长**：与 GBW 和 PM 耦合

---

## 四、四种优化方法

### 4.1 RL（DDPG + R-GCN）

**核心思路**：把电路优化建模为马尔可夫决策过程，每个 step 建议一组参数，看 FoM 变好还是变差。

**状态表示**（`ckt_graphs.py` → `GraphAMPNMCF`）：
- 电路拓扑 → 有向图，28 个节点（晶体管 + 偏置源 + 输出节点），边表示电流/电压关系
- 每个晶体管节点：12 个器件物理参数
- 全局节点：5 个电路级参数

**GNN 编码器**（`models.py` → `ActorCriticRGCN`）：
- R-GCN（关系图卷积）处理电路图，学习节点间的关系
- Actor 输出 24 维连续动作（参数变化方向）
- Critic 评估当前状态+动作的价值

**策略**（`ddpg.py`）：
- DDPG（深度确定性策略梯度），off-policy
- 带探索噪声（uniform noise），前 1000 步纯随机探索
- 200 步训练中，最优 FoM 出现在 step 101（刚好过随机阶段）

**运行入口**：`experiments/run_rl.py` → 调用 `analoggym/RGNN_RL/` 中的 DDPG agent

---

### 4.2 BO（Bayesian Optimization）

**核心思路**（`bo_baseline/bo_solver.py`）：
1. **Sobol 序列**采样 10 个初始点（比纯随机覆盖更均匀）
2. 用 **Gaussian Process** 拟合已有的 (参数 → FoM) 数据，形成对未见区域的 FoM 估计（均值 + 不确定度）
3. 用 **Expected Improvement**（EI）选下一个评估点：平衡 exploitation（均值高的区域）和 exploration（不确定度高的区域）
4. 重复 10 轮，每轮 1 次 ngspice 仿真

10 轮 BO 之后，20 次评估内收敛到 FoM = -1.39。

为什么效率最高：GP 在 24 维空间能做有意义的回归（不像 LLM），每次仿真只需 ~4s。

---

### 4.3 LLM+BO（自建混合方法）

**核心思路**（`llmbo/llmbo_solver.py`）：
- BO 做主力 optimizer，**每轮额外叫 LLM 提 2 个候选点**
- LLM 看到的内容：历史 (参数 → FoM) 列表，例如「参数 [0.3, -0.5, ...] → FoM -1.5」
- 每轮评估 3 个点（1 个 BO 提议 + 2 个 LLM 提议），选最优的进入下一轮

**LLM 接口**（`llmbo/llm_interface.py`）：
- 通过 Anthropic SDK 兼容接口调 DeepSeek
- Prompt 很简单：给 LLM 看历史数据，让它提议有希望的参数

**为什么没超过 BO**：
- LLM 在 24 维空间提的点质量不稳定，经常不如 EI
- API 延迟高（每轮 ~60s），占实验时间的 90%+
- LLM 对数值优化缺乏直觉

---

### 4.4 LLANA（LLM 替代 GP 的 BO）

**核心思路**：
LLANA 比自建 LLM+BO 更激进——**完全放弃 GP**，用 LLM 做两件事：
- **Surrogate**：LLM 看历史数据预测候选点的 FoM（替代 GP 的回归）
- **Acquisition**：LLM 提议新候选点（替代 EI 优化）

**LLANA 的工作流程**：
1. 随机采样 5 个初始点
2. 每轮：
   - Acquisition Phase：给 LLM 看「历史参数 + FoM」的 few-shot 示例，让它提议 5 个新候选配置
   - Discriminative Surrogate Phase：给 LLM 看历史数据 + 每个候选配置，让它预测每个候选的 FoM
   - 基于 LLM 预测的 FoM 算 Expected Improvement，选 EI 最大的点 → ngspice 真实评估
3. 重复 15 轮

**为什么要 patch**（`scripts/llana_patches/`）：
- LLANA 是为 Ollama/OpenAI 写的，默认不支持自定义 `base_url`
- `discriminative_sm.py`：修了 n=1 循环（DeepSeek 不支持 n>1）
- `acquisition_function.py`：修了 langchain 导入问题
- 其他 patch：logger 配置、timeout 设置

**运行入口**：`experiments/run_llana.py` → 调 `llana/llambo/llambo.py`

**为什么更差**：
- LLM 在不同轮次反复输出退化候选（全零向量、交替 ±0.5、交替 ±1）
- 只用 14 次评估就触发「候选点都是重复的，放弃」
- LLM surrogate 预测几乎是对历史值的机械复述

---

## 五、实验结果

| 方法 | 最优 FoM | 仿真次数 | 耗时 | 效率 |
|------|----------|----------|------|------|
| **BO** | -1.39 | 20 | 83s | **最优** |
| RL | **-1.24** | 200 | 1903s | 最终最好但样本效率低 |
| LLM+BO | -1.51 | 40 | 1345s | LLM 未带来增益 |
| LLANA | -1.77 | 14 | 1063s | 探索退化，早停 |

**结论**：GP 仍然是高维连续优化中最可靠的 surrogate model。LLM 不适合直接替代数学 optimizer，但可以在低维子问题或领域知识注入上发挥作用。

---

## 六、项目文件导航

```
analogopt/
│
├── 实验与代码 ──────────────────────────────────
│
├── experiments/                    ← 实验入口（你直接跑的脚本）
│   ├── run.py                     ← BO / LLM+BO 实验
│   ├── run_rl.py                  ← RL 实验（DDPG+RGCN）
│   └── run_llana.py               ← LLANA 实验
│
├── bo_baseline/                    ← 纯 BO 求解器
│   └── bo_solver.py               ← GP + EI（BoTorch），171行
│
├── llmbo/                          ← LLM+BO 求解器（自建混合）
│   ├── llm_interface.py           ← LLM API 封装（Anthropic SDK）
│   └── llmbo_solver.py            ← BO + LLM 混合候选生成
│
├── runner/                         ← 实验运行器
│   └── experiment.py              ← 实验循环、结果保存/加载、方法对比
│
├── env_interface/                  ← AnalogGym 适配
│   └── analoggym_adapter.py       ← 封装 AMPNMCFEnv 为统一 evaluate() 接口
│
├── tests/                          ← 单元测试（15个）
│
├── 外部依赖 ────────────────────────────────────
│
├── analoggym/                      ← [外部] AnalogGym benchmark clone
│   └── RGNN_RL/                   ← 14 种放大器 gym 环境 + RL 代码
│       ├── AMP_NMCF.py            ← NMCF 电路 gym 环境（核心）
│       ├── main_AMP.py            ← RL 训练入口（AnalogGym 原始 demo）
│       ├── ddpg.py                ← DDPG agent
│       ├── models.py              ← GNN 模型（RGCN/GCN/GAT/MLP）
│       ├── ckt_graphs.py          ← 电路拓扑转图结构
│       ├── dev_params.py          ← 提取 BSIM4 器件参数（RL 状态表示需要）
│       └── utils.py               ← ActionNormalizer, OutputParser
│
├── llana/                          ← [外部] LLANA 框架 clone
│   └── llambo/                    ← LLANA 核心代码
│       ├── llambo.py              ← LLANA 主循环
│       ├── acquisition_function.py ← LLM acquisition（提议候选点）
│       ├── discriminative_sm.py   ← LLM discriminative surrogate（预测FoM）
│       └── ...
│
├── scripts/llana_patches/         ← 我们写的 LLANA 兼容性补丁
│   └── llambo/*.py                ← 覆盖 llana/llambo/ 同级文件
│
├── 报告与文档 ──────────────────────────────────
│
├── report/                        ← 作业报告
│   ├── report.md                  ← 主报告（6 节）
│   ├── plot_results.py            ← 生成对比图
│   └── figures/comparison.png     ← 四方法对比图
│
├── results/                        ← 实验结果 JSON（gitignored）
│   ├── bo_nmcf_results.json       ← BO 实验：10 轮，best=-1.39
│   ├── llmbo_nmcf_results.json    ← LLM+BO 实验：10 轮，best=-1.51
│   ├── rl_nmcf_results.json       ← RL 实验：200 步，best=-1.24
│   └── llana_nmcf_results.json    ← LLANA 实验：14 次评估，best=-1.77（崩溃）
│
├── 项目管理 ────────────────────────────────────
│
├── README.md                       ← 项目概览、运行命令、结果摘要
├── LOG.md                          ← 实验日志/开发记录
├── shell.nix                       ← NixOS 开发环境定义
├── docs/                           ← 本地设计文档（gitignored）
│
└── ~/.claude/projects/.../memory/  ← 项目记忆（跨会话保持上下文）
    ├── project_analogopt.md        ← 项目进度/待办
    └── user_prefs.md               ← Python venv/nix 偏好
```

---

## 七、运行命令备忘

```bash
# 进入开发环境（ngspice + Python 311 + numpy/scipy/matplotlib/pandas）
nix-shell

# 跑测试
~/.venv/py311/bin/python -m pytest tests/ -v

# 各实验
nix-shell -p ngspice --run "~/.venv/py311/bin/python experiments/run.py --method bo --iterations 20"
nix-shell -p ngspice --run "~/.venv/py311/bin/python experiments/run.py --method llmbo --iterations 20"
nix-shell -p ngspice --run "~/.venv/py311/bin/python experiments/run_rl.py --steps 200"
nix-shell -p ngspice --run "~/.venv/py311/bin/python experiments/run_llana.py"

# 生成对比图
~/.venv/py311/bin/python report/plot_results.py
```
