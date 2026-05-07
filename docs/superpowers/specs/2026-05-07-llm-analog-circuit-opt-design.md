# LLM辅助模拟电路参数优化 设计文档

**Goal:** 对比 RL、BO、LLM+BO 三种方法在 AnalogGym NMCF 电路参数优化上的效率和结果

**Approach:** 混合模式（AnalogGym Docker 跑仿真 + 本地跑优化代码），三条实验路线并行，重点在 LLM+BO 的自建实现与对比

**Tech Stack:** Python 3.10+, AnalogGym (Ngspice+SKY130 PDK), BoTorch, Anthropic SDK (DeepSeek/Qwen API)

---

## 目录结构

```
/home/xen/Project/llmeda/analogopt/
├── dev_params.py               # 已有的参数配置文件
├── analoggym/                  # AnalogGym 源码 (git clone)
├── rl_baseline/                # RL 实验
│   └── run_rl.py               # 跑 main_AMP.py 的脚本
├── bo_baseline/                # 普通 Bayesian Optimization
│   ├── bo_solver.py            # BoTorch BO 实现
│   └── run_bo.py               # BO 实验主脚本
├── llmbo/                      # 自建 LLM+BO
│   ├── llm_interface.py        # LLM API 封装 (DeepSeek/Qwen)
│   ├── llmbo_solver.py         # LLM-enhanced BO
│   └── run_llmbo.py            # LLM+BO 实验主脚本
├── results/                    # 实验结果
│   ├── rl_results.json
│   ├── bo_results.json
│   └── llmbo_results.json
├── report/                     # 作业报告
│   └── report.md
└── docs/superpowers/
    ├── specs/2026-05-07-llm-analog-circuit-opt-design.md
    └── plans/
```

## 组件设计

### 1. AnalogGym 环境

- **职责:** 提供 NMCF 电路仿真 Gym 环境
- **接口:** `env.reset()` → 初始状态, `env.step(action)` → (next_state, reward, done, info)
- **安装:** Docker 方式（RGNN_RL_Docker），通过文件系统共享仿真结果
- **关键文件:** `main_AMP.py`（RL 参考实现）, `ckt_graphs.py`（电路图定义）

### 2. RL Baseline（路线 A）

- **职责:** 运行 AnalogGym 自带的 RGNN_RL 方法作为对比 baseline
- **实现:** 基于 `main_AMP.py`，记录每步的 reward 和参数变化
- **输出:** `results/rl_results.json` — episodes × rewards, 最终优化参数, 运行时间

### 3. BO Baseline（路线 B）

- **职责:** 使用 BoTorch 做标准 Bayesian Optimization
- **实现:** 
  - Gaussian Process surrogate model
  - Expected Improvement (EI) 采集函数
  - 每次迭代：BO 提议参数 → AnalogGym 仿真 → 更新 GP
- **输出:** `results/bo_results.json` — iterations × rewards, 最终参数, 运行时间

### 4. LLM+BO（路线 C）

- **职责:** LLM 增强的 BO 优化
- **自建实现:**
  - `llm_interface.py`: 封装 Anthropic SDK，支持 DeepSeek/Qwen API
  - `llmbo_solver.py`: LLM 参与以下环节：
    1. 基于历史数据生成有希望的候选参数点
    2. 对 BO 提议的点做合理性检查
    3. 动态调整采集策略
- **LLAMBO/LLANA 尝试:** git clone 后尝试适配 AnalogGym 环境
- **输出:** `results/llmbo_results.json`

### 5. API 配置

- DeepSeek: `https://api.deepseek.com/anthropic`，Anthropic SDK 兼容
- Qwen: `https://dashscope.aliyuncs.com/apps/anthropic`，Anthropic SDK 兼容
- 配置文件统一放在环境变量中，代码通过 `os.environ` 读取

## 实验设计

### 评估指标
- **收敛速度:** 达到目标性能所需迭代次数
- **最终性能:** 最大 FoM (Figure of Merit)
- **样本效率:** 总仿真次数 vs 最终性能
- **稳定性:** 多次运行的方差

### 对比维度
| 方法 | 样本效率 | 收敛速度 | 最终性能 | 额外成本 |
|------|----------|----------|----------|----------|
| RL (RGNN) | baseline | baseline | baseline | 无 |
| BO (BoTorch) | 对比1 | 对比1 | 对比1 | 无 |
| LLM+BO (自建) | 对比2 | 对比2 | 对比2 | LLM API 调用 |
| LLAMBO/LLANA | 可选尝试 | 可选尝试 | 可选尝试 | LLM API 调用 |

## 数据流

```
docker exec analoggym python run_sim.py --params [...] → 仿真结果
                                                           ↓
本地 Python 脚本 ← 读取仿真结果 (stdout/file)
     ↓
BO/LLM 优化逻辑 ← 更新模型，生成新参数
     ↓
docker exec analoggym python run_sim.py --params [...] → 下一轮迭代
```

## 风险与备选

- **Docker 下载失败:** 用代理 `docker pull` 或手动下载镜像
- **LLAMBO/LLANA 不可用:** 自建 LLM+BO 足够完成核心实验
- **API 配额不足:** Qwen 老师提供优先使用，DeepSeek 备用
- **仿真太慢:** 限制最大迭代次数，先用少迭代验证流程
