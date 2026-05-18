# LLM辅助模拟电路参数优化

对比 **RL** vs **BO** vs **LLM+BO** vs **LLANA** 在 NMCF 放大器电路参数优化上的表现。

## 项目结构

```
analogopt/
├── README.md
├── pyproject.toml               ← Python 依赖 + uv 配置
├── uv.lock                      ← 精确版本锁 (可复现)
├── shell.nix                    ← 只提供 ngspice (不是 Python)
│
├── analoggym/                   ← [外部] AnalogGym 开源 benchmark
├── llana/                       ← [外部] LLANA (LLM surrogate BO)
├── llambo/                      ← [外部] LLAMBO (LLM Bayesian Optimization)
│
├── llmbo/                       ← [自建] BO + LLM+BO 求解器 (BoTorch GP+EI / LLM 增强)
├── env_interface/               ← [自建] AnalogGym 适配层
├── experiments/                 ← [自建] 实验入口 + 工具类
├── scripts/                     ← LLANA patch 文件
│
├── results/                     ← 实验结果 (JSON)
├── report/                      ← 作业报告
└── docs/                        ← 参考文档
```

## 环境

### Python 环境

使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖：

```bash
uv sync                          # 创建 .venv/ 并安装所有依赖
uv run python ...                # 在 .venv/ 中运行 Python
```

### Ngspice

`shell.nix` 只提供 `ngspice` 系统二进制（仿真器），不碰 Python：

```bash
nix-shell --run "uv run python experiments/run.py --method bo"
```

不用 Nix 的话，系统装好 ngspice 后直接 `uv run python ...` 即可。

### LLM API 配置

LLM+BO 实验需要 Anthropic 兼容 API，通过环境变量配置：

```bash
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_AUTH_TOKEN="your-api-key"
export ANTHROPIC_MODEL="your-model-name"
```

LLANA 使用 OpenAI 兼容 endpoint，配置方法见 `experiments/run_llana.py` 头部注释。

## 初始化

```bash
# Clone vendored repos
git clone https://github.com/CODA-Team/AnalogGym.git analoggym
git clone --depth 1 https://github.com/dekura/LLANA.git llana
cp scripts/llana_patches/llambo/*.py llana/llambo/

# 解压 PDK
unzip analoggym/PDK/sky130_pdk.zip -d analoggym/RGNN_RL/mosfet_model/

# 安装 Python 依赖
uv sync
```

## 运行实验

```bash
# BO 实验
uv run python experiments/run.py --method bo --iterations 20

# LLM+BO 实验 (需先配置 API 环境变量)
uv run python experiments/run.py --method llmbo --iterations 20

# 同时跑 BO + LLM+BO
uv run python experiments/run.py --method both --iterations 20

# RL 实验
uv run python experiments/run_rl.py --steps 500

# LLANA 实验 (LLM 替代 GP 做 surrogate + acquisition)
uv run python experiments/run_llana.py

# 快速验证 LLANA (3+3 轮)
LLANA_TRIALS=3 LLANA_INITIAL=3 uv run python experiments/run_llana.py

# 查看结果
uv run python -c "
from experiments.utils import ExperimentRunner
bo = ExperimentRunner.load_results('results/bo_nmcf_results.json')
llmbo = ExperimentRunner.load_results('results/llmbo_nmcf_results.json')
print(ExperimentRunner.compare([bo, llmbo]))
"
```

## 实验结果 (2026-05-10)

### 四方法对比

| 方法 | 最佳 FoM | 时间 | 评估次数 | 备注 |
|------|----------|------|----------|------|
| RL (DDPG+RGCN) | **-1.2446** | 1903.9s | 200 | **最优 FoM**, 样本效率低 |
| **BO (BoTorch)** | -1.3923 | 83.8s | 20 | GP + EI, **效率最佳** |
| LLM+BO (自建) | -1.5063 | 1345.7s | 40 | LLM 辅助提议, 未超纯 BO |
| LLANA | -1.7673 | 1063.5s | 14 | LLM 替代 GP, 探索退化, Trial 9 崩溃 |

### 关键发现

- **BO 效率最高**：20 次评估达 -1.39，适合有限仿真预算
- **RL 潜力最大**：200 步达 -1.24，但需更多样本
- **LLM 方法均未超越 BO**：自建 LLM+BO 和 LLANA 均未带来优化效果提升
- **LLANA 不适合高维连续空间**：LLM 在 24 维空间中探索退化为离散模式，无法替代 GP

## 参考

- [AnalogGym](https://github.com/CODA-Team/AnalogGym)
- [BoTorch](https://botorch.org/)
- [LLAMBO](https://github.com/tennisonliu/LLAMBO)
- [LLANA](https://github.com/dekura/LLANA)
