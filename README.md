# LLM辅助模拟电路参数优化

对比 **RL** vs **BO** vs **LLM+BO** 在 NMCF 放大器电路参数优化上的表现。

## 项目结构

```
analogopt/
├── README.md                    ← 本文件
│
├── analoggym/                   ← [外部] AnalogGym 开源benchmark
│   ├── RGNN_RL/                 ←    仿真环境 + RL 代码
│   └── PDK/                     ←    SKY130 工艺包
│
├── bo_baseline/                 ← [自建] 贝叶斯优化求解器
│   └── bo_solver.py             ←    GP + EI (BoTorch)
│
├── llmbo/                       ← [自建] LLM+BO 求解器
│   ├── llm_interface.py         ←    LLM API 封装 (Anthropic SDK)
│   └── llmbo_solver.py          ←    BO + LLM 混合候选生成
│
├── runner/                      ← [自建] 实验运行器
│   └── experiment.py            ←    实验循环、结果保存/加载/对比
│
├── env_interface/               ← [自建] AnalogGym 适配
│   └── analoggym_adapter.py     ←    封装为统一评估接口
│
├── experiments/                 ← [自建] 实验入口
│   ├── run.py                   ←    BO / LLM+BO
│   └── run_rl.py                ←    RL (DDPG+RGCN)
│
├── tests/                       ← [自建] 15 个单元测试
├── results/                     ← 实验结果 (JSON, 不入 git)
├── report/                      ← 作业报告
└── docs/                        ← 设计文档
```

## 依赖

- **Python** ≥ 3.10
- **Ngspice** ≥ 41 (v45 已验证)
- **SKY130 PDK**（AnalogGym 自带，需解压）

### Python 包

```
pip install botorch gpytorch numpy anthropic gymnasium tabulate torch-geometric pytest
```

### 初始化

```bash
# Clone AnalogGym
git clone https://github.com/CODA-Team/AnalogGym.git analoggym

# 解压 PDK
unzip analoggym/PDK/sky130_pdk.zip -d analoggym/RGNN_RL/mosfet_model/

# 验证
python -m pytest tests/ -v

# Clone LLANA (LLM-Enhanced BO, arXiv 2406.05250)
git clone --depth 1 https://github.com/dekura/LLANA.git llana
cp scripts/llana_patches/llambo/*.py llana/llambo/
```

### Nix 用户

项目提供了 `shell.nix`，一键进入开发环境：

```bash
nix-shell
```

需自行配置 LLM API 环境变量（见下方）。

## LLM API 配置

LLM+BO 实验需要 Anthropic 兼容 API。通过环境变量配置：

```bash
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"   # 或其他兼容端点
export ANTHROPIC_AUTH_TOKEN="your-api-key"
export ANTHROPIC_MODEL="your-model-name"
```

支持的兼容端点：DeepSeek、Qwen (DashScope)、火山引擎等。

## 运行实验

```bash
# BO 实验
python experiments/run.py --method bo --iterations 20

# LLM+BO 实验 (需先配置 API 环境变量)
python experiments/run.py --method llmbo --iterations 20

# RL 实验
python experiments/run_rl.py --steps 500

# LLANA 实验 (LLM 替代 GP 做 surrogate + acquisition, 需先配置 LLANA)
nix-shell -p ngspice --run "python experiments/run_llana.py"

# 快速验证 LLANA 管道 (3+3 轮)
LLANA_TRIALS=3 LLANA_INITIAL=3 nix-shell -p ngspice --run \
    "python experiments/run_llana.py"

# 查看结果
python -c "
from runner.experiment import ExperimentRunner
bo = ExperimentRunner.load_results('results/bo_nmcf_results.json')
llmbo = ExperimentRunner.load_results('results/llmbo_nmcf_results.json')
print(ExperimentRunner.compare([bo, llmbo]))
"
```

## 实验结果 (2026-05-07)

### 三方对比

| 方法 | 最佳 FoM | 时间 | 评估次数 | 备注 |
|------|----------|------|----------|------|
| **BO (BoTorch)** | -1.3923 | 83.8s | 30 | GP + EI, **效率最佳** |
| LLM+BO | -1.5063 | 1345.7s | 30 | BO + LLM 提议, LLM 未带来增益 |
| RL (DDPG+RGCN) | **-1.2446** | 1903.9s | 200 | **最优 FoM**, CPU 可跑 |

### 关键发现

- **BO 效率最高**：30 次评估达 -1.39，适合有限仿真预算
- **RL 潜力最大**：200 步达 -1.24，但需 6.7x 样本。模型小（电路图级 GNN），瓶颈在仿真
- **LLM+BO 未达预期**：简单 prompt 缺乏电路领域知识，API 耗时远超 BO 改进

## dev_params.py

生成 Spice 命令提取 BSIM4 晶体管参数（Vth, gm, gds 等），供 RL 状态表示。BO/LLMBO 仅用 FoM，不需要。

## 参考

- [AnalogGym](https://github.com/CODA-Team/AnalogGym)
- [BoTorch](https://botorch.org/)
- [LLAMBO](https://github.com/tennisonliu/LLAMBO)
- [LLANA](https://github.com/dekura/LLANA)
