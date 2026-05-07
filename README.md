# LLM辅助模拟电路参数优化

对比 **RL** vs **BO** vs **LLM+BO** 三种方法在 NMCF 放大器电路参数优化上的表现。

## 项目结构

```
analogopt/
├── README.md                    ← 本文件
├── LOG.md                       ← 实验日志（时间线记录）
│
├── analoggym/                   ← [外部] AnalogGym 开源benchmark (git clone)
│   ├── RGNN_RL/                 ←    仿真环境 + RL 代码 (main_AMP.py 等)
│   ├── RGNN_RL_Docker/          ←    Docker 构建（本项目不用）
│   └── PDK/                     ←    SKY130 工艺包 (sky130_pdk.zip)
│
├── bo_baseline/                 ← [自建] 贝叶斯优化求解器
│   └── bo_solver.py             ←    GP + EI, 基于 BoTorch
│
├── llmbo/                       ← [自建] LLM+BO 求解器
│   ├── llm_interface.py         ←    LLM API 封装 (Anthropic SDK 兼容)
│   └── llmbo_solver.py          ←    BO + LLM 混合候选生成
│
├── runner/                      ← [自建] 实验运行与结果管理
│   └── experiment.py            ←    实验循环、结果保存/加载/对比
│
├── env_interface/               ← [自建] AnalogGym 环境适配
│   └── analoggym_adapter.py     ←    封装 AMPNMCFEnv 为标准评估接口
│
├── experiments/                 ← [自建] 实验入口脚本
├── tests/                       ← [自建] 15 个单元测试
├── results/                     ← 实验结果 (JSON)
├── report/                      ← 作业报告
└── docs/                        ← 设计文档
```

### 依赖关系

```
[自建 optimizers]         [外部仿真]
   bo, llmbo          ←  →  analoggym/ngspice
        ↓                         
   runner (实验循环)             
        ↓
   results/ (JSON)
```

## 环境配置

### 系统依赖
- **NixOS** + `nix-shell`
- **Python**: `~/.venv/py311` (uv 管理), Python 3.11
- **Ngspice**: `nix-shell -p ngspice` (v45)
- **SKY130 PDK**: `unzip analoggym/PDK/sky130_pdk.zip -d analoggym/RGNN_RL/mosfet_model/`

### Python 包
```bash
uv pip install --python ~/.venv/py311/bin/python \
    botorch gpytorch numpy anthropic gymnasium tabulate \
    torch-geometric pytest matplotlib pandas
```

### LLM API 配置

| 文件 | 提供商 | 模型 | 说明 |
|------|--------|------|------|
| `~/nix-shell/cc-ds.nix` | DeepSeek | deepseek-v4-pro[1m] | 用量计费 |
| `~/nix-shell/cc-qw.nix` | Qwen/阿里 | qwen3.6-plus | 老师提供 |
| `~/nix-shell/cc-vo.nix` | 火山引擎 | glm-5.1 | coding plan (5h限制) |

API 通过环境变量配置（Anthropic SDK 兼容）：
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_AUTH_TOKEN`
- `ANTHROPIC_MODEL`

### 一键启动实验环境
```bash
# 终端1: 配置 API（选一个）
nix-shell ~/nix-shell/cc-ds.nix

# 终端2 或合并: 运行实验
nix-shell -p ngspice --run "python experiments/run.py --method bo"
```

## 运行实验

```bash
# 先解压 PDK（仅需一次）
unzip analoggym/PDK/sky130_pdk.zip -d analoggym/RGNN_RL/mosfet_model/

# 跑测试验证
python -m pytest tests/ -v

# BO 实验 (20 迭代, 10 初始采样)
nix-shell -p ngspice --run "python run_real_experiments.py --method bo --iterations 20"

# LLM+BO 实验 (需要 API 环境变量)
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_AUTH_TOKEN="..."
export ANTHROPIC_MODEL="deepseek-v4-pro[1m]"
nix-shell -p ngspice --run "python run_real_experiments.py --method llmbo --iterations 20"

# 查看结果对比
python -c "
from runner.experiment import ExperimentRunner
bo = ExperimentRunner.load_results('results/bo_nmcf_results.json')
llmbo = ExperimentRunner.load_results('results/llmbo_nmcf_results.json')
print(ExperimentRunner.compare([bo, llmbo]))
"
```

## 实验结果

### BO vs LLM+BO (2026-05-07, 各 30 次评估)

| 方法 | 最佳 FoM | 时间 | 备注 |
|------|----------|------|------|
| BO (BoTorch) | **-1.3923** | 83.8s | GP + EI |
| LLM+BO | -1.5063 | 1345.7s | BO + DeepSeek LLM 提议 |

#### 迭代对比

```
Iter    BO FoM    LLMBO FoM    BO Best   LLMBO Best
   0   -1.5063    -1.5063     -1.5063    -1.5063
   1   -1.3923    -1.6574     -1.3923    -1.5063    ← BO 找到最优
   2   -1.5712    -1.8633     -1.3923    -1.5063
   3   -2.0489    -3.6479     -1.3923    -1.5063
   4   -1.6102    -1.7518     -1.3923    -1.5063
   5   -2.5628    -1.7511     -1.3923    -1.5063
   6   -3.4556    -1.6444     -1.3923    -1.5063
   7   -1.5685    -1.6172     -1.3923    -1.5063
   8   -1.5713    -2.3935     -1.3923    -1.5063
   9   -2.8557    -4.1196     -1.3923    -1.5063
```

### 关键发现
- BO 在当前设置下优于 LLM+BO（更好的 FoM，更快的速度）
- LLM API 调用显著增加时间成本（~1345s vs ~84s）
- LLM 提议的候选点未带来明显收益，可能原因：prompt 设计简单、LLM 缺乏电路领域训练

## dev_params.py 说明

`dev_params.py` 用于生成 Spice 命令来提取 BSIM4 晶体管内部参数（Vth, gm, gds 等），供 RL 的状态表示使用。对于 BO/LLMBO 不需要——我们只用 FoM（reward）。

## 参考

- [AnalogGym](https://github.com/CODA-Team/AnalogGym) - 模拟电路优化 benchmark
- [BoTorch](https://botorch.org/) - Bayesian Optimization 框架
- [LLAMBO](https://github.com/tennisonliu/LLAMBO) - LLM+BO 参考
- [LLANA](https://github.com/dekura/LLANA) - LLM+BO 参考
