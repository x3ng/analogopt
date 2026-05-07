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
├── experiments/                 ← [自建] 实验入口
│   ├── run.py                   ←    BO / LLM+BO
│   └── run_rl.py                ←    RL (DDPG+RGCN)
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
nix-shell -p ngspice --run "python experiments/run.py --method bo --iterations 20"

# LLM+BO 实验 (需要 API 环境变量)
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_AUTH_TOKEN="..."
export ANTHROPIC_MODEL="deepseek-v4-pro[1m]"
nix-shell -p ngspice --run "python experiments/run.py --method llmbo --iterations 20"

# RL 实验 (DDPG + RGCN)
nix-shell -p ngspice --run "python experiments/run_rl.py --steps 500"
```

## 实验结果 (2026-05-07)

### 三方对比

| 方法 | 最佳 FoM | 时间 | 评估次数 | 备注 |
|------|----------|------|----------|------|
| **BO (BoTorch)** | -1.3923 | 83.8s | 30 | GP + EI, **效率最佳** |
| LLM+BO | -1.5063 | 1345.7s | 30 | BO + DeepSeek, LLM 未带来增益 |
| RL (DDPG+RGCN) | **-1.2446** | 1903.9s | 200 | **最优 FoM**, 需更多样本 |

### 关键发现
- **BO 效率最高**：30 次评估达 -1.39，适合有限仿真预算
- **RL 潜力最大**：200 步达 -1.24，但需 6.7x 样本。CPU 可跑（模型很小，瓶颈是仿真）
- **LLM+BO 未达预期**：简单 prompt 无电路领域知识，API 耗时远大于 BO 改进
- LLM 提议的候选点未带来明显收益，可能原因：prompt 设计简单、LLM 缺乏电路领域训练

## dev_params.py 说明

`dev_params.py` 用于生成 Spice 命令来提取 BSIM4 晶体管内部参数（Vth, gm, gds 等），供 RL 的状态表示使用。对于 BO/LLMBO 不需要——我们只用 FoM（reward）。

## 参考

- [AnalogGym](https://github.com/CODA-Team/AnalogGym) - 模拟电路优化 benchmark
- [BoTorch](https://botorch.org/) - Bayesian Optimization 框架
- [LLAMBO](https://github.com/tennisonliu/LLAMBO) - LLM+BO 参考
- [LLANA](https://github.com/dekura/LLANA) - LLM+BO 参考
