# LLM辅助模拟电路参数优化 — 实验日志

## 目标
对比 **RL (RGNN)** vs **BO (BoTorch)** vs **LLM+BO (自建)** 在 NMCF 电路参数优化上的表现。

## 环境配置

| 组件 | 来源 | 版本 |
|------|------|------|
| OS | NixOS | — |
| Python | ~/.venv/py311 | 3.11.15 |
| ngspice | nix-shell -p ngspice | 45 |
| PyTorch | uv pip install | 2.x (latest) |
| BoTorch | uv pip install | latest |
| GPyTorch | uv pip install | latest |
| Anthropic SDK | uv pip install | 0.100.0 |
| gymnasium | uv pip install | 1.3.0 |
| torch-geometric | uv pip install | 2.7.0 |
| AnalogGym | git clone | main branch |
| PDK | AnalogGym 自带 | SKY130 |
| LLM API (主) | DeepSeek (cc-ds.nix) | deepseek-v4-pro[1m] |
| LLM API (备) | Qwen (cc-qw.nix) | qwen3.6-plus |

## 关键发现

- **不需要 Docker**：ngspice (nix) + Python venv 直接跑 AnalogGym
- Docker Hub 上 `chenzhenxin/analoggym_rgcn` 镜像为私有，不可用
- ngspice v45 兼容 AnalogGym 的 SKY130 PDK（原要求 v41-43）
- NMCF 环境：24 维 action，29x12 维 observation，Reward ≤ 0

## 完成进度

### 2026-05-07

- [x] 项目结构初始化
- [x] Clone AnalogGym，替换 dev_params.py
- [x] 实现 BO 求解器 (`bo_baseline/bo_solver.py`) — 4 tests pass
- [x] 实现 LLM 接口 (`llmbo/llm_interface.py`) — 3 tests pass
- [x] 实现 LLMBO 求解器 (`llmbo/llmbo_solver.py`) — 4 tests pass
- [x] 实现实验运行器 (`runner/experiment.py`) — 4 tests pass
- [x] 实现 AnalogGym 适配器 (`env_interface/analoggym_adapter.py`)
- [x] 验证 ngspice 仿真能跑通（一次 step ~数秒）
- [ ] 跑 RL baseline
- [ ] 跑 BO baseline
- [ ] 跑 LLM+BO 实验
- [ ] 结果对比 + 可视化
- [ ] 撰写报告

## 实验结果

（待填充）
