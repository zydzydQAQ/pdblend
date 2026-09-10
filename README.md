# PDblend

PDblend 是面向大语言模型推理服务的 SLO 约束能耗控制项目。本仓库保留已有部署与 Selective PD 实验文件，并收录 2026-09-10 工作区中可用的 PDblend 源码。

## 目录

| 目录 | 内容 |
| --- | --- |
| `pdblend/` | 主 Python 包 `ecopadg`、可扩展性评估模块、项目配置和测试 |
| `pdblend-next-v1/releases/` | 各版本运行时源码、版本说明、清单和补丁 |
| `pdblend-next-v1/campaign/` | 实验开发与执行脚本、候选实现和补丁；保留原相对路径 |
| `pdblend-next-v1/tests/` | 后续版本测试 |
| `vllm-pd-fork/vllm/` | 工作区中的 vLLM Python 源码及 PDblend 修改 |
| `benchmarks/scripts/` | 工作区中的基准测试脚本 |
| `docker/`、`experiments/`、`scripts/` | 仓库已有的部署配置与 Selective PD 实验工具 |

## 安装主 Python 包

需要 Python 3.10 或更新版本。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./pdblend
```

包名为 `pdblend`，Python 导入名为 `ecopadg`。GPU 服务还需要与目标环境匹配的 CUDA、PyTorch、vLLM 及相应运行配置。`vllm-pd-fork/` 仅包含本次工作区中可用的 Python 源码树，不是完整的 vLLM 构建发行包。

已有 Selective PD 实验的运行方式见 [experiments/selective_pd/README.md](experiments/selective_pd/README.md)。现有 Dockerfile 构建的是该实验环境，不会自动安装新收录的主 Python 包。

## 测试与快照范围

2026-09-10 更新收录可扩展性评估模块及测试、最新速率与 SLO 实验源码和运行时快照，并支持通过 `engine_request_timeout_s` 配置引擎请求超时。已验证可扩展性测试 144 项、异步运行时及回滚验证测试 32 项，共 176 项通过。

主程序测试位于 `pdblend/tests/`，可安装 `pytest` 后按需执行。当前工作区缺少部分测试所引用的 `script/bench/`、`pdblend/new-results/scripts/` 和实验测量表，因此完整测试套件仍需要补齐这些输入；本次源码上传不表示完整测试或 GPU 实验已经通过。

`releases/` 保留不同版本的独立源码，不自动覆盖 `pdblend/src/`。`campaign/` 收录 Python、Shell、补丁及源码配置文件，未收录运行结果、日志和 JSON 实验输入。部分脚本还依赖特定机器路径、容器与运行时状态，需要按实际环境配置后使用。

本次新增内容不包含模型权重、实验输出、依赖安装目录、Python 缓存或本地凭据。仓库原有文件及提交历史保留。
