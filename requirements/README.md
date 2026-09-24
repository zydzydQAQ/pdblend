# pdblend4-v3 环境身份

`pdblend4-v3-image.json` 来自当前正在使用的镜像
`pdblend:l20-cu128-vllm-v1`，包含完整镜像层、配置、Python 包和两份引擎补丁的 SHA256。
它不替代镜像文件；同一镜像须用 `RESTART.md` 的归档导入步骤。

容器 CUDA 基础层是 12.8.1，实际 PyTorch 是 **2.7.1+cu126 / CUDA 12.6**，
vLLM 是 **0.10.1.1**。实验 launcher 使用 `VLLM_USE_V1=1` 和两文件 P2pNccl 补丁。
旧版 vLLM 0.9.2 / V0 / 八文件补丁的安装说明不适用于此分支。

`pdblend4-v3-container.lock` 固定当前镜像的全部已安装 Python 包；源码构建单独安装
当前 checkout 的 pdblend。系统 apt 包与构建时间会使重建镜像具有不同的层及 ID，
因此重建后只能通过 `verify --allow-rebuilt` 校验版本/补丁一致，不能宣称逐层相同。
当前镜像内还包含 nixl 及 cu12/cu13 包；保留这些原有包是为了保持已安装环境一致，
不代表实验改用 NIXL 或 CUDA 13 引擎。

`pdblend4-v3-host.lock` 用于新机器的宿主控制器、预测器、分析及测试环境。
它与容器环境不同：宿主是 **Torch 2.7.0+cu128 / Transformers 4.51.3**。
锁中已将源机不存在的 `file:///opt/pdblend/wheels/` 引用转换为固定版本，
并添加本分支需要的 pybind11、psutil、msgpack 和 pyzmq。
使用 `bash scripts/2026-09-25_bootstrap_host.sh` 在当前 checkout 建立 `.venv`。
可通过 `PDBLEND_PYTHON` 选择 Python 3.10；可通过 `PDBLEND_VENV` 指定独立目标目录。
全部依赖安装使用 `--no-deps` 消费已锁定闭包，并通过 `pip check` 验收。

归档身份以 `pdblend4-v3-image-archive.json` 为准；归档本体不提交到 Git。
仓库根目录旧 `pdblend-l20-v1.tar.zst` 是 9 月 21 日的另一镜像，不能用于同一环境复现。

如果不能传输镜像，可在网络可用的主机上重建到独立 tag：

```bash
docker build -t pdblend:pdblend4-v3-rebuilt \
  --build-arg PIP_INDEX=https://pypi.org/simple \
  --build-arg APT_MIRROR=archive.ubuntu.com .
python3 scripts/2026-09-25_environment.py verify \
  --image pdblend:pdblend4-v3-rebuilt --allow-rebuilt
```

重建镜像不符合本版精确历史复现入口的全部系统层身份；采用新镜像运行需要重新绑定执行身份和验证。
按 RESTART 的镜像包导入则无需这条分支流程。
