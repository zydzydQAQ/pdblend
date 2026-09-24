# pdblend4-v3 迁移范围

新 8×L20 的完整执行顺序见 [RESTART.md](RESTART.md)。该手册是本分支的部署入口；
旧技能中 CUDA 12.8 / vLLM 0.9.2 / V0 八文件补丁的命令不适用于这里。

## 源码与大文件分别交付

- Git 分支 `pdblend4-v3`：当前维护源码、独立 baseline、测试、日期脚本、Dockerfile、引擎补丁、
  依赖锁、镜像/模型/输入身份清单、文档和历史删除索引。可以从源码机导出的 Git bundle 克隆。
- `/home/pdblend4-v3-release/image/`：当前 Docker 的真实导出包和 SHA 收据。不是根目录旧镜像包。
- `/home/pdblend4-v3-release/inputs/`：被复现入口实际读取的不可变文件闭包；含旧 profile、三模型轨迹、
  配置、冻结源码以及 baseline predictor。不会打包活动 queue、lease 或 stop 文件。
- `/home/pdblend4-v3-release/external/`：baseline 真实到达记录的外部依赖，恢复到 `/home/pdblend` 下原路径。
- `/home/models/`：Qwen 三模型、BERT 与相关清单，单独 rsync；目标机全量验证模型 SHA。
- `/home/pdblend4-v3-release/validation/`：本次实际检查记录；不能代替目标机硬件与性能验收。

主项目固定恢复到 `/home/pdblend4`，权重到 `/home/models`。这是现有 immutable artifact 的绝对路径
约束。输入提取器拒绝不同根目录、符号链接和不同内容的覆盖；不改写历史 manifest 或伪造新校准。
将项目迁往任意新根目录需要单独实现可验证的引用映射，本版不声称已支持。

## 固定的真实环境

镜像的 CUDA base 是 12.8.1，实际引擎 Torch 是 2.7.1+cu126（CUDA 12.6），vLLM 是 0.10.1.1，
使用 V1。宿主控制器 Torch 是 2.7.0+cu128，Transformers 是 4.51.3；两者分别锁定。
`requirements/pdblend4-v3-image.json` 绑定镜像层/配置、依赖和补丁；归档收据另绑定导出 tar 的 SHA。
镜像 save/load 保持系统层相同，联网重建只保证已锁定的源码/依赖输入，不承诺逐层相同。

宿主需可用驱动、Docker、NVIDIA Container Toolkit、Python 3.10 和实验锁频权限。
bootstrap 新建项目 `.venv`，不复制旧虚拟环境。宿主完整 wheelhouse 未包含在交付中，安装需要联网。
目标机环境验证包含八卡真实 BF16 CUDA 运算、Python 包、补丁和 pip 依赖检查。

## 新机实验与历史证据

复现入口默认冻结当前分支源码，采用新的 GPU UUID、queue、attempt 和执行源码 SHA；
历史模式则使用每个系统原来绑定的冻结源码。保持原请求、seed701、150秒窗口、SLO及完整尾部，
所有测量写入新输出，不接管源机仍运行的队列。

旧 profile 以 development 方式复用，明确 `target_machine_calibrated=false` 与 `formal_eligible=false`。
模型相同和 GPU 型号相同不足以继承 profile、锁频、能耗或完整系统资格。新机正式结论需要自己的
校准、模型服务/KV/恢复检查、计量资格和独立实验。准备/CPU preflight 成功也不是 GPU 实验完成。

runtime 输入包不含全部 44 GiB 原始历史证据。完整历史复算仍需源机相应原件；不能把缺原始测量的
包写成 full audit archive。既有失败 attempt 和已删除原始数据的记录原样保留，不补造或追认指标。
旧 `results/v2` 清理保留于 `results/maintenance/2026-09-23-history-pruning-v6` 的逐文件身份索引；
本次提交记录已有删除，不重新删除正在使用的原始实验数据。

旧文档保留于 `results/archive/docs-2026-09-23/`，仅作为历史说明。
