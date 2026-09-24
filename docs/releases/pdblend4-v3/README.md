# pdblend4-v3 发布验收

本分支提交 9 月 25 日的维护源码及复现工具；操作入口为 [RESTART](../../../RESTART.md)。
它包含既有容量恢复、SLO 路由、预算 Shield、profile、计量和比较优化，补齐了只存在于结果目录的
trace equivalence 模块和分析脚本，并移除了 baseline 的 PDblend 在线状态校验依赖。
当前机器已接受的 d4fe GPU 源码仍作为历史模式保留。最新维护源码的冻结 SHA 是
`63a44934e1e573e079b453b9adbb9af93aee739a799be6ae15abdb05bbfddc0c`。

- 完整 CPU 回归：3472 passed、48 skipped、28 deselected、0 failed；JUnit 见 `cpu-tests.xml`。
- 复现入口与归档专项：20 passed；针对的是全量测试启动后完成的最终工具版本。
- 三模型 50 个文件重新读取并通过 SHA256，约 104 GiB；固定 manifest 已纳入分支。
- 真实当前镜像完成导出、归档校验、docker load 和复核；21 层、完整配置、167 个包、2 个补丁一致。
- 在隔离目录提取 4237 个项目输入和 1 个外部文件，容器无法访问源机原始 results，禁网且未分配 GPU。
  使用模拟新服务器 UUID，在准确镜像内通过 PDblend 36 点和 baseline 144 点的 CPU 输入/启动契约检查。
  基线使用各自原冻结实现，共 20 个 session。详见 `cleanroom-runtime.json`。
- 202 个既有历史文件删除有已执行的清理索引；没有重新删除当前运行原始数据。具体审计在 `source-audit.json`。

48 项跳过包括宿主缺失的可选依赖、仅适用于锁定 vLLM 镜像的 CPU contract 以及 collection skip；
28 项由测试 marker 排除。CPU、模型完整性与镜像迁移验证不能当成新硬件的 GPU、功耗或性能验收。
本次没有中断源机实验，没有运行新的 GPU 实验，也没有连接另一台 8×L20 服务器。
宿主完整依赖未在全新 venv 重新下载/安装；81 个固定版本已核验官方索引存在，安装器最终执行 pip check。

交付目录 `/home/pdblend4-v3-release` 包含分支 bundle、精确镜像、runtime 输入包和完整本次验证日志。
Qwen 权重独立从 `/home/models` 迁移。runtime 包不含所有历史原始测量，旧 profile 只以明确 development
资格在新机重跑；新服务器的正式校准和 GPU 验收仍按 RESTART 执行。
