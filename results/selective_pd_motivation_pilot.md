# Selective PD 本机 motivation 预实验

## 结论

- 相位级 DVFS 确实有收益：A100 decode 从 1410 降到 930 MHz，GPU J/request 降 33.2%，p90 TBT 仅增加 0.9%；V100 的 1380→900 MHz 对应节能 18.7%、p90 TBT 增加 8.7%。
- 但本机 A100/V100 不支持 CUDA P2P。NIXL 被强制走 UCX host-staged/TCP 后，PD p90 TTFT 为 1634–17945 ms，而等资源 colocated 为 69–1059 ms。
- 不考虑 SLO 时，有 14 个 PD 配置的 GPU J/request 低于对应 colocated；加入 90% TTFT+TBT 双 SLO 后，可行 PD 配置为 0 个，所有可行单元均选择 colocated。这正说明能耗目标不能脱离传输拓扑和 SLO 做全分离。
- 当前设备支持的结论是“弱互连/无 P2P 区域应选择不分离”，不是“PD 在任何硬件上都无效”。要观察真正 crossover，需要同代、P2P 可用或 NVLink/RDMA 的第二组硬件。

## 计量口径

- 主模型：Qwen2.5-7B-Instruct，FP16，vLLM 0.11.2；显式关闭 prefix caching，避免重复 trace 命中 KV 缓存。
- PD 传输：NIXL 1.4.0，UCX `self,sm,tcp,cuda_copy`，用于绕过本机双向 CUDA P2P=`False`。
- 本报告只积分两张 GPU 的 NVML 功率；主机 CPU、DRAM 与 PCIe 交换芯片能量不在计量范围内，因此对 host-staged PD 的能量估计偏乐观。

## 数据完整性

- 原始有效运行：84
- 已发现 suite：crossover_a100_prefill, crossover_colocated, crossover_v100_prefill, phase_a100, phase_v100
- 缺失 suite：无

## 相位频率敏感性

| GPU / case | p90 TTFT (ms) | p90 TBT (ms) | J/request | J/output token |
|---|---:|---:|---:|---:|
| phase_a100 / decode_f1410 | 39.91 | 15.55 | 1101.87 | 4.3042 |
| phase_a100 / decode_f615 | 41.68 | 19.60 | 735.26 | 2.8721 |
| phase_a100 / decode_f930 | 39.81 | 15.69 | 736.18 | 2.8757 |
| phase_a100 / prefill_f1410 | 161.66 | n/a | 80.39 | 80.3934 |
| phase_a100 / prefill_f615 | 271.33 | n/a | 70.17 | 70.1671 |
| phase_a100 / prefill_f930 | 185.97 | n/a | 66.28 | 66.2790 |
| phase_v100 / decode_f1380 | 56.20 | 23.59 | 1478.70 | 5.7762 |
| phase_v100 / decode_f600 | 80.99 | 52.69 | 1248.15 | 4.8756 |
| phase_v100 / decode_f900 | 57.58 | 25.63 | 1201.77 | 4.6944 |
| phase_v100 / prefill_f1380 | 710.81 | n/a | 165.64 | 165.6409 |
| phase_v100 / prefill_f600 | 1800.78 | n/a | 138.21 | 138.2129 |
| phase_v100 / prefill_f900 | 1062.49 | n/a | 125.68 | 125.6840 |

## 等资源架构 crossover

| Architecture | Case | p90 TTFT (ms) | p90 TBT (ms) | J/request | 95% CI |
|---|---|---:|---:|---:|---:|
| PD(A100-P,V100-D) | balanced_low_max | 8764.88 | 27.21 | 1136.02 | [1131.78, 1140.25] |
| PD(A100-P,V100-D) | balanced_medium_max | 6180.21 | 28.35 | 843.72 | [772.94, 914.51] |
| PD(A100-P,V100-D) | decode_low_max | 9874.07 | 25.81 | 2372.64 | [2325.90, 2419.39] |
| PD(A100-P,V100-D) | decode_medium_eco | 1634.41 | 57.22 | 1637.34 | [1632.21, 1642.47] |
| PD(A100-P,V100-D) | decode_medium_max | 5371.48 | 26.31 | 1110.54 | [1107.62, 1113.46] |
| PD(A100-P,V100-D) | prefill_low_max | 12338.05 | 28.08 | 892.11 | [813.37, 970.85] |
| PD(A100-P,V100-D) | prefill_medium_eco | 11401.68 | 72.05 | 527.12 | [512.54, 541.70] |
| PD(A100-P,V100-D) | prefill_medium_max | 10318.26 | 31.09 | 519.54 | [498.96, 540.12] |
| PD(A100-P,V100-D) | short_low_max | 7012.38 | 25.46 | 543.25 | [540.43, 546.06] |
| PD(A100-P,V100-D) | short_medium_max | 17945.10 | 26.52 | 328.82 | [302.62, 355.02] |
| colocated | balanced_low_max | 289.76 | 24.05 | 1229.16 | [1219.71, 1238.60] |
| colocated | balanced_medium_max | 279.02 | 24.04 | 933.28 | [923.27, 943.28] |
| colocated | decode_low_max | 75.36 | 23.67 | 2666.34 | [2624.11, 2708.56] |
| colocated | decode_medium_eco | 77.48 | 26.06 | 1599.51 | [1547.19, 1651.83] |
| colocated | decode_medium_max | 74.12 | 25.12 | 1508.81 | [1479.91, 1537.71] |
| colocated | prefill_low_max | 709.68 | 24.60 | 872.25 | [862.99, 881.51] |
| colocated | prefill_medium_eco | 1058.87 | 27.27 | 462.83 | [457.14, 468.53] |
| colocated | prefill_medium_max | 710.33 | 24.57 | 586.13 | [560.56, 611.69] |
| colocated | short_low_max | 68.80 | 23.49 | 656.71 | [650.59, 662.82] |
| colocated | short_medium_max | 236.84 | 24.19 | 399.05 | [389.28, 408.82] |
| PD(V100-P,A100-D) | balanced_low_max | 17246.88 | 15.81 | 1169.52 | [1087.46, 1251.58] |
| PD(V100-P,A100-D) | balanced_medium_max | 11778.67 | 15.73 | 647.79 | [604.27, 691.31] |
| PD(V100-P,A100-D) | decode_low_max | 9109.67 | 15.28 | 2139.24 | [2094.48, 2183.99] |
| PD(V100-P,A100-D) | decode_medium_eco | 11172.71 | 21.57 | 1184.00 | [1176.09, 1191.91] |
| PD(V100-P,A100-D) | decode_medium_max | 10960.81 | 15.87 | 925.68 | [877.16, 974.20] |
| PD(V100-P,A100-D) | prefill_low_max | 14948.50 | 15.28 | 921.74 | [767.87, 1075.62] |
| PD(V100-P,A100-D) | prefill_medium_eco | 11191.72 | 22.51 | 479.38 | [433.37, 525.39] |
| PD(V100-P,A100-D) | prefill_medium_max | 11437.38 | 16.29 | 633.82 | [583.55, 684.10] |
| PD(V100-P,A100-D) | short_low_max | 16410.89 | 15.77 | 607.73 | [551.57, 663.88] |
| PD(V100-P,A100-D) | short_medium_max | 14902.48 | 15.98 | 261.80 | [214.35, 309.25] |

## 90% 双 SLO 下的 oracle epoch-selective 选择

| Workload | Load | SLO | TTFT/TBT threshold (ms) | Winner | J/request |
|---|---|---|---:|---|---:|
| balanced | low | tight | 313.04 / 26.46 | colocated (balanced_low_max) | 1229.16 |
| balanced | medium | tight | 304.54 / 26.45 | colocated (balanced_medium_max) | 933.28 |
| decode | low | tight | 86.69 / 26.04 | colocated (decode_low_max) | 2666.34 |
| decode | medium | tight | 81.73 / 27.63 | colocated (decode_medium_max) | 1508.81 |
| prefill | low | tight | 780.80 / 27.06 | colocated (prefill_low_max) | 872.25 |
| prefill | medium | tight | 781.33 / 27.02 | colocated (prefill_medium_max) | 586.13 |
| short | low | tight | 74.76 / 25.84 | colocated (short_low_max) | 656.71 |
| short | medium | tight | 79.74 / 27.29 | colocated (short_medium_max) | 399.05 |
| balanced | low | loose | 426.87 / 36.08 | colocated (balanced_low_max) | 1229.16 |
| balanced | medium | loose | 415.28 / 36.06 | colocated (balanced_medium_max) | 933.28 |
| decode | low | loose | 118.22 / 35.51 | colocated (decode_low_max) | 2666.34 |
| decode | medium | loose | 111.45 / 37.68 | colocated (decode_medium_max) | 1508.81 |
| prefill | low | loose | 1064.73 / 36.91 | colocated (prefill_low_max) | 872.25 |
| prefill | medium | loose | 1065.45 / 36.85 | colocated (prefill_medium_eco) | 462.83 |
| short | low | loose | 101.94 / 35.23 | colocated (short_low_max) | 656.71 |
| short | medium | loose | 108.74 / 37.21 | colocated (short_medium_max) | 399.05 |

## 解释边界

- Colocated 与 PD 始终使用同一组 A100+V100，PD 还交换两次卡的角色；结果不能归因于单一 GPU 型号。
- Gross energy 从首个请求调度到最后一个 token，包含实验窗口中的空闲池功耗，但不含模型加载。
- P2pNcclConnector 的失败由硬件能力核验复现：`nvidia-smi topo -p2p` 为 NS，CUDA peer access 双向为 False；最终 PD 数据均来自获批的 NIXL host-staged fallback。
- 两次重复的 bootstrap 区间只能反映 pilot 抖动；论文级结果应至少五次重复并扩大请求数。
- 两卡只能给出 workload epoch 级 oracle 选择，无法实现三池同时常驻的请求级 hybrid。

## 下一步

本机下一步应把 host-staged 结果作为“不可分离区”写入拓扑判据；若获得 P2P 可用的第二张同代 GPU，再扩到 25%/50%/75% 负载、三组 SLO、至少五次重复。按本次 pilot 墙钟时间估算，完整矩阵约需 8–12 小时。
