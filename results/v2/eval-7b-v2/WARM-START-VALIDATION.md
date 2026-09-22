# sharegpt-x0.5 PDblend 暖启动复核

本点使用迁移前同一 spec、语料、profile 和 8 卡配置重跑。旧冷启动原始结果保存在 `results/archive/2026-09-21-migration-preclean/sharegpt-x0.5-pdblend-cold-head.tar.gz`；本目录保留新的暖启动结果。

| 版本 | mean power | J/token | TTFT p90 | controller events (plan / forecast / reroute / park / wake) |
|---|---:|---:|---:|---:|
| 冷启动（归档） | 1336.95 W | 0.532003 | 0.9906 s | 13 / 29 / 34 / 9 / 5 |
| 暖启动（当前） | 1279.06 W | 0.509712 | 0.9753 s | 15 / 29 / 42 / 9 / 5 |

当前暖启动相对旧冷启动降低约 4.19% 功耗和 4.19% J/token。与 `sharegpt-x0.5-ecoserve` 的 0.5078 J/token 相比，仍为 **+0.37%，LOSE**；joint SLO rate 为 1.0。该结果说明修复有效，但尚未超过最佳 baseline。

恢复验证：原 `sharegpt-x0.3-mixed` 的引擎退出日志已归档，按原 spec 重跑后成功，joint SLO rate=1.0、mean power=2032.68 W；原 `sharegpt-x0.4-mixed` 的残点也已归档，当前矩阵正在按原 spec 重跑该点。
