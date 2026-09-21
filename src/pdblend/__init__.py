# -*- coding: utf-8 -*-
"""PDblend:同负载 min gross J,且 SLO 不低于 mixed@2520。

方法是三层在线控制面:启动满配 mixed,只看已到达请求与实例状态。
L0 滚动 prefill 角色并钉 2520;L1 突发扩环/满频;L2 park,拓扑重启另测。
无 inherit 表。时间分离不是严格 PaDG。A9 hybrid 不是默认。
"""

__version__ = "0.1.0"
