# PDblend 方法与请求时序图

按照用户提供的 runtime/frontend timing 图制作：白底、衬线字体、黑色时间轴，绿色 Prefill、蓝色 Decode、黄色输出 token、灰色 KV 交接／等待。

最终图片：[pdblend-method-timing.png](pdblend-method-timing.png)，1536 × 1024 PNG。生成方式：内置 imagegen；保留的完整提示词顺序如下：

1. [初次生成](prompt.txt)
2. [连线与时序修订](refinement-prompt.txt)
3. [局部修订](final-correction-prompt.txt)
4. [token 标签修订](label-correction-prompt.txt)
5. [最终单标签修订](single-label-correction-prompt.txt)

图包含成本建模与联合规划、选择性分流、两条请求执行时间轴、在线保护与反馈。采用方法文档中的示意配置 P1 + D2 + M4 + L1，阈值 1024，输入长度 256 / 2048，输出预算 128。

长请求的 P 产生首 token y₁；代理向客户端发送 y₁ 后，以 input ⊕ y₁ 和 127-token 剩余预算启动 D，复用原始 2048 个位置的 KV。TTFT 截止于客户端收到 y₁；y₁ 到 y₂ 的续接间隙包含交接和续接工作，不能等同于纯 KV 拷贝耗时。图中只显示部分 token，中间使用省略号。

内容依据：

- `../pdblend-method-latex/sections/01-overview.tex`
- `../pdblend-method-latex/sections/02-model.tex`
- `../pdblend-method-latex/sections/03-planning.tex`
- `../pdblend-method-latex/sections/04-online.tex`
- `../../src/pdblend/control/planner.py`
- `../../src/pdblend/proxy/router.py`
- `../../src/pdblend/engine/carry.py`

这是方法设计示意图，条块长度不代表测量时长，配置不代表某次最优规划结果。完整增量能耗核算与就绪协调等设计内容不应理解为当前实现已全部验收。当前 carry 协议的实现范围以代码为准：token-ID 输入、greedy、固定输出预算、同 TP 且 PP1。

图片是栅格输出；提示词可用于继续修改，原有方法文档和实验结果没有改动。
