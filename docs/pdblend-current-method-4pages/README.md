# PDblend 当前方法：双栏四页

本文按 2026-09-24 当前工作树分析，三个子代理分别核对 profiler、
在线调度和自适应池配置，再合并并复审。分析包含已有未提交代码，
不将旧文档中的目标设计直接当作当前实现。

正文为中文，A4 双栏、11 pt，PDF 共 4 页，含 2 段算法及 7 个编号公式。
结构：

1. Profiler：测量、建模、能耗口径及有效覆盖域。
2. 在线算法：有序 P/D 实例配对、SLO 路由、KV 交接与回退。
3. Adaptive Pool Configuration：Frequency Control、Role Change、TP Weight Change。

其中 TP weight change 指常驻异构 TP 池的请求流量权重调整；
模型权重在线重分片仍未开放。正文区分普通 Router 的端点交接预算与
池级/resident 路径的近似，也区分活跃角色调整与停车排空。

## 文件

- pdblend-current-method.pdf：最终 PDF。
- pdblend-method.tex：展开各节后的单文件 LaTeX，可独立编译。
- main.tex、sections/01-profiler.tex、sections/02-online.tex、
  sections/03-adaptive.tex：便于逐节修改的来源文件。
- pdblend-method-source.zip：可编译源码包，附核对笔记和来源清单。
- notes/*.md：逐部分源码依据与行号，不计入四页正文。
- source-manifest.json：本次分析涉及的源码 SHA256；不是服务进程版本证明。
- validation.json、preview/：PDF 页数、排版检查与预览。

## 编译

需要 XeLaTeX、latexmk、ctex/Fandol、amsmath、algorithm/algpseudocode、
flushend、microtype 和 hyperref。使用：

    bash build.sh

也可直接编译单文件两次：

    xelatex -interaction=nonstopmode -halt-on-error pdblend-method.tex
    xelatex -interaction=nonstopmode -halt-on-error pdblend-method.tex

修改分节版后，运行 python3 package.py 更新单文件和源码包。
verify.py 需要 PyMuPDF 与 Pillow，用于检查最终 PDF 并生成预览：

    python3 verify.py

本次只生成文档，未改动服务代码、未运行 GPU 实验。文中的可配置默认值与实现机制
不是新增性能或能耗实验结论。各代理的 *-draft.tex 中间稿不参与编译或源码包。
