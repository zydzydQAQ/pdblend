# PDblend 双语方法说明

`pdblend-method-bilingual.pdf` 是 12 页最终文档，包含五张方法图、两段主算法及一个请求执行辅助过程。正文以理想 selective PD 设计为主，各技术章末注明与当前代码的差异。说明性数字不代表实测结果。

下列可编辑源文件、字体与许可直接保存在本目录。重复的源文件 ZIP 和逐页预览已清理；运行构建脚本可按需重新生成 `pdblend-method-source.zip`。PDF 已检查逐页排版、文字抽取与四章书签；不存在额外 GPU 实验。

## 编辑与重建

- `chapters/`：四章可编辑 HTML；固定共 12 个页面。
- `style.css`：A4 排版、双语字体、表格和算法样式。
- `build_figures.py` / `figures/`：五张可编辑 SVG 方法图及其生成脚本。
- `build_pdf.py`：合稿、生成 PDF、书签、预览与结构验证。
- `evidence.json`：取证时间、代码版本、工作区变更、关键文件 SHA-256 与设计边界。
- `validation.json`：保留的页数、文字抽取和页面边界检查；`preview/` 为构建时按需生成的逐页预览。

在独立 Python 环境安装 `requirements.txt` 后，运行：

```bash
python build_pdf.py
```

若系统缺少 venv，依赖也可装入独立目录，不修改服务环境：

```bash
python3 -m pip install --target /tmp/pdblend-document-packages -r requirements.txt
PYTHONPATH=/tmp/pdblend-document-packages python3 build_pdf.py
```

WeasyPrint 需要系统的 Pango / HarfBuzz / Fontconfig 库。字体来自 Noto CJK，保存在 `assets/`，许可见 `assets/FONT-LICENSE.txt`。构建仅使用 CPU，不操作 GPU、服务或实验队列。

## 证据定位

离线建模对应 `profile/model.py`、`profile/profiler.py` 和 `profile/calibration.py`；规划对应 `control/planner.py`、`control/forecast.py` 和策略表；在线机制对应 `control/controller.py`、`control/shield.py`、`proxy/router.py`、`proxy/server.py` 和 `engine/carry.py`。完整仓库路径与校验值记录于 `evidence.json`。

PDF 中的成本 100 / 90 / 103、功率 1200 W 和 10 s、请求长度与阈值都是明确标注的机制算例。未进行新的 GPU 测量，未声称理想设计已取得已验证的性能收益。
