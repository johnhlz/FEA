# 楔形变截面矩形拉杆

## 有什么

- fem/：网格、组装、一维杆、Q4、CST，以及报告用的后处理。
- scripts/：各方法驱动、统一对比、绘图、LaTeX 表格与 PDF 生成。
- anchors/calculix/：CalculiX 输入生成、运行、.dat/.frd 解析。
- anchors/fenics/：FEniCS/dolfinx 可选校核。
- results/：CSV/JSON 结果，study_* 为统一口径，ccx_* 为第三方参考。
- figures/：报告图；latex/：LaTeX 源文件与 PDF。
- tests/：单元、锚点与交叉校核。

## 怎么跑

    cd tapered_bar
    python3 -m pytest -q tests
    python3 scripts/run_bar.py
    python3 scripts/run_q4.py
    python3 scripts/run_cst.py
    python3 scripts/run_ccx.py
    python3 scripts/run_study.py
    python3 scripts/make_figures.py
    python3 scripts/make_latex_figures.py
    python3 scripts/make_latex_tables.py
    cd latex && xelatex main.tex && xelatex main.tex

依赖：Python 3.13、NumPy、SciPy、matplotlib；CalculiX 2.23（环境 fem-ccx）；
FEniCS/dolfinx 0.11（环境 fem-fenics，可选）；LaTeX 用 TeX Live + XeLaTeX。

## 结果在哪

- 端/中位移、应力误差、收敛阶：results/study_runs.csv、study_convergence.csv
- 应力/位移剖面、探针点：results/study_profiles.csv、study_probes.csv
- CalculiX 参考与锚点：results/ccx_reference.json、ccx_anchor.csv、ccx_note.md
- 各方法原始输出：results/bar_*、q4_*、cst_*
- 图：figures/
- 报告：latex/main.pdf；旧版 Markdown/PDF 在 report/
