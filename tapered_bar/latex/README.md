# LaTeX 报告

## 有什么

- main.tex：导言区、字体、宏与章节入口。
- sections/：摘要、引言、理论、产出形态对比、实现、结果、讨论、结论、附录、参考文献。
- tables/：booktabs + siunitx 表格，由 scripts/make_latex_tables.py 生成。
- figures/：报告图，含三方法产出形态并排图。
- main.pdf：编译结果。

## 怎么跑

    cd latex
    xelatex main.tex
    xelatex main.tex

需要 ctex、xeCJK、booktabs、siunitx、caption、geometry、amsmath、graphicx、fancyhdr、hyperref 等 TeX 包。

## 结果在哪

- main.pdf：21 页 A4 报告。
- 数值来源：../results/study_runs.csv、study_profiles.csv、study_convergence.csv、ccx_reference.json。
- 表格生成：scripts/make_latex_tables.py；新图生成：scripts/make_latex_figures.py。
