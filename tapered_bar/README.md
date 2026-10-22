# 楔形变截面矩形拉杆

有限元理论课程大作业。研究对象是对称楔形变截面矩形拉杆：截面高度沿杆轴线性变化，两端受轴向拉力，按平面应力计算。

这个问题适合做方法对比。一维杆有闭式解，但固定端约束和楔形会让截面内应力不再均匀；网格加密能减小离散误差，却消不掉一维模型与二维模型之间的差异。因此除了比较位移和应力，还比较三种离散的产出形态，以及一维理论在什么锥度下仍然可信。

## 题目与模型

杆轴为 $x\in[0,L]$，截面高度 $h(x)=h_0(1+\beta x/L)$，关于 $y=0$ 对称；面外厚度 $b=0.1\,\mathrm{m}$，平面应力。取 $L=1.0\,\mathrm{m}$、$h_0=0.2\,\mathrm{m}$、$E=2.0\times10^{11}\,\mathrm{Pa}$、$\nu=0.3$、端部合力 $P=1000\,\mathrm{N}$。采用对称楔形而不是平底楔形，是为了让端部均布轴力的合力通过形心线，不引入附加整体弯曲。

五种工况覆盖从等截面到极端楔形：

| 工况 | beta | h0 / m | eta | 用途 |
|---|---|---|---|---|
| S0 | 0 | 0.2 | 0.00 | 等截面退化校核 |
| S1 | +1.0 | 0.2 | 0.20 | 主构型 |
| S2 | -0.5 | 0.2 | 0.10 | 收缩楔形 |
| S3 | +1.0 | 0.4 | 0.40 | 粗短强楔形，B 级边界 |
| S4 | +2.0 | 0.3 | 0.60 | 极端楔形，C 级 |

一维适用性判据取 $\eta=|\beta|h_0/L=\tan\alpha$：$\eta\le0.20$ 为 A 级，$0.20<\eta\le0.40$ 为 B 级，$\eta>0.40$ 为 C 级。阈值用二维参考标定，再用 CalculiX 做后验核对。

## 三种方法

1. **一维变截面杆单元**：用柔度积分定义等效刚度  
   $k_e=1/\int_{x_1}^{x_2}\mathrm{d}x/(EA(x))$，等价于对数平均面积刚度；另保留算术平均面积刚度作对照，说明“刚度积分”和“柔度积分”的差别。
2. **Q4 双线性等参元**：2×2 高斯全积分；后处理包括截面平均应力、探针点应力、沿轴应力剖面的梯度保真误差和剪切能比。
3. **CST 常应变三角形**：显式刚度 $k_e=t|A|B^{\mathsf T}DB$；面积加权节点应力恢复；用两种对角线剖分比较全局不变量与局部离散度。

第三方锚点用 CalculiX CPS4/CPS8，CPS8 240×48 作为参考；FEniCS/dolfinx 0.11 对 S0/S1 做独立校核。所有方法统一采用均布轴向拉力的一致节点载荷，截面平均位移统一用梯形（面积加权）平均。

## 结果概览

- Q4 在 160×32 网格上相对 CCX 参考：端/中截面位移最大偏差 0.025% / 0.014%，探针点应力 0.042%，应力梯度保真误差 0.35%–0.95%；位移观测阶约 1.7–2.5，应力约 1.6–2.3。
- CST 同网格：端/中位移 0.075% / 0.067%，探针应力 0.146%；两对角线剖分的全局量差约 $10^{-11}\%$，但局部单元应力离散度有 0.2%–1.1%，面积加权恢复能压低粗网格的截面内振荡。
- 一维杆：柔度积分刚度下节点位移与解析解一致到 $10^{-13}$；标签为 S0/S1/S2=A、S3=B、S4=C，与 CCX 的端部偏差依次约 +0.25%、−0.55%、+0.05%、−3.21%、−7.35%。
- 锚点：同网格 Q4 与 CPS4 端位移差不超过 0.049%；FEniCS 对 S1 的端位移与 CCX 差 +0.0029%。
- 产出形态：一维杆只给两条曲线，Q4 给连续云图，CST raw 给分块常应力场；报告第 3 节把三者放在同一工况、同一网格下并排比较。

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
