# 一维杆结果

k_e = 1 / integral dx/(E A(x))；mean 用 E A(x_mid)/Le。

| case | eta | label | u_mid/um | CCX u_mid/um | rel mid/% |
|---|---|---|---|---|---|
| S0 | 0.00 | A | 0.12500 | 0.12438 | -0.496 |
| S1 | 0.20 | A | 0.10137 | 0.10142 | +0.053 |
| S2 | 0.10 | A | 0.14384 | 0.14343 | -0.287 |
| S3 | 0.40 | B | 0.05068 | 0.05128 | +1.178 |
| S4 | 0.60 | C | 0.05776 | 0.06023 | +4.271 |

## 自检
- reaction: 6.047e-13 / 1e-08
- force: 8.019e-13 / 1e-08
- equiv_node: 4.576e-13 / 1e-10
- energy: 3.380e-13 / 1e-10
- gauss_mean: 5.804e-16 / 1e-12
- beta0: 5.190e-13 / 1e-12

## 文件
- results/bar_runs.csv, bar_nodes.csv, bar_criterion.csv, bar_vs_ccx.csv
- 图/报告由 scripts/make_figures.py、scripts/make_report.py 生成
