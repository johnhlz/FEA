# FEniCS 校核

## 有什么

- check.py：平面应力本构/位移校核，P2 三角形，80x16。
- fenics_check.json：S0/S1 的 u_end、sigma_mid 与对照。

## 怎么跑

    cd tapered_bar
    /home/john/miniconda3/envs/fem-fenics/bin/python anchors/fenics/check.py

环境：Python 3.11、dolfinx 0.11、ufl 2026.1、scipy。系统 python3 没有 dolfinx。

## 结果在哪

- fenics_check.json。
- S0 u_end=2.493808e-07 m（一维 2.5e-07，端部约束效应）；sigma_mid=50000.00 Pa。
- S1 u_end=1.742433e-07 m，与 CalculiX CPS8 240x48 差 +0.0029%。

主流程的 CalculiX 锚点在 anchors/calculix/ 与 results/ccx_*。
