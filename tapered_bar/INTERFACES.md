# 变截面矩形拉杆三方法研究 — 接口与交接约定

工作目录：`/home/john/桌面/有限元/tapered_bar`，所有命令在项目根执行。

## 1. 统一力学模型（已固定在 fem/config.py）

- 对称楔形：h(x)=h0(1+beta*x/L)，截面关于 y=0 对称（上下轮廓 y=±h/2），
  避免“平底楔形”端部载荷偏心引入的附加弯曲。
- 平面应力，面外厚度 b=width=0.1 m；左端 x=0 全约束 (ux=uy=0)，
  右端 x=L 均布轴向拉力（合力 P=1000 N），采用一致节点载荷
  （fem.assembly.end_load_vector 默认 mode="consistent"：线性边每段载荷一半到两端节点；
  旧等分模式保留为 mode="lumped" 仅作对比）。
- 材料 E=2.0e11 Pa，nu=0.3。
- 工况 CASES: S0(beta=0,h0=0.2 退化)、S1(beta=+1,h0=0.2 主构型)、
  S2(beta=-0.5,h0=0.2 收缩)、S3(beta=+1,h0=0.4 粗短)、
  S4(beta=+2,h0=0.3 极端)。几何定义见 fem/config.py。
- 网格 MESH_LEVELS = (10,2),(20,4),(40,8),(80,16),(160,32)；
  nx 必为 10 的倍数，因此每个网格都有 x=0,0.1,...,1.0 节点列。
- 一维名义解：fem/analytic.py（exact_u, exact_sigma, exact_sigma_gradient,
  exact_u_end, exact_u_mid, bar_equiv_stiffness, ...）。

## 2. 共享核心（Lead 已实现，勿改）

- `fem/config.py`：Geometry / Material / Case / CASES / MESH_LEVELS / STATIONS。
- `fem/analytic.py`：一维解析解、等效刚度。
- `fem/mesh.py`：`make_mesh(geom,nx,ny)`，节点编号 i*(ny+1)+j，
  `mesh.quads`，`mesh.tris_pos`，`mesh.tris_neg`，`mesh.right_nodes` 等。
- `fem/assembly.py`：`dofs_2d`、`assemble_sparse`、`dirichlet_solve`、
  `fixed_dofs_left`、`end_load_vector`。
- `fem/io.py`：`save_csv`、`save_json`、`load_csv`、`fnum`。

若发现核心接口不够用：在**自己的模块**里扩展，不要改共享文件；
如确需修改，先给 lead 发消息。

## 3. 各自文件所有权（严格不越界）

| 角色 | 独占文件 |
|---|---|
| bar-dev | fem/bar.py, scripts/run_bar.py, tests/test_bar.py, results/bar_* |
| q4-dev | fem/q4.py, scripts/run_q4.py, tests/test_q4.py, results/q4_* |
| cst-dev | fem/cst.py, scripts/run_cst.py, tests/test_cst.py, results/cst_* |
| ccx-dev | anchors/**, scripts/run_ccx.py, tests/test_ccx.py, results/ccx_* |
| lead | fem/config.py,analytic.py,mesh.py,assembly.py,io.py, scripts/run_study.py, figures/**, report/**, README.md |

共享目录 results/ 下按前缀区分，不要覆盖别人文件。

## 4. 结果 CSV 约定（列名固定；导出用 utf-8-sig）

### 4.1 bar-dev
- `results/bar_runs.csv`
  case,n_elem,method,u_end,u_mid,u_end_exact,u_mid_exact,e_end_pct,e_mid_pct,
  sigma_mid,sigma_mid_exact,e_sigma_pct
  method ∈ {equiv, mean, gauss}；equiv 为主方法。
- `results/bar_nodes.csv`
  case,n_elem,method,x,u_fem,u_exact,sigma_fem,sigma_exact
- `results/bar_criterion.csv`
  case,eta,angle_deg,slenderness,area_ratio,label,criterion,recommendation,u_mid_exact_um
  label ∈ {A,B,C}（A 可信、B 工程可用需 2D 复核、C 不可信），判据写清楚。
- `results/bar_summary.json`：数值摘要 + 判据阈值 + 结论。

### 4.2 q4-dev
- `results/q4_runs.csv`
  case,nx,ny,nnodes,ndof,u_end,u_mid,u_end_exact,u_mid_exact,e_end_pct,e_mid_pct,
  sigma_mid,sigma_mid_exact,e_sigma_mid_pct,sigma_profile_rms_pct,
  grad_profile_rms_pct,shear_energy_ratio,reaction_x,note
- `results/q4_profiles.csv`
  case,nx,ny,x,u_section_avg,sigma_section_avg,sigma_nom,dsigma_dx_fem,dsigma_dx_nom
  （x=0,0.1,...,1.0；sigma_section_avg 由 2×2 高斯点应力在截面列上平均）
- `results/q4_gradient.csv`
  case,nx,ny,x,dsigma_dx_fem,dsigma_dx_nom
- `results/q4_summary.json`

### 4.3 cst-dev
- `results/cst_runs.csv`
  case,nx,ny,diag,u_end,u_mid,e_end_pct,e_mid_pct,sigma_mid,e_sigma_mid_pct,
  sigma_profile_rms_pct,reaction_x,note
- `results/cst_profiles.csv`
  case,nx,ny,diag,x,u_section_avg,sigma_section_avg_raw,sigma_section_avg_recovered,sigma_nom
  raw=单元常应力直接按单元平均；recovered=面积加权节点恢复后再按列平均。
- `results/cst_diagonal.csv`
  case,nx,ny,u_end_pos,u_end_neg,u_mid_pos,u_mid_neg,d_u_pct,
  sigma_end_pos,sigma_end_neg,d_sigma_axial_pct,max_nodal_recovered_diff_pct
  或按你们定义补充列（在 JSON 里说明公式）。
- `results/cst_summary.json`

### 4.4 ccx-dev
- `results/ccx_anchor.csv`
  case,mesh,element_type,nnodes,nelem,u_end,u_mid,sigma_mid,sigma_mid_exact,
  e_end_pct,e_mid_pct,e_sigma_pct,note
- `results/ccx_profiles.csv`
  case,mesh,element_type,x,u_section_avg,sigma_section_avg,sigma_nom
- `results/ccx_reference.json`：每个工况收敛参考 {u_end,u_mid,profile:[...],mesh,
  element_type,convergence_check}。
- `results/ccx_vs_inhouse.csv`（可选，若 in-house 结果已存在）：case,mesh,
  u_end_ccx,u_end_inhouse,rel_diff_pct,sigma_mid_ccx,sigma_mid_inhouse,rel_diff_pct。

## 5. 数值与后处理口径（统一，避免对比歧义）

- 端部位移 u_end：右端面 ux 的面积加权（梯形权重，按实际 y 排序）平均。
- 中截面位移 u_mid：x=L/2 节点列 ux 的面积加权（梯形权重）平均。
  （不要用节点算术平均：含边中点/端点权重不对，会带 O(1/ny) 偏差；共享实现见 fem/post.py。）
- 截面平均轴向应力 sigma_section_avg(x)：该 x 列所有单元（或高斯点）sigma_x 的
  等权平均（列内单元面积相等，等价面积加权）。
- 剖面误差 sigma_profile_rms_pct：在公共站 x=0.1..0.9 上与一维名义
  sigma_nom=P/A(x) 的 RMS 相对误差，`100*sqrt(mean((s-s_nom)^2))/rms(s_nom)`。
  与 CCX 参考的误差由 lead 统一补算；你们先给名义误差。
- 梯度保真误差 grad_profile_rms_pct：对截面平均应力曲线用二阶中心差分求
  dsigma/dx，在 x=0.1..0.9 与一维名义梯度比较的 RMS 相对误差。
  （若用单元局部最小二乘梯度，另存列并在 JSON 说明。）
- 收敛曲线最终由 lead 用“与 CCX CPS8 细网格参考值”的误差统一绘制；
  你们的 CSV 保留原始量即可。

## 6. 质量要求（每人都要自测）

1. 解析/极限校核：beta=0 时一维解与 P L/(E A) 一致；等效刚度极限正确。
2. 刚体位移/零能模式检查（2D）：给全部节点同一平移，应变/应力应为 0。
3. 常应变分片试验（Q4/CST）：线性位移场下单元应力应为常数且等于解析值。
4. 反力校核：固定端 x 反力合力 = -P（相对误差 <1e-8）。
5. 网格收敛：加细后 u_end/u_mid 单调趋于稳定；报告观测阶。
6. 所有脚本可在项目根用 `python3 scripts/xxx.py` 复现；结果落盘。

## 7. 汇报格式（send_message 给 lead）

简短列出：完成项、关键数值（表格 3-5 行）、产出文件路径、遇到的问题/风险、
是否建议 lead 调整统一口径。不要贴大段代码/日志。
