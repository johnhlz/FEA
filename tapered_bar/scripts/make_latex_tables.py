"""LaTeX table fragments."""
import csv, json, os
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
OUT = os.path.join(PROJ, "latex", "tables")
os.makedirs(OUT, exist_ok=True)

def load_csv(name):
    with open(os.path.join(RES, name), "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def load_json(name):
    with open(os.path.join(RES, name), "r", encoding="utf-8") as f:
        return json.load(f)

RUNS = load_csv("study_runs.csv")
FITS = load_csv("study_convergence.csv")
BAR = load_csv("bar_criterion.csv")
CCX = load_json("ccx_reference.json")
TABLES = load_json("study_tables.json")
CCX_ANCHOR = load_csv("ccx_anchor.csv")
PROFILES = load_csv("study_profiles.csv")
PROBES = load_csv("study_probes.csv")
FEN = json.load(open(os.path.join(PROJ, "anchors", "fenics", "fenics_check.json"), encoding="utf-8"))

CASES = ["S0", "S1", "S2", "S3", "S4"]
CASE_INFO = TABLES["cases"]

def run(case, method, nx=160, diag=None):
    for r in RUNS:
        if r["case"] != case or r["method"] != method:
            continue
        if str(int(float(r["nx"]))) != str(nx):
            continue
        if diag is not None and r["diag"] != diag:
            continue
        return r
    return {}

def fit(case, method, metric):
    for r in FITS:
        if r["case"] == case and r["method"] == method and r["metric"] == metric:
            return float(r["slope"])
    return float("nan")

def n(x, nd=3, plus=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "{}"
    s = f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"
    return s

def sci(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "{}"
    return f"{x:.{nd}e}"

def write(name, body):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    print("wrote", path)

# T1 parameters
write("tab_params.tex", r"""\begin{tabular}{@{}llS[table-format=1.1e-2]l@{}}
\toprule
参数 & 符号 & {取值} & 单位 \\
\midrule
杆长 & $L$ & 1.0 & m \\
固定端高度 & $h_0$ & 0.2 & m \\
面外厚度 & $b$ & 0.1 & m \\
弹性模量 & $E$ & 2.0e11 & Pa \\
泊松比 & $\nu$ & 0.3 & -- \\
端部总拉力 & $P$ & 1000 & N \\
平面应力厚度 & $t=b$ & 0.1 & m \\
\bottomrule
\end{tabular}
""")

# T2 cases
rows = []
for c in CASES:
    e = CASE_INFO[c]
    rows.append(f"{c} & {e['beta']:+.2f} & {e['h0']:.2f} & {e['h1']:.2f} & "
                f"{e['eta']:.2f} & {e['taper_angle_deg']:.1f} & {e['h1']/e['h0']:.2f} \\\\")
write("tab_cases.tex", r"""\begin{tabular}{@{}lS[table-format=+1.2]S[table-format=1.2]S[table-format=1.2]S[table-format=1.2]S[table-format=2.1]S[table-format=1.2]@{}}
\toprule
工况 & {$\beta$} & {$h_0$/m} & {$h_1$/m} & {$\eta=|\beta|h_0/L$} & {锥角/$^\circ$} & {$A_1/A_0$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# T3 output forms (static, text)
write("tab_forms.tex", r"""\begin{tabular}{@{}p{0.17\textwidth}p{0.25\textwidth}p{0.25\textwidth}p{0.25\textwidth}@{}}
\toprule
维度 & 一维杆单元 & Q4 等参元 & CST 三角形 \\
\midrule
每节点自由度 & 1（$u_x$） & 2（$u_x,u_y$） & 2（$u_x,u_y$） \\
单元未知量 & 2 & 8 & 6 \\
单元刚度 & $2\times2$，闭式 & $8\times8$，$2\times2$ 高斯积分 & $6\times6$，显式 \\
应力表示 & 单元常数（按平衡恢复） & 单元内双线性 & 单元常数（常应变） \\
主要产出 & $u(x)$、$\sigma(x)$ 两条一维曲线 & 连续 $\sigma_x$ 云图、任意点应力 & 分块 $\sigma_x$ 云图、节点恢复场 \\
信息量/代价 & 信息最少、自由度最省 & 场连续、代价中等 & 单元数约为 Q4 两倍，局部应力不连续 \\
后处理 & 截面平均、1D 判据 & 高斯点应力、梯度保真、剪切能比 & 面积加权恢复、对角线离散 \\
典型适用 & 杆系快速估算、A/B 级工况 & 光滑二维场、端部复核 & 复杂几何、快速原型 \\
\bottomrule
\end{tabular}
""")

# T4/T5 displacement
def disp_table(component):
    rows = []
    for c in CASES:
        bar = run(c, "BAR-equiv")
        q4 = run(c, "Q4")
        cst = run(c, "CST", diag="pos")
        e = CASE_INFO[c]
        vals = [e["u_end_1d"] if component == "e" else e["u_mid_1d"],
                float(bar["u_end"] if component == "e" else bar["u_mid"]),
                float(q4["u_end"] if component == "e" else q4["u_mid"]),
                float(cst["u_end"] if component == "e" else cst["u_mid"]),
                e["u_end_ref"] if component == "e" else e["u_mid_ref"]]
        rows.append(c + " & " + " & ".join(f"{v*1e6:.6f}" for v in vals) + r" \\")
    return r"""\begin{tabular}{@{}lS[table-format=1.6]S[table-format=1.6]S[table-format=1.6]S[table-format=1.6]S[table-format=1.6]@{}}
\toprule
工况 & {1D} & {杆} & {Q4} & {CST} & {CCX CPS8} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
"""
write("tab_disp_end.tex", disp_table("e"))
write("tab_disp_mid.tex", disp_table("m"))

# T6 deviations
rows = []
for c in CASES:
    bar = run(c, "BAR-equiv"); q4 = run(c, "Q4"); cst = run(c, "CST", diag="pos")
    rows.append(f"{c} & {n(float(bar['e_end_ref_pct']),3,True)} & "
                f"{n(float(q4['e_end_ref_pct']),4,True)} & {n(float(q4['e_mid_ref_pct']),4,True)} & "
                f"{n(float(cst['e_end_ref_pct']),4,True)} & {n(float(cst['e_mid_ref_pct']),4,True)} & "
                f"{n(float(q4['e_sigma_probe_ref_pct']),4)} & {n(float(cst['e_sigma_probe_ref_pct']),4)} \\\\")
write("tab_dev.tex", r"""\begin{tabular}{@{}lS[table-format=+1.2]S[table-format=+1.4]S[table-format=+1.4]S[table-format=+1.4]S[table-format=+1.4]S[table-format=1.4]S[table-format=1.4]@{}}
\toprule
工况 & {杆 $e_{\rm end}$} & {Q4 $e_{\rm end}$} & {Q4 $e_{\rm mid}$} & {CST $e_{\rm end}$} & {CST $e_{\rm mid}$} & {Q4 $e_\sigma$} & {CST $e_\sigma$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# T7 stress profile S1 in kPa (section averages)
def profile_lookup(method, x, diag=None):
    for r in PROFILES:
        if r["case"] != "S1" or r["method"] != method:
            continue
        if str(int(float(r["nx"]))) != "160":
            continue
        if diag is not None and r["diag"] != diag:
            continue
        if abs(float(r["x"]) - x) < 1e-9:
            return float(r["sigma"]) / 1e3
    return float("nan")

def probe0(method, x):
    for r in PROBES:
        if r["case"] != "S1" or r["method"] != method:
            continue
        if str(int(float(r["nx"]))) != "160":
            continue
        if method == "CST" and r["diag"] != "pos":
            continue
        if abs(float(r["x"]) - x) < 1e-9 and abs(float(r["y"])) < 1e-9:
            return float(r["sxx"]) / 1e3
    return float("nan")

e = CASE_INFO["S1"]
prof = {round(float(p["x"]), 6): p for p in CCX["cases"]["S1"]["profile"]}
rows = []
for xf in np.arange(0.1, 0.95, 0.1):
    p = prof.get(round(float(xf), 6), {})
    sn = float(p.get("sigma_nom", 0)) / 1e3
    sc = float(p.get("sigma_section_avg", 0)) / 1e3
    rows.append(f"{xf:.1f} & {sn:.2f} & {sc:.2f} & {profile_lookup('Q4', xf):.2f} & "
                f"{profile_lookup('CST', xf, 'pos'):.2f} \\\\")
write("tab_stress_profile.tex", r"""\begin{tabular}{@{}S[table-format=1.1]S[table-format=2.2]S[table-format=2.2]S[table-format=2.2]S[table-format=2.2]@{}}
\toprule
{$x/L$} & {$\sigma_{\rm nom}$/kPa} & {CCX 截面平均/kPa} & {Q4 截面平均/kPa} & {CST 截面平均/kPa} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# T7b point stress at y=0, S1, kPa
CP = load_csv("ccx_reference_points.csv")
def ccx_point0(x):
    best = None
    for r in CP:
        if r["case"] != "S1":
            continue
        try:
            nx = int(r["mesh"].split("x")[0])
        except Exception:
            continue
        if abs(float(r["x"]) - x) < 1e-9 and abs(float(r["y"])) < 1e-9:
            if best is None or nx > best[0]:
                best = (nx, float(r["sxx"]))
    return best[1] / 1e3 if best else float("nan")

rows = []
for xf in (0.1, 0.2, 0.5, 0.8, 0.9):
    rows.append(f"{xf:.1f} & {probe0('Q4', xf):.2f} & {probe0('CST', xf):.2f} & {ccx_point0(xf):.2f} \\\\")
write("tab_point_stress.tex", r"""\begin{tabular}{@{}S[table-format=1.1]S[table-format=2.2]S[table-format=2.2]S[table-format=2.2]@{}}
\toprule
{$x/L$} & {Q4 $\sigma_x(y=0)$/kPa} & {CST $\sigma_x(y=0)$/kPa} & {CCX $\sigma_x(y=0)$/kPa} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# T8 convergence
rows = []
for c in CASES:
    for method, lab in (("Q4", "Q4"), ("CST-pos", "CST")):
        q = run(c, "Q4" if method == "Q4" else "CST", diag=None if method == "Q4" else "pos")
        eg = "--" if method != "Q4" else n(fit(c, "Q4", "grad_fidelity_ref_pct"),2)
        pg = "--" if method != "Q4" else n(fit(c, "Q4", "grad_fidelity_ref_pct"),2)
        rows.append(f"{c} & {lab} & {n(float(q['e_sigma_probe_ref_pct']),4)} & "
                    f"{n(fit(c, method, 'e_sigma_probe_ref_pct'),2)} & "
                    f"{n(float(q['e_sigma_profile_ref_pct']),5)} & "
                    f"{n(fit(c, method, 'e_sigma_profile_ref_pct'),2)} & "
                    f"{n(float(q.get('grad_fidelity_ref_pct', np.nan)),3)} & "
                    f"{n(fit(c, method, 'grad_fidelity_ref_pct'),2)} \\\\")
# fix NaN gradient for CST -> use multicolumn; simpler rebuild rows
rows = []
for c in CASES:
    for method, lab in (("Q4", "Q4"), ("CST-pos", "CST")):
        q = run(c, "Q4" if method == "Q4" else "CST", diag=None if method == "Q4" else "pos")
        if method == "Q4" and c != "S0":
            eg = n(float(q.get("grad_fidelity_ref_pct", np.nan)),3)
            pg = n(fit(c, "Q4", "grad_fidelity_ref_pct"),2)
            tail = f"{eg} & {pg}"
        else:
            tail = r"\multicolumn{1}{c}{--} & \multicolumn{1}{c}{--}"
        rows.append(f"{c} & {lab} & {n(float(q['e_sigma_probe_ref_pct']),4)} & "
                    f"{n(fit(c, method, 'e_sigma_probe_ref_pct'),2)} & "
                    f"{n(float(q['e_sigma_profile_ref_pct']),5)} & "
                    f"{n(fit(c, method, 'e_sigma_profile_ref_pct'),2)} & {tail} \\\\")
write("tab_conv.tex", r"""\begin{tabular}{@{}llS[table-format=1.4]S[table-format=1.2]S[table-format=1.5]S[table-format=1.2]S[table-format=1.3]S[table-format=1.2]@{}}
\toprule
工况 & 方法 & {$e_{\sigma,\rm probe}$/\%} & {$p_{\rm probe}$} & {$e_{\sigma,\rm profile}$/\%} & {$p_{\rm profile}$} & {$e_{\rm grad}$/\%} & {$p_{\rm grad}$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# T9 gradient and shear
rows = []
for c in CASES:
    q = run(c, "Q4")
    if c == "S0":
        rows.append(f"{c} & {float(CASE_INFO[c]['eta']):.2f} & \\multicolumn{{1}}{{c}}{{--}} & \\multicolumn{{1}}{{c}}{{--}} & {100*float(q['shear_energy_ratio']):.3f} \\\\")
    else:
        rows.append(f"{c} & {float(CASE_INFO[c]['eta']):.2f} & {float(q['grad_fidelity_ref_pct']):.3f} & "
                    f"{fit(c,'Q4','grad_fidelity_ref_pct'):.2f} & {100*float(q['shear_energy_ratio']):.3f} \\\\")
write("tab_grad.tex", r"""\begin{tabular}{@{}lS[table-format=1.2]S[table-format=1.3]S[table-format=1.2]S[table-format=1.3]@{}}
\toprule
工况 & {$\eta$} & {$e_{\rm grad}$/\%} & {$p_{\rm grad}$} & {$R_s$/\%} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# T10 CST diagonal
rows = []
for c in CASES:
    r = run(c, "CST-DIAG")
    rows.append(f"{c} & {float(r['d_u_pct']):.2e} & {float(r['d_sigma_axial_pct']):.2e} & "
                f"{float(r['d_sigma_element_pct']):.3f} & {float(r['d_sigma_nodal_pct']):.3f} & "
                f"{100*float(r['cv_max_raw_pos']):.2f} & {100*float(r['cv_max_recovered_pos']):.2f} \\\\")
write("tab_diag.tex", r"""\begin{tabular}{@{}lS[table-format=1.2e-2]S[table-format=1.2e-2]S[table-format=1.3]S[table-format=1.3]S[table-format=2.2]S[table-format=2.2]@{}}
\toprule
工况 & {$D_u$/\%} & {$D_{\sigma,\rm ax}$/\%} & {$D_{\sigma,\rm el}$/\%} & {$D_{\sigma,\rm node}$/\%} & {$CV_{\rm raw}$/\%} & {$CV_{\rm rec}$/\%} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# T11 bar criterion
bc = {r["case"]: r for r in BAR}
rows = []
for c in CASES:
    b = run(c, "BAR-equiv")
    rows.append(f"{c} & {float(bc[c]['eta']):.2f} & {float(bc[c]['angle_deg']):.1f} & {bc[c]['label']} & "
                f"{n(float(b['e_end_ref_pct']),2,True)} & {n(float(b['e_mid_ref_pct']),2,True)} \\\\")
write("tab_bar.tex", r"""\begin{tabular}{@{}lS[table-format=1.2]S[table-format=2.1]cS[table-format=+1.2]S[table-format=+1.2]@{}}
\toprule
工况 & {$\eta$} & {锥角/$^\circ$} & 标签 & {端部偏差/\%} & {中截面偏差/\%} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# T12 anchor
rows = []
for c in CASES:
    e = CASE_INFO[c]; q = run(c, "Q4"); s = run(c, "CST", diag="pos")
    rows.append(f"{c} & {e['u_end_ref']:.4e} & {float(q['u_end']):.4e} & {n(float(q['e_end_ref_pct']),3)} & "
                f"{float(s['u_end']):.4e} & {n(float(s['e_end_ref_pct']),3)} \\\\")
write("tab_anchor.tex", r"""\begin{tabular}{@{}lS[table-format=1.4e-2]S[table-format=1.4e-2]S[table-format=+1.3]S[table-format=1.4e-2]S[table-format=+1.3]@{}}
\toprule
工况 & {CPS8 $u_{\rm end}$/m} & {Q4 $u_{\rm end}$/m} & {Q4 偏差/\%} & {CST $u_{\rm end}$/m} & {CST 偏差/\%} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# T13 load-mode
rows = []
for c in CASES:
    ls = CCX["cases"][c].get("load_mode_sensitivity") or {}
    rows.append(f"{c} & {abs(float(ls.get('u_end_diff_pct', np.nan))):.3f} & {abs(float(ls.get('u_mid_diff_pct', np.nan))):.3f} \\\\")
write("tab_load.tex", r"""\begin{tabular}{@{}lS[table-format=1.3]S[table-format=1.3]@{}}
\toprule
工况 & {$\Delta u_{\rm end}$/\%} & {$\Delta u_{\rm mid}$/\%} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# T14 FEniCS
rows = []
for c in ("S0", "S1"):
    e = FEN["cases"][c]
    rows.append(f"{c} & {e['u_end']:.4e} & {e['u_end_exact_1d']:.4e} & "
                f"{n(100*(e['u_end']/e['u_end_exact_1d']-1),3)} & {e['sigma_mid']:.2f} \\\\")
write("tab_fenics.tex", r"""\begin{tabular}{@{}lS[table-format=1.4e-2]S[table-format=1.4e-2]S[table-format=+1.3]S[table-format=5.2]@{}}
\toprule
工况 & {FEniCS $u_{\rm end}$/m} & {1D $u_{\rm end}$/m} & {偏差/\%} & {$\sigma_{\rm mid}$/Pa} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")
