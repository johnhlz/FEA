"""Report figures."""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PolyCollection

from fem.config import CASES, CASES_BY_KEY, MESH_LEVELS
from fem.mesh import make_mesh
from fem.assembly import end_load_vector
from fem import analytic as an

FIG = os.path.join(PROJ, "figures")
RES = os.path.join(PROJ, "results")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 6.8,
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

C_Q4 = "#1f5fa8"
C_CST = "#c84630"
C_BAR = "#2e8b57"
C_CCX = "#222222"
C_1D = "#777777"
C_REF = "#8e44ad"
CASE_COLORS = {"S0": "#4d4d4d", "S1": "#1f5fa8", "S2": "#2e8b57",
               "S3": "#d17c00", "S4": "#b22222"}


def load_csv(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_json(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def f(row, key, default=np.nan):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


RUNS = load_csv("study_runs.csv")
PROFILES = load_csv("study_profiles.csv")
PROBES = load_csv("study_probes.csv")
CONV = load_csv("study_convergence.csv")
TABLES = load_json("study_tables.json")
BAR_RUNS = load_csv("bar_runs.csv")
CST_DISP = load_csv("cst_dispersion.csv")
CCX_ANCHOR = load_csv("ccx_anchor.csv")
CCX_POINTS = load_csv("ccx_reference_points.csv")
CCX_PROFILES = load_csv("ccx_profiles.csv")
CCX_REF = load_json("ccx_reference.json")
CCX_VS = load_csv("ccx_vs_inhouse.csv")

CASE_KEYS = [c.key for c in CASES]


def runs(case=None, method=None, nx=None, diag=None, ny=None):
    out = []
    for r in RUNS:
        if case is not None and r["case"] != case:
            continue
        if method is not None and r["method"] != method:
            continue
        if nx is not None and int(float(r["nx"])) != int(nx):
            continue
        if ny is not None and int(float(r["ny"])) != int(ny):
            continue
        if diag is not None and r["diag"] != diag:
            continue
        out.append(r)
    return out


def one(case, method, nx=None, diag=None):
    sel = runs(case, method, nx, diag)
    if not sel:
        return None
    sel.sort(key=lambda r: (int(float(r["nx"])), int(float(r["ny"]))))
    return sel[0]


def ref_u(case):
    c = TABLES.get("cases", {}).get(case, {})
    return f(c, "u_end_ref"), f(c, "u_mid_ref"), f(c, "u_end_1d"), f(c, "u_mid_1d")


def fit_slope(case, method, metric):
    for r in CONV:
        if r["case"] == case and r["method"] == method and r["metric"] == metric:
            return f(r, "slope")
    return np.nan


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig01_geometry_mesh():
    g0 = CASES_BY_KEY["S1"].geom
    fig = plt.figure(figsize=(7.9, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.25, 1.1], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    for case in CASES:
        g = case.geom
        x = np.linspace(0, g.L, 200)
        h = g.h(x)
        ax.fill_between(x, -h / 2, h / 2, color=CASE_COLORS[case.key],
                        alpha=0.10)
        ax.plot(x, h / 2, color=CASE_COLORS[case.key], lw=1.4,
                label=f"{case.key}: β={g.beta:+.2f}, η={g.eta:.2f}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("(a) 五种工况的对称楔形轮廓")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(alpha=0.25)

    # S1 mesh + consistent end loads
    ax = fig.add_subplot(gs[0, 1])
    m = make_mesh(g0, 20, 4)
    for q in m.quads:
        xy = m.nodes[np.append(q, q[0])]
        ax.plot(xy[:, 0], xy[:, 1], color="0.35", lw=0.5)
    ax.plot(m.nodes[:, 0], m.nodes[:, 1], "o", ms=1.4, color="0.2")
    left = m.left_nodes
    ax.plot(m.nodes[left, 0], m.nodes[left, 1], "s", ms=3.5,
            color=C_BAR, label="固定端 ux=uy=0")
    F = end_load_vector(m, CASES_BY_KEY["S1"].load, mode="consistent")
    right = m.right_nodes
    fy = F[2 * right]
    scale = 0.18 / fy.max()
    ax.quiver(np.full(right.size, g0.L), m.nodes[right, 1],
              np.full(right.size, fy.max() * 0.9), np.zeros(right.size),
              scale=1 / scale, width=0.006, color=C_CST,
              label="一致节点载荷（端部均布拉力）")
    ax.set_xlim(-0.06, 1.14)
    ax.set_aspect("equal")
    ax.set_title("(b) S1 的 Q4 网格（20×4）与端部载荷")
    ax.set_xlabel("x / m")
    ax.legend(loc="upper left", fontsize=6.8)
    ax.grid(alpha=0.2)

    # CST diagonals on a zoom
    ax = fig.add_subplot(gs[0, 2])
    m2 = make_mesh(g0, 3, 1)
    q = m2.quads[1]
    xy = m2.nodes[q]
    ax.add_patch(Polygon(xy, closed=True, fill=False, ec="k", lw=1.2))
    n1, n2, n3, n4 = xy
    ax.plot([n1[0], n3[0]], [n1[1], n3[1]], color=C_CST, lw=1.8,
            label="pos 对角线 n1-n3")
    ax.plot([n2[0], n4[0]], [n2[1], n4[1]], color=C_Q4, lw=1.8, ls="--",
            label="neg 对角线 n2-n4")
    for p, lab in zip(xy, ["n1", "n2", "n3", "n4"]):
        ax.text(p[0], p[1], lab, fontsize=8, ha="center", va="bottom")
    ax.set_aspect("equal")
    ax.set_xlim(xy[:, 0].min() - 0.03, xy[:, 0].max() + 0.03)
    ax.set_ylim(xy[:, 1].min() - 0.03, xy[:, 1].max() + 0.03)
    ax.set_title("(c) 同一四边形的两种 CST 剖分")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.2)
    save(fig, "fig01_geometry_mesh.png")


def fig02_displacement():
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 4.0))
    methods = [
        ("BAR-equiv", C_BAR, "1D 杆单元（equiv）", "-"),
        ("Q4", C_Q4, "Q4 160×32", "-"),
        ("CST", C_CST, "CST 160×32", "-"),
    ]
    xpos = np.arange(len(CASE_KEYS))
    width = 0.2
    for ax, key, label in ((axs[0], "e_end_1d_pct", "(a) 端截面平均位移相对一维解偏差"),
                           (axs[1], "e_mid_1d_pct", "(b) 中截面平均位移相对一维解偏差")):
        for k, (method, color, lab, _) in enumerate(methods):
            vals = []
            for case in CASE_KEYS:
                r = one(case, method, nx=160)
                vals.append(f(r, key) if r else np.nan)
            ax.bar(xpos + (k - 1.5) * width, vals, width, color=color,
                   label=lab, edgecolor="white", lw=0.7)
        # reference (CCX / fallback)
        vals = []
        src = TABLES.get("reference_source", "reference")
        for case in CASE_KEYS:
            ue_ref, um_ref, ue1, um1 = ref_u(case)
            vals.append(100 * ((um_ref if key == "e_mid_1d_pct" else ue_ref)
                               / (um1 if key == "e_mid_1d_pct" else ue1) - 1))
        ax.bar(xpos + 1.5 * width, vals, width, color=C_CCX,
               label=f"CCX CPS8 参考", edgecolor="white", lw=0.7)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{c}\nη={CASES_BY_KEY[c].geom.eta:.2f}"
                            for c in CASE_KEYS], fontsize=8)
        ax.set_ylabel("相对一维解偏差 / %")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle("五种工况、160×32 网格下的端部/中截面位移对比（参考为 CalculiX CPS8）",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "fig02_displacement.png")


def _probe_curve(case, method, nx, y_frac=0.0):
    """Probe values at a given y/h fraction (y=0 is exact)."""
    sel = []
    for r in PROBES:
        if r["case"] != case or r["method"] != method:
            continue
        if int(float(r["nx"])) != nx:
            continue
        if method == "CST" and r["diag"] != "pos":
            continue
        try:
            x, y, s = float(r["x"]), float(r["y"]), float(r["sxx"])
        except (TypeError, ValueError):
            continue
        h = float(CASES_BY_KEY[case].geom.h(x))
        if abs(y - y_frac * h) < 1e-9:
            sel.append((x, s))
    sel.sort()
    if not sel:
        return np.array([]), np.array([])
    return np.array([p[0] for p in sel]), np.array([p[1] for p in sel])


def _ccx_curve(case, y_frac=0.0):
    pts = []
    for r in CCX_POINTS:
        if r["case"] != case:
            continue
        try:
            x, y, s = float(r["x"]), float(r["y"]), float(r["sxx"])
        except (TypeError, ValueError):
            continue
        h = float(CASES_BY_KEY[case].geom.h(x))
        if abs(y - y_frac * h) < 1e-6:
            pts.append((x, s))
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def fig03_stress_profile():
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 6.8))
    for ax, case in ((axs[0, 0], "S1"), (axs[0, 1], "S0")):
        g = CASES_BY_KEY[case].geom
        x = np.linspace(0, g.L, 300)
        sn = an.exact_sigma(g, CASES_BY_KEY[case].material,
                            CASES_BY_KEY[case].load, x)
        ax.plot(x, sn / sn[0], color=C_1D, lw=1.6, label="1D 名义 σ=P/A(x)")
        xq, sq = _probe_curve(case, "Q4", 160, 0.0)
        ax.plot(xq, sq / sn[0], "o-", color=C_Q4, ms=3, lw=1.3,
                label="Q4 160×32，y=0")
        xc, sc = _probe_curve(case, "CST", 160, 0.0)
        ax.plot(xc, sc / sn[0], "s--", color=C_CST, ms=3, lw=1.2,
                label="CST 160×32，y=0")
        xr, sr = _ccx_curve(case, 0.0)
        if xr.size:
            ax.plot(xr, sr / sn[0], "k^", ms=5, lw=1.0,
                    label="CalculiX CPS8 240×48，y=0")
        ax.set_xlabel("x / L")
        ax.set_ylabel("σx / σ_nom(0)")
        ax.set_title(f"({'a' if case=='S1' else 'b'}) {case} 轴向应力沿轴曲线（y=0）")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.2)

    # cross sections
    for ax, case, xf in ((axs[1, 0], "S1", 0.5), (axs[1, 1], "S3", 0.1)):
        g = CASES_BY_KEY[case].geom
        sn = float(an.exact_sigma(g, CASES_BY_KEY[case].material,
                                  CASES_BY_KEY[case].load, xf * g.L))
        for method, color, marker, lab in (("Q4", C_Q4, "o", "Q4"),
                                           ("CST", C_CST, "s", "CST")):
            xs, ys, ss = [], [], []
            for r in PROBES:
                if r["case"] != case or r["method"] != method:
                    continue
                if int(float(r["nx"])) != 160:
                    continue
                if method == "CST" and r["diag"] != "pos":
                    continue
                x, y, s = float(r["x"]), float(r["y"]), float(r["sxx"])
                if abs(x - xf * g.L) < 1e-9:
                    xs.append(y / float(g.h(x)))
                    ss.append(s / sn)
            if xs:
                order = np.argsort(xs)
                ax.plot(np.array(xs)[order], np.array(ss)[order], marker + "-",
                        color=color, ms=4, lw=1.2, label=f"{lab} 160×32")
        # CCX cross-section
        xs, ss = [], []
        for r in CCX_POINTS:
            if r["case"] != case:
                continue
            x, y, s = float(r["x"]), float(r["y"]), float(r["sxx"])
            if abs(x - xf * g.L) < 1e-6:
                xs.append(y / float(g.h(x)))
                ss.append(s / sn)
        if xs:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ss)[order], "k^--",
                    ms=5, lw=1.0, label="CalculiX CPS8 240×48")
        ax.set_xlabel("y / h(x)")
        ax.set_ylabel("σx / σ_nom(x)")
        ax.set_title(f"({'c' if case=='S1' else 'd'}) {case} 截面应力分布，x/L={xf}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.2)
    fig.tight_layout()
    save(fig, "fig03_stress_profile.png")


def fig04_convergence():
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.6))
    styles = {"Q4": (C_Q4, "o", "Q4"), "CST-pos": (C_CST, "s", "CST")}
    # (a),(b): displacement errors vs the CCX reference (no order claim:
    # the fixed-end corner singularity and the end boundary layer make the
    # pointwise displacement convergence irregular).
    for ax, metric, title, ylab in (
        (axs[0, 0], "e_end_ref_pct", "(a) 端截面平均位移相对误差", "|e_u,end| / %"),
        (axs[0, 1], "e_mid_ref_pct", "(b) 中截面平均位移相对误差", "|e_u,mid| / %"),
    ):
        for method, (color, marker, lab) in styles.items():
            run_method = "CST" if method == "CST-pos" else method
            diag = "pos" if method == "CST-pos" else "-"
            for case, ls in (("S1", "-"), ("S3", "--")):
                rows = [r for r in RUNS if r["case"] == case
                        and r["method"] == run_method and r["diag"] == diag]
                rows.sort(key=lambda r: f(r, "nx"))
                hs, es = [], []
                for r in rows:
                    e = abs(f(r, metric))
                    if np.isfinite(e) and e > 0:
                        hs.append(1.0 / f(r, "nx"))
                        es.append(e)
                ax.loglog(hs, es, marker=marker, ls=ls, color=color,
                          ms=3.5, lw=1.2, alpha=0.9, label=f"{lab} {case}")
        ax.set_xlabel("h = 1/nx")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(which="both", alpha=0.2)
        ax.legend(fontsize=7)

    # (c),(d): stress errors with fitted orders
    for ax, metric, title, ylab in (
        (axs[1, 0], "e_sigma_probe_ref_pct", "(c) 探针点应力误差 e_σ", "e_σ,probe / %"),
        (axs[1, 1], "e_sigma_profile_ref_pct", "(d) 截面平均应力误差", "e_σ,profile / %"),
    ):
        for method, (color, marker, lab) in styles.items():
            run_method = "CST" if method == "CST-pos" else method
            diag = "pos" if method == "CST-pos" else "-"
            for case, ls in (("S1", "-"), ("S3", "--")):
                rows = [r for r in RUNS if r["case"] == case
                        and r["method"] == run_method and r["diag"] == diag]
                rows.sort(key=lambda r: f(r, "nx"))
                hs = np.array([1.0 / f(r, "nx") for r in rows])
                es = np.array([f(r, metric) for r in rows])
                slope = fit_slope(case, method, metric)
                slope_txt = f" (p={slope:.2f})" if np.isfinite(slope) else ""
                ax.loglog(hs, es, marker=marker, ls=ls, color=color,
                          ms=3.5, lw=1.2, alpha=0.9,
                          label=f"{lab} {case}{slope_txt}")
        ax.set_xlabel("h = 1/nx")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(which="both", alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle("Q4 / CST 的位移与应力收敛（参考：CalculiX CPS8 240×48）",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "fig04_convergence.png")


def fig05_gradient_shear():
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.6))
    ax = axs[0]
    for case in CASE_KEYS:
        if case == "S0":
            continue
        rows = [r for r in RUNS if r["case"] == case and r["method"] == "Q4"]
        rows.sort(key=lambda r: f(r, "nx"))
        hs, es = [], []
        for r in rows:
            e = f(r, "grad_fidelity_ref_pct")
            if np.isfinite(e) and 0 < e < 50:
                hs.append(1.0 / f(r, "nx"))
                es.append(e)
        if hs:
            ax.loglog(hs, es, "o-", ms=3, lw=1.2,
                      color=CASE_COLORS[case], label=case)

    ax.set_xlabel("h = 1/nx")
    ax.set_ylabel("梯度保真误差 / %")
    ax.set_title("(a) Q4 截面平均应力梯度保真误差")
    ax.grid(which="both", alpha=0.2)
    ax.legend(fontsize=7)

    ax = axs[1]
    for case in CASE_KEYS:
        r = one(case, "Q4", nx=160)
        if not r:
            continue
        eta = CASES_BY_KEY[case].geom.eta
        rs = f(r, "shear_energy_ratio") * 100
        ax.plot(eta, rs, "o", ms=8, color=CASE_COLORS[case])
        ax.annotate(f"{case}\nη={eta:.2f}", (eta, rs),
                    textcoords="offset points", xytext=(6, 2), fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("一维适用性指标 η = |β| h0 / L")
    ax.set_ylabel("剪切能比 R_s / %")
    ax.set_title("(b) 横向/剪切应变能占比（160×32）")
    ax.grid(which="both", alpha=0.2)
    ax.axvspan(0.0, 0.2, color=C_BAR, alpha=0.07)
    ax.axvspan(0.2, 0.4, color="#d17c00", alpha=0.07)
    ax.axvspan(0.4, 0.75, color=C_CST, alpha=0.07)
    ax.text(0.1, ax.get_ylim()[1] * 0.9, "A", ha="center", fontsize=9,
            color=C_BAR)
    ax.text(0.3, ax.get_ylim()[1] * 0.9, "B", ha="center", fontsize=9,
            color="#d17c00")
    ax.text(0.55, ax.get_ylim()[1] * 0.9, "C", ha="center", fontsize=9,
            color=C_CST)
    fig.tight_layout()
    save(fig, "fig05_gradient_shear.png")


def fig06_cst_diagonal():
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.4))
    ax = axs[0]
    for case, ls in (("S1", "-"), ("S4", "--")):
        rows = [r for r in RUNS if r["case"] == case
                and r["method"] == "CST-DIAG"]
        rows.sort(key=lambda r: f(r, "nx"))
        h = np.array([1.0 / f(r, "nx") for r in rows])
        ax.loglog(h, [f(r, "d_sigma_element_pct") for r in rows],
                  "o" + ls, color=C_CST, ms=4, lw=1.2,
                  label=f"{case} 单元应力 RMS 差")
        ax.loglog(h, [f(r, "d_sigma_nodal_pct") for r in rows],
                  "s" + ls, color=C_Q4, ms=4, lw=1.2,
                  label=f"{case} 节点恢复应力 RMS 差")
    ax.set_xlabel("h = 1/nx")
    ax.set_ylabel("对角线离散度 / %")
    ax.set_title("(a) CST 两对角线的局部应力离散度")
    ax.grid(which="both", alpha=0.2)
    ax.legend(fontsize=7)

    ax = axs[1]
    for case, color in (("S1", C_CST), ("S0", C_Q4)):
        rows = [r for r in CST_DISP if r["case"] == case
                and r["diag"] == "pos" and abs(f(r, "x") - 0.5) < 1e-9]
        rows.sort(key=lambda r: f(r, "nx"))
        ns = np.array([f(r, "nx") for r in rows])
        ax.loglog(ns, [f(r, "cv_raw") * 100 for r in rows], "o-",
                  color=color, ms=4, label=f"{case} raw 单元常应力")
        ax.loglog(ns, [f(r, "cv_recovered") * 100 for r in rows], "s--",
                  color=color, alpha=0.7, ms=4,
                  label=f"{case} 面积加权恢复")
    ax.set_xlabel("nx（ny = nx/5）")
    ax.set_ylabel("截面内应力离散系数 / %")
    ax.set_title("(b) 中截面节点恢复对棋盘振荡的抑制")
    ax.grid(which="both", alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig06_cst_diagonal.png")


def fig07_bar():
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.4))
    ax = axs[0]
    for method, color, marker in (("equiv", C_BAR, "o"), ("mean", C_CST, "s")):
        rows = [r for r in BAR_RUNS if r["case"] == "S1"
                and r["method"] == method]
        rows.sort(key=lambda r: f(r, "n_elem"))
        ns = np.array([f(r, "n_elem") for r in rows])
        es = np.array([abs(f(r, "e_end_pct")) + 1e-16 for r in rows])
        ax.loglog(ns, es, marker + "-", color=color, ms=4, lw=1.2,
                  label=("柔度积分等效刚度（equiv）" if method == "equiv"
                         else "算术平均面积刚度（mean）"))
    ax.set_xlabel("单元数 n")
    ax.set_ylabel("|端部位移相对误差| / %")
    ax.set_title("(a) S1 杆单元：等效刚度 vs 平均面积刚度")
    ax.grid(which="both", alpha=0.2)
    ax.legend(fontsize=7.5)

    ax = axs[1]
    for case in CASE_KEYS:
        ue_ref, um_ref, ue1, um1 = ref_u(case)
        eta = CASES_BY_KEY[case].geom.eta
        e = 100 * (ue_ref / ue1 - 1)
        ax.plot(eta, e, "o", ms=8, color=CASE_COLORS[case])
        ax.annotate(case, (eta, e), textcoords="offset points",
                    xytext=(6, 2), fontsize=7.5)
    ax.axvspan(0.0, 0.2, color=C_BAR, alpha=0.08)
    ax.axvspan(0.2, 0.4, color="#d17c00", alpha=0.08)
    ax.axvspan(0.4, 0.75, color=C_CST, alpha=0.08)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("η = |β| h0 / L")
    ax.set_ylabel("CCX 参考端部位移相对一维解偏差 / %")
    ax.set_title("(b) 一维杆理论在二维参考下的端部位移偏差（A/B/C 判据区间）")
    ax.grid(alpha=0.25)
    ax.text(0.1, ax.get_ylim()[0] * 0.9, "A", ha="center", color=C_BAR)
    ax.text(0.3, ax.get_ylim()[0] * 0.9, "B", ha="center", color="#d17c00")
    ax.text(0.55, ax.get_ylim()[0] * 0.9, "C", ha="center", color=C_CST)
    fig.tight_layout()
    save(fig, "fig07_bar.png")


def fig08_anchor():
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.6))

    # (a) CPS4 vs in-house Q4 / CST on the same mesh (area-weighted u_end)
    ax = axs[0]
    cps4 = {}
    for r in CCX_ANCHOR:
        if r.get("element_type") != "CPS4":
            continue
        try:
            cps4[(r["case"], r["mesh"])] = float(r["u_end"])
        except (KeyError, ValueError):
            pass
    for method, diag, case, color, marker, lab in (
        ("Q4", "-", "S0", "0.45", "o", "Q4 S0"),
        ("Q4", "-", "S1", C_Q4, "s", "Q4 S1"),
        ("CST", "pos", "S1", C_CST, "^", "CST S1"),
    ):
        sel = []
        for r in RUNS:
            if r["case"] != case or r["method"] != method:
                continue
            if diag is not None and r["diag"] != diag:
                continue
            if int(float(r["nx"])) <= 1 or int(float(r["ny"])) == 0:
                continue
            key = (case, f"{int(float(r['nx']))}x{int(float(r['ny']))}")
            if key in cps4:
                sel.append((int(float(r["nx"])), abs(100 * (float(r["u_end"]) / cps4[key] - 1))))
        if not sel:
            continue
        sel.sort()
        ns = np.array([p[0] for p in sel])
        vals = np.array([p[1] for p in sel])
        ax.loglog(ns, vals + 1e-10, marker + "-", ms=3.5, lw=1.1,
                  color=color, label=lab)
    ax.set_xlabel("nx")
    ax.set_ylabel("|u_end - u_end,CPS4| / %")
    ax.set_title("(a) 同网格 CPS4 锚点对比")
    ax.grid(which="both", alpha=0.2)
    ax.legend(fontsize=6.5)

    # (b) CPS8 reference vs in-house at 160x32
    ax = axs[1]
    xpos = np.arange(len(CASE_KEYS))
    width = 0.34
    vals_q4, vals_cst = [], []
    for case in CASE_KEYS:
        ue_ref = float(TABLES["cases"][case]["u_end_ref"])
        rq = one(case, "Q4", nx=160)
        rc = one(case, "CST", nx=160, diag="pos")
        vals_q4.append(100 * (f(rq, "u_end") / ue_ref - 1) if rq else np.nan)
        vals_cst.append(100 * (f(rc, "u_end") / ue_ref - 1) if rc else np.nan)
    ax.bar(xpos - width / 2, vals_q4, width, color=C_Q4, label="Q4")
    ax.bar(xpos + width / 2, vals_cst, width, color=C_CST, label="CST")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(xpos)
    ax.set_xticklabels(CASE_KEYS, fontsize=7.5)
    ax.set_ylabel("u_end 相对 CPS8 参考 / %")
    ax.set_title("(b) 与 CPS8 参考的端部位移偏差")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig08_anchor.png")

def fig09_deformed():
    case = CASES_BY_KEY["S1"]
    g = case.geom
    from fem import q4 as q4_mod
    m = make_mesh(g, 40, 8)
    res = q4_mod.analyze_case(m, case)
    U = res["U"]
    scale = 2.5e5
    disp = np.stack([U[0::2], U[1::2]], axis=1) * scale
    xy_def = m.nodes + disp
    D = case.material.D_plane_stress()
    sigma_e = []
    for conn in m.quads:
        xy = m.nodes[conn]
        # element-centre stress from the q4 data
        dof = np.stack([2 * conn, 2 * conn + 1], axis=1).ravel()
        # use q4 evaluate at centroid through a temporary point
        c = xy.mean(axis=0)
        pt = q4_mod.evaluate_point(m, case, U, c[0], c[1])
        sigma_e.append(pt["sxx"])
    sigma_e = np.array(sigma_e)
    polys = [xy_def[q] for q in m.quads]
    fig, ax = plt.subplots(figsize=(7.3, 3.0))
    pc = PolyCollection(polys, array=sigma_e, cmap="viridis",
                        edgecolors="0.4", linewidths=0.15)
    ax.add_collection(pc)
    cb = fig.colorbar(pc, ax=ax, pad=0.01)
    cb.set_label("σx / Pa")
    ax.plot(m.nodes[m.quads[:, 0], 0], m.nodes[m.quads[:, 0], 1], alpha=0)
    x = np.linspace(0, g.L, 200)
    ax.plot(x, g.h(x) / 2, "k--", lw=1, label="未变形轮廓")
    ax.plot(x, -g.h(x) / 2, "k--", lw=1)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.08)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("S1 主构型：Q4 40×8 变形网格与 σx（变形放大 2.5e5）")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig09_deformed.png")


if __name__ == "__main__":
    fig01_geometry_mesh()
    fig02_displacement()
    fig03_stress_profile()
    fig04_convergence()
    fig05_gradient_shear()
    fig06_cst_diagonal()
    fig07_bar()
    fig08_anchor()
    fig09_deformed()
    print("figures ok")
