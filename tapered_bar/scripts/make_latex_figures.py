"""Output-forms figure."""
import os
import sys
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from fem.config import CASES_BY_KEY
from fem.mesh import make_mesh
from fem import bar as bar_mod
from fem import q4 as q4_mod
from fem import cst as cst_mod
from fem import analytic as an
from fem.assembly import end_load_vector

OUT = os.path.join(PROJ, "latex", "figures")
CASE_COLORS = {"S0": "#4d4d4d", "S1": "#1f5fa8", "S2": "#2e8b57",
               "S3": "#d17c00", "S4": "#b22222"}
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 6.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

case = CASES_BY_KEY["S1"]
g, mat, P = case.geom, case.material, case.load
nx, ny = 40, 8
mesh = make_mesh(g, nx, ny)

# 1D bar model
bm = bar_mod.solve_case(case, 40, "equiv")
x = bm.nodes
u_um = bm.U * 1e6
sig_kpa = an.exact_sigma(g, mat, P, x) / 1e3

# Q4 field at element centres
qres = q4_mod.analyze_case(mesh, case)
Q4_U = qres["U"]
cent = mesh.element_centers(mesh.quads)
sx_q4 = np.empty(mesh.nquads)
for k, conn in enumerate(mesh.quads):
    pt = q4_mod.evaluate_point(mesh, case, Q4_U, cent[k, 0], cent[k, 1])
    sx_q4[k] = pt["sxx"]
disp_q4 = np.stack([Q4_U[0::2], Q4_U[1::2]], axis=1)
scale = 2.5e5
q4_def = mesh.nodes + disp_q4 * scale

# CST raw field
cres = cst_mod.solve_case(case, nx, ny, "pos", mesh=mesh)
CST_U = cres["U"]
sx_cst = cres["sigma_e"][:, 0]
cst_def = mesh.nodes + np.stack([CST_U[0::2], CST_U[1::2]], axis=1) * scale

vmin = min(sx_q4.min(), sx_cst.min()) / 1e3
vmax = max(sx_q4.max(), sx_cst.max()) / 1e3
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = "viridis"

fig = plt.figure(figsize=(7.2, 2.55))
gs = fig.add_gridspec(1, 4, width_ratios=[1.02, 1.0, 1.0, 0.045], wspace=0.30)

# (a) 1D bar curves
ax = fig.add_subplot(gs[0, 0])
ax.plot(x, u_um, color="#1f5fa8", lw=1.6, label="u(x) / μm")
ax.set_xlabel("x / m")
ax.set_ylabel("u / μm", color="#1f5fa8")
ax.tick_params(axis="y", colors="#1f5fa8")
ax2 = ax.twinx()
ax2.plot(x, sig_kpa, color="#c84630", lw=1.4, ls="--", label="σ(x) / kPa")
ax2.tick_params(axis="y", colors="#c84630", labelsize=7)
ax.set_title("(a) 杆单元：一维曲线")
ax.set_xlim(0, 1)
ax.grid(alpha=0.2)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)

# (b) Q4 contour
ax = fig.add_subplot(gs[0, 1])
polys = [q4_def[q] for q in mesh.quads]
pc = PolyCollection(polys, array=sx_q4 / 1e3, cmap=cmap, norm=norm,
                    edgecolors="face", linewidths=0.1)
ax.add_collection(pc)
xline = np.linspace(0, g.L, 200)
ax.plot(xline, g.h(xline) / 2, "k--", lw=0.6, alpha=0.7)
ax.plot(xline, -g.h(xline) / 2, "k--", lw=0.6, alpha=0.7)
ax.set_xlim(-0.02, 1.04)
ax.set_ylim(-0.24, 0.24)
ax.set_aspect("equal")
ax.set_title(r"(b) Q4：$\sigma_x$ 云图")
ax.set_xlabel("x / m")
ax.set_ylabel("")
ax.tick_params(labelsize=7)

# (c) CST raw contour
ax = fig.add_subplot(gs[0, 2])
polys = [cst_def[t] for t in mesh.tris_pos]
pc2 = PolyCollection(polys, array=sx_cst / 1e3, cmap=cmap, norm=norm,
                     edgecolors="face", linewidths=0.05)
ax.add_collection(pc2)
ax.plot(xline, g.h(xline) / 2, "k--", lw=0.6, alpha=0.7)
ax.plot(xline, -g.h(xline) / 2, "k--", lw=0.6, alpha=0.7)
ax.set_xlim(-0.02, 1.04)
ax.set_ylim(-0.24, 0.24)
ax.set_aspect("equal")
ax.set_title(r"(c) CST：raw 分块云图")
ax.set_xlabel("x / m")
ax.set_ylabel("")
ax.tick_params(labelsize=7)

# colorbar
cax = fig.add_subplot(gs[0, 3])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label(r"$\sigma_x$ / kPa")
cb.ax.tick_params(labelsize=7)

for ext in ("pdf", "png"):
    path = os.path.join(OUT, f"fig_production_forms.{ext}")
    fig.savefig(path)
    print("wrote", path)
plt.close(fig)

# geometry + mesh figure for the LaTeX report (cleaner layout than fig01)
def fig_geometry():
    from fem.config import CASES
    g0 = CASES_BY_KEY["S1"].geom
    fig = plt.figure(figsize=(7.5, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    for case in CASES:
        g = case.geom
        xline = np.linspace(0, g.L, 200)
        hh = g.h(xline)
        ax.fill_between(xline, -hh / 2, hh / 2, color=CASE_COLORS[case.key],
                        alpha=0.10)
        ax.plot(xline, hh / 2, color=CASE_COLORS[case.key], lw=1.2,
                label=f"{case.key}: $\\eta$={g.eta:.2f}")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("(a) 五种楔形轮廓")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=6.0)
    ax.grid(alpha=0.2)

    ax = fig.add_subplot(gs[0, 1])
    for q in mesh.quads:
        xy = mesh.nodes[np.append(q, q[0])]
        ax.plot(xy[:, 0], xy[:, 1], color="0.4", lw=0.45)
    left = mesh.left_nodes
    ax.plot(mesh.nodes[left, 0], mesh.nodes[left, 1], "s", ms=3.2,
            color="#2e8b57", label="固定端")
    F = end_load_vector(mesh, case.load, mode="consistent")
    right = mesh.right_nodes
    fy = F[2 * right]
    ax.quiver(np.full(right.size, g0.L), mesh.nodes[right, 1],
              np.full(right.size, fy.max()), np.zeros(right.size),
              scale=1 / (0.16 / fy.max()), width=0.006, color="#c84630",
              label="一致端载")
    ax.set_xlim(-0.05, 1.13)
    ax.set_ylim(-0.25, 0.25)
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("(b) S1 网格 $20\\times4$ 与端载")
    ax.legend(loc="upper left", fontsize=6.0)
    ax.grid(alpha=0.15)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig01_geometry_mesh.{ext}")
        fig.savefig(path)
        print("wrote", path)
    plt.close(fig)
fig_geometry()
