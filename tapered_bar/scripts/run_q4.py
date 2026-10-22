"""Q4 plane-stress study."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fem import analytic as an
from fem.assembly import end_load_vector
from fem.config import CASES, INTERIOR_STATIONS, MESH_LEVELS, STATIONS
from fem.io import fnum, save_csv, save_json
from fem.mesh import make_mesh
from fem.q4 import (
    analyze_case,
    b_matrix,
    consistent_end_load_vector,
    evaluate_point,
    jacobian,
    q4_element,
    shape_q4,
    solve_q4,
)

RESULTS = ROOT / "results"
PROBE_X = (0.1, 0.2, 0.5, 0.8, 0.9)

RUN_FIELDS = [
    "case", "nx", "ny", "nnodes", "ndof", "u_end", "u_mid",
    "u_end_exact", "u_mid_exact", "e_end_pct", "e_mid_pct",
    "sigma_mid", "sigma_mid_exact", "e_sigma_mid_pct",
    "sigma_profile_rms_pct", "grad_profile_rms_pct",
    "shear_energy_ratio", "reaction_x", "note",
]
PROFILE_FIELDS = [
    "case", "nx", "ny", "x", "u_section_avg", "sigma_section_avg",
    "sigma_nom", "dsigma_dx_fem", "dsigma_dx_nom",
]
GRAD_FIELDS = ["case", "nx", "ny", "x", "dsigma_dx_fem", "dsigma_dx_nom"]
PROBE_FIELDS = ["case", "nx", "ny", "x", "y", "ux", "uy", "sxx", "syy", "sxy"]
PROBE_SINGLE_FIELDS = PROBE_FIELDS + ["elem", "xi", "eta", "n_elems"]
INT_FIELDS = {"nx", "ny", "nnodes", "ndof", "elem", "n_elems"}
STR_FIELDS = {"case", "note"}
COORD_FIELDS = {"x", "y"}


def fmt_row(row, fields):
    """Format one result row for CSV: floats as scientific notation, blank for NaN."""
    out = {}
    for key in fields:
        val = row.get(key)
        if key in INT_FIELDS and val is not None:
            out[key] = int(val)
        elif key in STR_FIELDS:
            out[key] = "" if val is None else str(val)
        elif val is None:
            out[key] = ""
        else:
            try:
                fv = float(val)
            except (TypeError, ValueError):
                out[key] = str(val)
                continue
            if not math.isfinite(fv):
                out[key] = ""
            elif key in COORD_FIELDS:
                out[key] = fnum(fv, "{:.10g}")
            else:
                out[key] = fnum(fv)
    return out


def probe_points(case):
    """(x, y) probe list: x in PROBE_X and y = 0, +-h(x)/4."""
    pts = []
    for x in PROBE_X:
        h = float(case.geom.h(x))
        pts.extend([(float(x), 0.0), (float(x), 0.25 * h),
                    (float(x), -0.25 * h)])
    return pts


def observed_orders(values):
    """Self-convergence orders p = log2(d_k / d_{k+1}) for a mesh sequence."""
    vals = [float(v) for v in values]
    diffs = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
    orders = []
    for i in range(len(diffs) - 1):
        if diffs[i] > 0.0 and diffs[i + 1] > 0.0:
            orders.append(float(math.log2(diffs[i] / diffs[i + 1])))
        else:
            orders.append(None)
    return orders, diffs


def gradient_stencil_floor(case):
    """RMS relative error of the same central-difference stencil applied to the"""
    if abs(case.geom.beta) < 1e-14:
        return None
    xs = case.geom.L * STATIONS
    s = an.exact_sigma(case.geom, case.material, case.load, xs)
    dx = xs[1] - xs[0]
    fd = (s[2:] - s[:-2]) / (2.0 * dx)
    gn = an.exact_sigma_gradient(case.geom, case.material, case.load,
                                 case.geom.L * INTERIOR_STATIONS)
    return float(100.0 * np.sqrt(np.mean((fd - gn) ** 2)) / np.sqrt(np.mean(gn ** 2)))


# self-checks (also covered by tests/test_q4.py)
LINEAR_FIELD = (2.0, 0.5, -0.2, 0.1, -0.3, 0.7)   # a0, a1, a2, b0, b1, b2


def self_check_patch():
    """Constant-strain patch test on a tapered element."""
    case = CASES[1]                       # S1, beta = 1
    mesh = make_mesh(case.geom, 10, 2)
    xy = mesh.nodes[mesh.quads[3]]
    D = case.material.D_plane_stress()
    ke, info = q4_element(xy, D, case.geom.width)
    a = LINEAR_FIELD
    ue = np.empty(8)
    for k in range(4):
        x, y = xy[k]
        ue[2 * k] = 1e-3 * (a[0] + a[1] * x + a[2] * y)
        ue[2 * k + 1] = 1e-3 * (a[3] + a[4] * x + a[5] * y)
    eps = np.array([a[1], a[5], a[2] + a[4]]) * 1e-3
    sig = D @ eps
    err = max(float(np.max(np.abs(D @ B @ ue - sig)) / np.max(np.abs(sig)))
              for B in info["B"])
    assert err < 1e-10, f"patch test failed: rel err {err:.3e}"
    return err


def self_check_rigid():
    """Rigid-body translation must produce zero stress."""
    case = CASES[4]                       # S4, strong taper
    mesh = make_mesh(case.geom, 10, 2)
    D = case.material.D_plane_stress()
    err_max = 0.0
    for conn in mesh.quads:
        xy = mesh.nodes[conn]
        ke, info = q4_element(xy, D, case.geom.width)
        ue = np.tile([1e-3, -2e-3], 4)
        for B in info["B"]:
            sigma = D @ B @ ue
            err_max = max(err_max, float(np.max(np.abs(sigma))))
    rel = err_max / (case.material.E * 1e-3)
    assert rel < 1e-10, f"rigid translation stress too large: {rel:.3e}"
    return rel


def self_check_quadrature():
    """For a rectangle 2x2 Gauss must equal 3x3 Gauss."""
    case = CASES[0]                       # S0, rectangular elements
    mesh = make_mesh(case.geom, 10, 2)
    xy = mesh.nodes[mesh.quads[0]]
    D = case.material.D_plane_stress()
    ke2, _ = q4_element(xy, D, case.geom.width)

    gp = np.sqrt(0.6)
    pts = [(-gp, -gp), (0.0, -gp), (gp, -gp),
           (-gp, 0.0), (0.0, 0.0), (gp, 0.0),
           (-gp, gp), (0.0, gp), (gp, gp)]
    wts = [25.0 / 81, 40.0 / 81, 25.0 / 81,
           40.0 / 81, 64.0 / 81, 40.0 / 81,
           25.0 / 81, 40.0 / 81, 25.0 / 81]
    ke3 = np.zeros((8, 8))
    for (xi, eta), w in zip(pts, wts):
        N, dN = shape_q4(xi, eta)
        J, detJ, invJ, G = jacobian(xy, dN)
        B = b_matrix(G[:, 0], G[:, 1])
        ke3 += w * detJ * case.geom.width * (B.T @ D @ B)
    rel = float(np.max(np.abs(ke2 - ke3)) / np.max(np.abs(ke3)))
    assert rel < 1e-10, f"2x2 vs 3x3 rectangle mismatch: {rel:.3e}"
    return rel


def main():
    t_start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)

    patch_err = self_check_patch()
    rigid_err = self_check_rigid()
    quad_err = self_check_quadrature()
    print(f"checks patch={patch_err:.2e} rigid={rigid_err:.2e} "
          f"2x2_vs_3x3_rect={quad_err:.2e}")

    run_rows, profile_rows, grad_rows = [], [], []
    probe_rows, probe_single_rows = [], []
    per_case = {}
    max_reaction_err = 0.0

    for case in CASES:
        store = dict(run=[], extras=[], profile=[], grad=[],
                     probe=[], probe_single=[])
        for nx, ny in MESH_LEVELS:
            mesh = make_mesh(case.geom, nx, ny)
            out = analyze_case(mesh, case)
            rr, ex = out["run_row"], out["extras"]
            max_reaction_err = max(max_reaction_err, ex["reaction_rel_err"])

            run_rows.append(rr)
            profile_rows.extend(out["profile_rows"])
            grad_rows.extend(out["gradient_rows"])
            store["run"].append(rr)
            store["extras"].append(ex)
            store["profile"].extend(out["profile_rows"])
            store["grad"].extend(out["gradient_rows"])

            for x, y in probe_points(case):
                pa = evaluate_point(mesh, case, out["U"], x, y,
                                    average_shared=True)
                ps = evaluate_point(mesh, case, out["U"], x, y,
                                    average_shared=False)
                row_a = dict(case=case.key, nx=nx, ny=ny, x=x, y=y,
                             ux=pa["ux"], uy=pa["uy"], sxx=pa["sxx"],
                             syy=pa["syy"], sxy=pa["sxy"])
                probe_rows.append(row_a)
                store["probe"].append(row_a)
                row_s = dict(row_a)
                row_s.update(elem=ps["elem"], xi=ps["xi"], eta=ps["eta"],
                             n_elems=pa["n_elems"])
                probe_single_rows.append(row_s)
                store["probe_single"].append(row_s)

            assert ex["reaction_rel_err"] < 1e-8, (
                f"{case.key} {nx}x{ny}: reaction error {ex['reaction_rel_err']:.2e}")
            assert ex["detJ_min"] > 0.0, f"{case.key} {nx}x{ny}: detJ <= 0"
            assert rr["u_end"] > rr["u_mid"] > 0.0, (
                f"{case.key} {nx}x{ny}: invalid displacement ordering")
            for key in ("u_end", "u_mid", "sigma_profile_rms_pct",
                        "shear_energy_ratio"):
                assert math.isfinite(float(rr[key])), f"{case.key}: NaN in {key}"
            for pr in out["profile_rows"]:
                assert math.isfinite(pr["sigma_section_avg"]), (
                    f"{case.key}: non-finite section stress")

        per_case[case.key] = store
        # legacy lumped-load diagnostic on the finest mesh (used to quantify
        # how sensitive u_end / near-end point stresses are to load lumping)
        mesh_f = make_mesh(case.geom, *MESH_LEVELS[-1])
        F_lumped = end_load_vector(mesh_f, case.load, mode="lumped")
        res_l = solve_q4(mesh_f, case, load_vector=F_lumped)
        ux_l = res_l["U"][0::2]
        u_end_l = float(ux_l[mesh_f.right_nodes].mean())
        u_mid_l = float(ux_l[mesh_f.column_nodes(mesh_f.mid_index)].mean())
        pt_l = evaluate_point(mesh_f, case, res_l["U"], 0.9, 0.0)
        pt_c = evaluate_point(mesh_f, case, out["U"], 0.9, 0.0)
        last = store["run"][-1]
        consistency = dict(
            mesh=f"{MESH_LEVELS[-1][0]}x{MESH_LEVELS[-1][1]}",
            u_end_consistent=float(last["u_end"]),
            u_end_lumped=u_end_l,
            u_mid_consistent=float(last["u_mid"]),
            u_mid_lumped=u_mid_l,
            d_u_end_pct=float(100.0 * (u_end_l - last["u_end"]) / last["u_end"]),
            d_u_mid_pct=float(100.0 * (u_mid_l - last["u_mid"]) / last["u_mid"]),
            sxx_09_0_consistent=float(pt_c["sxx"]),
            sxx_09_0_lumped=float(pt_l["sxx"]),
            d_sxx_09_0_pct=float(100.0 * (pt_l["sxx"] - pt_c["sxx"])
                                 / pt_c["sxx"]),
        )
        store["lumped_check"] = consistency
        print(case.key, len(store["run"]), "meshes",
              f"u_end={last['u_end']:.6e} u_mid={last['u_mid']:.6e}")

    n_case, n_mesh = len(CASES), len(MESH_LEVELS)
    assert len(run_rows) == n_case * n_mesh
    assert len(profile_rows) == n_case * n_mesh * len(STATIONS)
    assert len(grad_rows) == n_case * n_mesh * len(INTERIOR_STATIONS)
    assert len(probe_rows) == n_case * n_mesh * len(PROBE_X) * 3
    assert len(probe_single_rows) == len(probe_rows)
    assert all(math.isfinite(float(r["sxx"])) for r in probe_rows)
    assert max_reaction_err < 1e-8, f"reaction check failed: {max_reaction_err:.2e}"

    # S0: deviation from the 1D value 1.25e-7 must shrink with refinement
    s0 = per_case["S0"]["run"]
    u_ref = 1.25e-7
    dev = [abs(float(r["u_mid"]) - u_ref) / u_ref for r in s0]
    assert all(dev[i + 1] < dev[i] for i in range(len(dev) - 1)), (
        f"S0 u_mid deviation not monotone: {dev}")
    assert dev[-1] < 1e-2, f"S0 finest u_mid deviation {dev[-1]:.3e} too large"

    # 80x16 vs 160x32 mid-section displacement difference
    u_mid_gap = {}
    for case in CASES:
        rows = [r for r in per_case[case.key]["run"] if r["nx"] in (80, 160)]
        rows.sort(key=lambda r: r["nx"])
        gap = abs(float(rows[1]["u_mid"]) - float(rows[0]["u_mid"])) / abs(
            float(rows[1]["u_mid"]))
        u_mid_gap[case.key] = float(gap)
        assert gap < 5e-3, f"{case.key}: 80->160 u_mid gap {gap:.3e} too large"
    for key in ("S0", "S1", "S2"):
        assert u_mid_gap[key] < 5e-4, (
            f"{key}: 80->160 u_mid gap {u_mid_gap[key]:.3e} >= 0.05%")
    print("u_mid gap "
          + ", ".join(f"{k}={v:.2e}" for k, v in u_mid_gap.items()))

    for case in CASES:
        store = per_case[case.key]
        last = store["run"][-1]
        assert float(last["sigma_profile_rms_pct"]) < 0.05, (
            f"{case.key}: finest sigma profile RMS too large")
        if abs(case.geom.beta) > 1e-14:
            assert float(last["grad_profile_rms_pct"]) < 5.0, (
                f"{case.key}: finest gradient RMS too large")
            assert float(last["grad_profile_rms_pct"]) < float(
                store["run"][0]["grad_profile_rms_pct"]), (
                f"{case.key}: gradient error does not decrease")

    summary_cases = {}
    for case in CASES:
        store = per_case[case.key]
        rows = sorted(store["run"], key=lambda r: (r["nx"], r["ny"]))
        extras = [store["extras"][store["run"].index(r)] for r in rows]
        u_mid = [float(r["u_mid"]) for r in rows]
        u_end = [float(r["u_end"]) for r in rows]
        sig = [float(r["sigma_profile_rms_pct"]) for r in rows]
        grad = [float(r["grad_profile_rms_pct"]) for r in rows]
        p_mid, d_mid = observed_orders(u_mid)
        p_end, d_end = observed_orders(u_end)
        p_sig, d_sig = observed_orders(sig)
        p_grad, d_grad = (observed_orders(grad)
                          if abs(case.geom.beta) > 1e-14 else ([], []))
        meshes = []
        for r, ex in zip(rows, extras):
            meshes.append(dict(
                nx=int(r["nx"]), ny=int(r["ny"]),
                nnodes=int(r["nnodes"]), ndof=int(r["ndof"]),
                u_end=float(r["u_end"]), u_mid=float(r["u_mid"]),
                u_end_exact=float(r["u_end_exact"]),
                u_mid_exact=float(r["u_mid_exact"]),
                e_end_pct=float(r["e_end_pct"]),
                e_mid_pct=float(r["e_mid_pct"]),
                sigma_mid=float(r["sigma_mid"]),
                sigma_mid_exact=float(r["sigma_mid_exact"]),
                e_sigma_mid_pct=float(r["e_sigma_mid_pct"]),
                sigma_profile_rms_pct=float(r["sigma_profile_rms_pct"]),
                grad_profile_rms_pct=(None if not math.isfinite(
                    float(r["grad_profile_rms_pct"]))
                    else float(r["grad_profile_rms_pct"])),
                u_mid_weighted=float(ex["u_mid_weighted"]),
                u_end_weighted=float(ex["u_end_weighted"]),
                u_mid_arith_bias_pct=float(ex["u_mid_arith_bias_pct"]),
                grad_5point_rms_pct=(None if not math.isfinite(
                    float(ex["grad_5point_rms_pct"]))
                    else float(ex["grad_5point_rms_pct"])),
                shear_energy_ratio=float(r["shear_energy_ratio"]),
                reaction_x=float(r["reaction_x"]),
                reaction_rel_err=float(ex["reaction_rel_err"]),
                detJ_min=float(ex["detJ_min"]),
                detJ_max=float(ex["detJ_max"]),
                note=r["note"],
            ))
        u_mid_w = [m["u_mid_weighted"] for m in meshes]
        u_end_w = [m["u_end_weighted"] for m in meshes]
        p_midw, d_midw = observed_orders(u_mid_w)
        p_endw, d_endw = observed_orders(u_end_w)
        summary_cases[case.key] = dict(
            label=case.label,
            beta=float(case.geom.beta),
            h0=float(case.geom.h0),
            h1=float(case.geom.h1),
            eta=float(case.geom.eta),
            load=float(case.load),
            meshes=meshes,
            observed_orders=dict(
                u_mid=p_mid, u_end=p_end,
                u_mid_weighted=p_midw, u_end_weighted=p_endw,
                sigma_profile_rms=p_sig, grad_profile_rms=p_grad,
            ),
            self_convergence_diffs=dict(
                u_mid=d_mid, u_end=d_end,
                u_mid_weighted=d_midw, u_end_weighted=d_endw,
                sigma_profile_rms=d_sig, grad_profile_rms=d_grad,
            ),
            gradient_stencil_floor_pct=gradient_stencil_floor(case),
            lumped_load_check=store["lumped_check"],
            u_mid_80_160_gap_pct=float(100.0 * u_mid_gap[case.key]),
        )

    save_csv(str(RESULTS / "q4_runs.csv"),
             [fmt_row(r, RUN_FIELDS) for r in run_rows], RUN_FIELDS)
    save_csv(str(RESULTS / "q4_profiles.csv"),
             [fmt_row(r, PROFILE_FIELDS) for r in profile_rows], PROFILE_FIELDS)
    save_csv(str(RESULTS / "q4_gradient.csv"),
             [fmt_row(r, GRAD_FIELDS) for r in grad_rows], GRAD_FIELDS)
    save_csv(str(RESULTS / "q4_probe.csv"),
             [fmt_row(r, PROBE_FIELDS) for r in probe_rows], PROBE_FIELDS)
    save_csv(str(RESULTS / "q4_probe_single.csv"),
             [fmt_row(r, PROBE_SINGLE_FIELDS) for r in probe_single_rows],
             PROBE_SINGLE_FIELDS)
    print(f"csv runs={len(run_rows)} profiles={len(profile_rows)} "
          f"grad={len(grad_rows)} probes={len(probe_rows)}")

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj) if math.isfinite(float(obj)) else None
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    summary = dict(
        method="Q4 bilinear isoparametric plane stress, 2x2 Gauss full integration",
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        python=sys.version.split()[0],
        numpy=np.__version__,
        command="python3 scripts/run_q4.py",
        conventions=dict(
            formulation="self-written bilinear N, Jacobian, B(3x8), ke with 2x2 Gauss full integration",
            solver="shared fem.assembly assemble_sparse / dirichlet_solve / fixed_dofs_left / end_load_vector",
            end_load="consistent nodal loads of a uniform axial traction on the right end, total P (shared end_load_vector default mode='consistent')",
            u_end="arithmetic mean of ux over right-end nodes",
            u_mid="arithmetic mean of ux over the x=L/2 node column",
            u_mid_weighted="trapezoidal (area-weighted) mean over the same column; diagnostic only",
            sigma_section_avg="equal-weight mean of sigma_x at all 2x2 Gauss points of the element columns adjacent to the station node column",
            sigma_profile_rms_pct="100*RMS(sigma_avg-sigma_nom)/RMS(sigma_nom) at x=0.1..0.9",
            grad_profile_rms_pct="100*RMS(dsigma/dx_fd-dsigma/dx_nom)/RMS(dsigma/dx_nom) at x=0.1..0.9, second-order central difference with dx=0.1 L (NaN for beta=0); profile endpoints x=0,L use one-sided second-order differences",
            grad_5point="4th-order 5-point derivative of the same section profile at x=0.2..0.8; pure stencil-truncation diagnostic, not written to CSV",
            shear_energy_ratio="sum_gp (sigma_y^2+tau_xy^2) detJ / sum_gp sigma_x^2 detJ (Gauss-point volume weights)",
            convergence_order="p = log2(|v_{k+1}-v_k| / |v_{k+2}-v_{k+1}|) for successive mesh doublings",
        ),
        checks=dict(
            patch_test_rel_err=patch_err,
            rigid_body_translation_rel_err=rigid_err,
            rectangle_2x2_vs_3x3_rel_err=quad_err,
            n_run_rows=len(run_rows),
            n_profile_rows=len(profile_rows),
            n_gradient_rows=len(grad_rows),
            n_probe_rows=len(probe_rows),
            max_reaction_rel_err=max_reaction_err,
            u_mid_80_160_gap_pct=u_mid_gap,
        ),
        cases=summary_cases,
        files=["results/q4_runs.csv", "results/q4_profiles.csv",
               "results/q4_gradient.csv", "results/q4_probe.csv",
               "results/q4_probe_single.csv", "results/q4_summary.json",
               "results/q4_method_note.md"],
    )
    save_json(str(RESULTS / "q4_summary.json"), clean(summary))

    write_method_note(summary, per_case, u_mid_gap, patch_err, rigid_err,
                      quad_err, time.time() - t_start)

    print(f"{time.time() - t_start:.1f}s", RESULTS)


def write_method_note(summary, per_case, u_mid_gap, patch_err, rigid_err,
                      quad_err, elapsed):
    L = ["# Q4 结果", ""]
    L.append("k_e = sum_g w_g detJ_g t B_g^T D B_g，2x2 Gauss。")
    L.append("")
    L.append("| case | eta | mesh | u_end/m | e_end/% | u_mid/m | e_mid/% | sigma RMS/% | grad RMS/% |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for case in summary["cases"]:
        c = summary["cases"][case]
        m = c["meshes"][-1]
        sp = m["sigma_profile_rms_pct"]
        gp = m["grad_profile_rms_pct"]
        sp = f"{sp:.4f}" if sp is not None else "-"
        gp = f"{gp:.3f}" if gp is not None else "-"
        L.append(f"| {case} | {c['eta']:.2f} | {m['nx']}x{m['ny']} | "
                 f"{m['u_end']:.5e} | {m['e_end_pct']:+.3f} | {m['u_mid']:.5e} | "
                 f"{m['e_mid_pct']:+.3f} | {sp} | {gp} |")
    L.append("")
    L.append("## 自检")
    L.append(f"- patch {patch_err:.2e}, rigid {rigid_err:.2e}, 2x2_vs_3x3 {quad_err:.2e}")
    L.append(f"- reaction {summary['checks']['max_reaction_rel_err']:.2e}")
    L.append("")
    L.append("## 文件")
    L.append("- results/q4_runs.csv, q4_profiles.csv, q4_gradient.csv, q4_probe.csv")
    text = "\n".join(L) + "\n"
    (RESULTS / "q4_method_note.md").write_text(text, encoding="utf-8")
    return text


def dev_text(rows, u_ref):
    devs = [abs(float(r["u_mid"]) - u_ref) / u_ref * 100.0 for r in rows]
    return " → ".join(f"{d:.3f}%" for d in devs)


if __name__ == "__main__":
    main()
