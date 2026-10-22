"""CST study driver: explicit CST stiffness, stress recovery, diagonal sensitivity."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from fem.config import (CASES, MESH_LEVELS, STATIONS, CST_DIAGONALS)
from fem.mesh import make_mesh, mesh_quality
from fem import analytic as an
from fem import cst
from fem.io import save_csv, save_json

RESULTS = os.path.join(PROJ, "results")

RUNS_COLS = ["case", "nx", "ny", "diag", "u_end", "u_mid", "e_end_pct",
             "e_mid_pct", "sigma_mid", "e_sigma_mid_pct",
             "sigma_profile_rms_pct", "reaction_x", "note"]
PROFILE_COLS = ["case", "nx", "ny", "diag", "x", "u_section_avg",
                "sigma_section_avg_raw", "sigma_section_avg_recovered",
                "sigma_nom"]
DIAG_COLS = ["case", "nx", "ny", "u_end_pos", "u_end_neg", "u_mid_pos",
             "u_mid_neg", "d_u_pct", "d_u_mid_pct", "sigma_mid_pos",
             "sigma_mid_neg", "d_sigma_mid_pct", "sigma_end_pos",
             "sigma_end_neg", "d_sigma_axial_pct", "d_sigma_element_pct",
             "d_sigma_nodal_pct", "max_nodal_recovered_diff_pct",
             "probe_max_rel_pct", "probe_rms_rel_pct",
             "cv_max_raw_pos", "cv_max_raw_neg",
             "cv_max_recovered_pos", "cv_max_recovered_neg"]
PROBE_COLS = ["case", "nx", "ny", "diag", "x", "y", "ux", "uy",
              "sxx", "syy", "sxy"]
DISP_COLS = ["case", "nx", "ny", "diag", "x", "sigma_mean_raw",
             "sigma_mean_recovered", "cv_raw", "cv_recovered",
             "cv_recovered_arith"]

PROBE_XS = (0.1, 0.2, 0.5, 0.8, 0.9)


def _rel_pct(a, b):
    den = max(abs(float(a)), abs(float(b)), 1e-300)
    return 100.0 * abs(float(a) - float(b)) / den


def run_case(case, runs, profiles, diagonals, probes, dispersion, trend,
             checks):
    """Solve both diagonals for every mesh level of one case."""
    g, mat, P = case.geom, case.material, case.load
    ue_exact = an.exact_u_end(g, mat, P)
    um_exact = an.exact_u_mid(g, mat, P)
    for nx, ny in MESH_LEVELS:
        mesh = make_mesh(g, nx, ny)
        q = mesh_quality(mesh)
        checks["min_tri_area"] = min(checks["min_tri_area"],
                                     q["min_tri_area_pos"], q["min_tri_area_neg"])
        # nominal sigma at the section stations (x = 0, 0.1, ..., 1.0 in L units)
        res = {}
        prof = {}
        nom = {}
        acc = {}          # per-diagonal scalars, promoted to per-mesh trends below
        for diag in CST_DIAGONALS:
            r = cst.solve_cst(g, mat, P, nx, ny, diag, mesh=mesh)
            res[diag] = r
            prof[diag] = cst.section_profile(r)
            nom[diag] = an.exact_sigma(g, mat, P, prof[diag]["x_m"])
            p = prof[diag]
            imid = int(np.argmin(np.abs(p["x"] - 0.5)))
            sigma_mid = float(p["sigma_section_avg_raw"][imid])
            sigma_mid_nom = float(nom[diag][imid])
            rms_raw = cst.profile_rms_pct(p["sigma_section_avg_raw"], p["x"],
                                          nom[diag])
            rms_rec = cst.profile_rms_pct(p["sigma_section_avg_recovered"],
                                          p["x"], nom[diag])
            sel = (p["x"] >= 0.1 - 1e-12) & (p["x"] <= 0.9 + 1e-12)
            cv_raw_max = float(np.nanmax(p["cv_raw"][sel]))
            cv_rec_max = float(np.nanmax(p["cv_recovered"][sel]))
            e_end = 100.0 * (r["u_end"] - ue_exact) / ue_exact
            e_mid = 100.0 * (r["u_mid"] - um_exact) / um_exact
            e_sig = 100.0 * (sigma_mid - sigma_mid_nom) / sigma_mid_nom
            checks["reaction_rel_err"] = max(
                checks["reaction_rel_err"], abs(r["reaction"] + P) / P)
            runs.append(dict(
                case=case.key, nx=nx, ny=ny, diag=diag,
                u_end=r["u_end"], u_mid=r["u_mid"],
                e_end_pct=e_end, e_mid_pct=e_mid,
                sigma_mid=sigma_mid, e_sigma_mid_pct=e_sig,
                sigma_profile_rms_pct=rms_raw,
                reaction_x=r["reaction"],
                note="cv_raw_max=%.3e;rec_rms%%=%.3f" % (cv_raw_max, rms_rec),
            ))
            for k in range(p["x"].size):
                profiles.append(dict(
                    case=case.key, nx=nx, ny=ny, diag=diag,
                    x=float(p["x_m"][k]),
                    u_section_avg=float(p["u_section_avg"][k]),
                    sigma_section_avg_raw=float(p["sigma_section_avg_raw"][k]),
                    sigma_section_avg_recovered=float(
                        p["sigma_section_avg_recovered"][k]),
                    sigma_nom=float(nom[diag][k]),
                ))
                dispersion.append(dict(
                    case=case.key, nx=nx, ny=ny, diag=diag,
                    x=float(p["x_m"][k]),
                    sigma_mean_raw=float(p["sigma_section_avg_raw"][k]),
                    sigma_mean_recovered=float(
                        p["sigma_section_avg_recovered"][k]),
                    cv_raw=float(p["cv_raw"][k]),
                    cv_recovered=float(p["cv_recovered"][k]),
                    cv_recovered_arith=float(p["cv_recovered_arith"][k]),
                ))
            probes.extend(cst.probe_rows(r, case, xs=PROBE_XS))
            acc[diag] = dict(
                cv_raw=cv_raw_max, cv_recovered=cv_rec_max,
                cv_arith=float(np.nanmax(p["cv_recovered_arith"][sel])),
                rms_raw=rms_raw, rms_recovered=rms_rec,
                e_end=e_end, e_mid=e_mid,
                u_end=r["u_end"], u_mid=r["u_mid"],
            )
        dm = cst.diagonal_metrics(res["pos"], res["neg"], nom["pos"])
        rows_p = cst.probe_rows(res["pos"], case, xs=PROBE_XS)
        rows_n = cst.probe_rows(res["neg"], case, xs=PROBE_XS)
        offax = [(a, b) for a, b in zip(rows_p, rows_n)
                 if abs(a["y"]) > 1e-12 * g.L]
        rel = np.array([_rel_pct(a["sxx"], b["sxx"]) for a, b in offax])
        probe_max = float(rel.max()) if rel.size else np.nan
        probe_rms = float(np.sqrt(np.mean(rel ** 2))) if rel.size else np.nan
        # mirror-plane probes must be identical (sigma_xx is even in y)
        mirror_rel = max([_rel_pct(a["sxx"], b["sxx"])
                          for a, b in zip(rows_p, rows_n)
                          if abs(a["y"]) <= 1e-12 * g.L] or [0.0])
        checks["probe_mirror_plane_rel_err"] = max(
            checks["probe_mirror_plane_rel_err"], mirror_rel)
        checks["d_u_pct"] = max(checks["d_u_pct"], dm["d_u_pct"])
        checks["d_sigma_axial_pct"] = max(checks["d_sigma_axial_pct"],
                                          dm["d_sigma_axial_pct"])
        diagonals.append(dict(
            case=case.key, nx=nx, ny=ny,
            u_end_pos=dm["u_end_pos"], u_end_neg=dm["u_end_neg"],
            u_mid_pos=dm["u_mid_pos"], u_mid_neg=dm["u_mid_neg"],
            d_u_pct=dm["d_u_pct"], d_u_mid_pct=dm["d_u_mid_pct"],
            sigma_mid_pos=dm["sigma_mid_pos"],
            sigma_mid_neg=dm["sigma_mid_neg"],
            d_sigma_mid_pct=dm["d_sigma_mid_pct"],
            sigma_end_pos=dm["sigma_end_pos"],
            sigma_end_neg=dm["sigma_end_neg"],
            d_sigma_axial_pct=dm["d_sigma_axial_pct"],
            d_sigma_element_pct=dm["d_sigma_element_pct"],
            d_sigma_nodal_pct=dm["d_sigma_nodal_pct"],
            max_nodal_recovered_diff_pct=dm["max_nodal_recovered_diff_pct"],
            probe_max_rel_pct=probe_max, probe_rms_rel_pct=probe_rms,
            cv_max_raw_pos=dm["cv_max_raw_pos"],
            cv_max_raw_neg=dm["cv_max_raw_neg"],
            cv_max_recovered_pos=dm["cv_max_recovered_pos"],
            cv_max_recovered_neg=dm["cv_max_recovered_neg"],
        ))
        # per-mesh trends use the pos solution (the neg global quantities are
        # identical to round-off; the diagonal sensitivity is stored above)
        for key in ("cv_raw", "cv_recovered", "cv_arith", "rms_raw",
                    "rms_recovered", "e_end", "e_mid", "u_end", "u_mid"):
            trend[key].append(acc["pos"][key])
        trend["d_sigma_element"].append(dm["d_sigma_element_pct"])
        trend["d_sigma_nodal"].append(dm["d_sigma_nodal_pct"])
        trend["probe_max"].append(probe_max)
        trend["probe_rms"].append(probe_rms)
        trend["d_u"].append(dm["d_u_pct"])
        trend["d_sigma_axial"].append(dm["d_sigma_axial_pct"])
        trend["max_nodal"].append(dm["max_nodal_recovered_diff_pct"])
    return None


def _order(trend, key):
    """Observed convergence order of a decreasing local metric (h -> h/2)."""
    vals = [v for v in trend[key] if v == v]
    if len(vals) < 2 or vals[-1] <= 0:
        return float("nan")
    return float(np.log(vals[-2] / vals[-1]) / np.log(2.0))


def case_summary(case, trend):
    g = case.geom
    return dict(
        case=case.key, label=case.label, beta=g.beta, h0=g.h0,
        eta=g.eta, taper_angle_deg=g.taper_angle_deg,
        meshes=[list(m) for m in MESH_LEVELS],
        u_end=trend["u_end"], u_mid=trend["u_mid"],
        e_end_pct=trend["e_end"], e_mid_pct=trend["e_mid"],
        sigma_profile_rms_pct_raw=trend["rms_raw"],
        sigma_profile_rms_pct_recovered=trend["rms_recovered"],
        d_u_pct=trend["d_u"],
        d_sigma_axial_pct=trend["d_sigma_axial"],
        d_sigma_element_pct=trend["d_sigma_element"],
        d_sigma_nodal_pct=trend["d_sigma_nodal"],
        max_nodal_recovered_diff_pct=trend["max_nodal"],
        probe_max_rel_pct=trend["probe_max"],
        probe_rms_rel_pct=trend["probe_rms"],
        cv_max_raw=trend["cv_raw"],
        cv_max_recovered=trend["cv_recovered"],
        cv_max_arithmetic=trend["cv_arith"],
        order_d_sigma_element=_order(trend, "d_sigma_element"),
        order_d_sigma_nodal=_order(trend, "d_sigma_nodal"),
        order_probe_rms=_order(trend, "probe_rms"),
        order_cv_raw=_order(trend, "cv_raw"),
        cv_ratio_recovered_over_raw_coarse=(
            trend["cv_recovered"][0] / trend["cv_raw"][0]),
        cv_ratio_recovered_over_raw_fine=(
            trend["cv_recovered"][-1] / trend["cv_raw"][-1]),
        e_end_pct_coarse=trend["e_end"][0],
        e_end_pct_fine=trend["e_end"][-1],
        e_mid_pct_coarse=trend["e_mid"][0],
        e_mid_pct_fine=trend["e_mid"][-1],
    )


def rigid_body_check(case):
    """Uniform translation must produce zero element stress."""
    mesh = make_mesh(case.geom, 20, 4)
    tris = mesh.tris_pos
    xy, det, area = cst.element_geometry(mesh, tris)
    B = cst.cst_B_batch(xy, det)
    D = case.material.D_plane_stress()
    U = np.zeros(mesh.ndof)
    U[0::2] = 1.0
    U[1::2] = -2.0
    eps, sigma = cst.cst_stress_batch(B, D, U, tris)
    return float(np.max(np.abs(eps))), float(np.max(np.abs(sigma)))


def build_note(case_sum, checks, elapsed):
    L = ["# CST 结果", ""]
    L.append("k_e = t |A| B^T D B；sigma_e = D B u_e。")
    L.append("")
    L.append("| case | eta | d_el/% | d_node/% | CV_raw/% | CV_rec/% | e_end/% |")
    L.append("|---|---|---|---|---|---|---|")
    for case, c in case_sum.items():
        L.append(f"| {case} | {c['eta']:.2f} | {c['d_sigma_element_pct'][-1]:.3f} | "
                 f"{c['d_sigma_nodal_pct'][-1]:.3f} | {100*c['cv_max_raw'][-1]:.2f} | "
                 f"{100*c['cv_max_recovered'][-1]:.2f} | {c['e_end_pct'][-1]:+.3f} |")
    L.append("")
    L.append("## 自检")
    L.append(f"- reaction {checks['reaction_rel_err']:.2e}, "
             f"mirror d_u {checks['d_u_pct']:.2e}, "
             f"S0 constant {checks['s0_constant_stress_rel_err']:.2e}")
    L.append("")
    L.append("## 文件")
    L.append("- results/cst_runs.csv, cst_profiles.csv, cst_diagonal.csv, cst_probe.csv, cst_dispersion.csv")
    return "\n".join(L) + "\n"


def main():
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    runs, profiles, diagonals, probes, dispersion = [], [], [], [], []
    checks = dict(min_tri_area=np.inf, reaction_rel_err=0.0,
                  d_u_pct=0.0, d_sigma_axial_pct=0.0,
                  probe_mirror_plane_rel_err=0.0,
                  s0_constant_stress_rel_err=0.0)
    case_sum = {}
    for case in CASES:
        trend = {k: [] for k in ("cv_raw", "cv_recovered", "cv_arith",
                                 "rms_raw", "rms_recovered", "e_end",
                                 "e_mid", "u_end", "u_mid",
                                 "d_sigma_element", "d_sigma_nodal",
                                 "probe_max", "probe_rms", "d_u",
                                 "d_sigma_axial", "max_nodal")}
        run_case(case, runs, profiles, diagonals, probes, dispersion,
                 trend, checks)
        case_sum[case.key] = case_summary(case, trend)
    case0 = [c for c in CASES if c.key == "S0"][0]
    dev = 0.0
    for diag in CST_DIAGONALS:
        for nx, ny in MESH_LEVELS:
            r = cst.solve_cst(case0.geom, case0.material, case0.load, nx, ny,
                              diag)
            p = cst.section_profile(r)
            nom = an.exact_sigma(case0.geom, case0.material, case0.load,
                                 p["x_m"])
            sel = p["x"] >= 0.1 - 1e-12
            dev = max(dev, float(np.max(np.abs(
                p["sigma_section_avg_raw"][sel] - nom[sel]) / nom[sel])))
    checks["s0_constant_stress_rel_err"] = dev
    eps_rb, sig_rb = rigid_body_check(case0)
    checks["rigid_eps"] = eps_rb
    checks["rigid_sigma"] = sig_rb
    case1 = [c for c in CASES if c.key == "S1"][0]
    nx1, ny1 = MESH_LEVELS[-1]
    r1 = cst.solve_cst(case1.geom, case1.material, case1.load, nx1, ny1, "pos")
    p1 = cst.section_profile(r1)
    nom1 = an.exact_sigma(case1.geom, case1.material, case1.load, p1["x_m"])
    dev1 = 100.0 * (p1["sigma_section_avg_recovered"] - nom1) / nom1
    k09 = int(np.argmin(np.abs(p1["x"] - 0.9)))
    checks["s1_rec_dev_x09_fine"] = float(dev1[k09])
    checks["s1_rec_dev_x10_fine"] = float(dev1[-1])
    assert checks["min_tri_area"] > 0.0, "non-positive triangle area"
    assert checks["reaction_rel_err"] < 1e-10, checks["reaction_rel_err"]
    assert checks["s0_constant_stress_rel_err"] < 1e-9
    assert checks["d_u_pct"] < 1e-8, checks["d_u_pct"]
    assert checks["d_sigma_axial_pct"] < 1e-8
    assert checks["probe_mirror_plane_rel_err"] < 1e-9
    assert checks["rigid_sigma"] < 1e-2, checks["rigid_sigma"]
    for c in case_sum.values():
        assert c["cv_max_raw"][-1] < c["cv_max_raw"][0], c["case"]
        assert c["d_sigma_element_pct"][-1] < 0.6 * c["d_sigma_element_pct"][0]
        assert c["probe_max_rel_pct"][-1] < 5.0, c["case"]
    save_csv(os.path.join(RESULTS, "cst_runs.csv"), runs, RUNS_COLS)
    save_csv(os.path.join(RESULTS, "cst_profiles.csv"), profiles, PROFILE_COLS)
    save_csv(os.path.join(RESULTS, "cst_diagonal.csv"), diagonals, DIAG_COLS)
    save_csv(os.path.join(RESULTS, "cst_probe.csv"), probes, PROBE_COLS)
    save_csv(os.path.join(RESULTS, "cst_dispersion.csv"), dispersion,
             DISP_COLS)
    summary = dict(
        method="CST (3-node constant strain triangle), plane stress",
        element_formulation=dict(
            B="1/(2A)*[[b1,0,b2,0,b3,0],[0,c1,0,c2,0,c3],[c1,b1,c2,b2,c3,b3]]",
            ke="t*|A|*B^T*D*B  (explicit, no numerical integration)",
            D="E/(1-nu^2)*[[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]",
            thickness="b = 0.1 m",
            assembly="shared fem.assembly.assemble_sparse / dirichlet_solve",
            load="shared end_load_vector default mode='consistent' "
                 "(consistent nodal loads of the uniform axial traction, total P)",
        ),
        mesh_levels=[list(m) for m in MESH_LEVELS],
        diagonals=list(CST_DIAGONALS),
        conventions=dict(
            u_end="arithmetic mean of ux on the right-end node column",
            u_mid="arithmetic mean of ux on the x=L/2 node column",
            x_units="x in metres (L=1, so numerically equal to x/L)",
            sigma_mid="raw section average at x=L/2",
            section_avg_raw=(
                "equal-weight mean of the element constant sigma_x over the "
                "triangles incident to the station node column (centred: both "
                "adjacent element columns for interior stations, the single "
                "adjacent column at x=0 and x=L)"),
            section_avg_recovered=(
                "equal-weight mean of the area-weighted recovered nodal "
                "sigma_x over the station node column"),
            sigma_nom="P/A(x) from fem.analytic.exact_sigma",
            probe_stress=(
                "area-weighted mean of the constant element stresses of all "
                "triangles containing the point (equals the element constant "
                "stress for points strictly inside one triangle; local element "
                "average at nodes/edges); displacement from the linear shape "
                "functions of the containing triangle"),
            sigma_profile_rms_pct=(
                "100*sqrt(mean((s-s_nom)^2))/sqrt(mean(s_nom^2)) over "
                "x=0.1..0.9; runs.csv uses the raw section average"),
            cv="std/|mean| of the values entering the section average "
               "(raw: element stresses; recovered: nodal recovered values)",
        ),
        metrics=dict(
            d_u_pct="100*|u_end_pos-u_end_neg|/mean(|u_end_pos|,|u_end_neg|)",
            d_u_mid_pct="same for u_mid",
            d_sigma_axial_pct=(
                "100*RMS_x(sigma_pos-sigma_neg)/RMS_x(sigma_nom), x=0.1..0.9, "
                "raw section averages"),
            d_sigma_mid_pct=(
                "100*|sigma_mid_pos-sigma_mid_neg|/mean of the two"),
            sigma_end="raw section average at x=L",
            d_sigma_element_pct=(
                "100*RMS_e(|sxx_pos-sxx_neg|)/RMS_e(sxx_nom), e over all "
                "triangles, same element index (same quad half)"),
            d_sigma_nodal_pct=(
                "100*RMS_n(|sxx_node_pos-sxx_node_neg|)/mean_n(sxx_node_pos), "
                "area-weighted recovered nodal values"),
            max_nodal_recovered_diff_pct=(
                "100*max_n(|sxx_node_pos-sxx_node_neg|)/mean_n"),
            probe_max_rel_pct=(
                "max over off-axis probe points (x=0.1,0.2,0.5,0.8,0.9; "
                "y=+-h/4) of 100*|sxx_pos-sxx_neg|/max(|sxx_pos|,|sxx_neg|)"),
            probe_rms_rel_pct="RMS of the same relative differences",
            mirror_note=(
                "pos/neg are mirror images under y->-y: u_end, u_mid and the "
                "section-averaged sigma_x are invariants (expected ~1e-11 %); "
                "the local metrics quantify the departure from mirror symmetry "
                "of the discrete local fields."),
        ),
        self_checks=checks,
        cases=case_sum,
        files=[
            "results/cst_runs.csv", "results/cst_profiles.csv",
            "results/cst_diagonal.csv", "results/cst_probe.csv",
            "results/cst_dispersion.csv", "results/cst_summary.json",
            "results/cst_method_note.md",
        ],
        elapsed_s=round(time.time() - t0, 2),
    )
    save_json(os.path.join(RESULTS, "cst_summary.json"), summary)
    note = build_note(case_sum, checks, time.time() - t0)
    with open(os.path.join(RESULTS, "cst_method_note.md"), "w",
              encoding="utf-8") as f:
        f.write(note)
    print("cst %.2fs" % (time.time() - t0))
    print("checks", json.dumps(
        {k: (round(v, 15) if isinstance(v, float) else v)
         for k, v in checks.items()}, ensure_ascii=False))
    print()
    print("%-4s %-8s %-13s %-9s %-13s %-9s %-8s %-8s %-8s"
          % ("case", "mesh", "u_end", "e_end[%]", "u_mid", "e_mid[%]",
             "rms_raw%", "d_el%", "d_u%"))
    for c in case_sum.values():
        for j, (nx, ny) in enumerate(MESH_LEVELS):
            print("%-4s %-8s %-13.6e %+9.3f %-13.6e %+9.3f %8.3f %8.3f %8.1e"
                  % (c["case"], "%dx%d" % (nx, ny), c["u_end"][j],
                     c["e_end_pct"][j], c["u_mid"][j], c["e_mid_pct"][j],
                     c["sigma_profile_rms_pct_raw"][j],
                     c["d_sigma_element_pct"][j], c["d_u_pct"][j]))
    print()
    print("diag:")
    for c in case_sum.values():
        print("  %s: d_el=%.3f%% d_nod=%.3f%% probe_max=%.2f%% "
              "max_nodal=%.1f%% order_el=%.2f"
              % (c["case"], c["d_sigma_element_pct"][-1],
                 c["d_sigma_nodal_pct"][-1], c["probe_max_rel_pct"][-1],
                 c["max_nodal_recovered_diff_pct"][-1],
                 c["order_d_sigma_element"]))
    print()
    print("files:", ", ".join(summary["files"]))
    return runs, profiles, diagonals, probes, dispersion, summary


if __name__ == "__main__":
    main()
