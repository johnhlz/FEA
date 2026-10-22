"""CalculiX anchor runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anchors.calculix import ccx_io as C
from fem.analytic import exact_sigma, exact_u_end, exact_u_mid
from fem.config import CASES, CASES_BY_KEY
from fem.io import load_csv, save_csv, save_json

RUNS_ROOT = ROOT / "anchors" / "calculix" / "runs"
RESULTS = ROOT / "results"

# (element_type, nx, ny, role)
RUN_PLAN = [
    ("CPS8", 20, 4, "anchor_cps8"),
    ("CPS8", 40, 8, "anchor_cps8"),
    ("CPS8", 80, 16, "reference_convergence"),
    ("CPS8", 160, 32, "reference_convergence"),
    ("CPS8", 240, 48, "reference_selected"),
    ("CPS4", 20, 4, "anchor_cps4"),
    ("CPS4", 40, 8, "anchor_cps4"),
    ("CPS4", 80, 16, "anchor_cps4"),
    ("CPS4", 160, 32, "anchor_cps4"),
]
CONVERGENCE_MESHES = ((80, 16), (160, 32), (240, 48))
REFERENCE_MESH = (240, 48)
SENSITIVITY_MESHES = ((80, 16), (160, 32), (240, 48))

ANCHOR_FIELDS = ["case", "mesh", "element_type", "nnodes", "nelem", "u_end",
                 "u_mid", "sigma_mid", "sigma_mid_exact", "e_end_pct",
                 "e_mid_pct", "e_sigma_pct", "note"]
PROFILE_FIELDS = ["case", "mesh", "element_type", "x", "u_section_avg",
                  "sigma_section_avg", "sigma_nom"]
POINT_FIELDS = ["case", "x", "y", "ux", "uy", "sxx", "syy", "sxy", "mesh",
                "element_type"]


def _f(x):
    return float(x)


def _probe_row(p):
    out = {}
    for k, v in p.items():
        out[k] = float(v) if isinstance(v, (int, float, np.floating)) else v
    return out


def _profile_row(p):
    return {k: float(v) for k, v in p.items()}


def run_one(case, nx, ny, et, load_mode, keep_dat, timeout, runs_root):
    t0 = time.time()
    res = C.run_case(case, nx, ny, et, str(runs_root), load_mode=load_mode,
                     timeout=timeout, keep_dat=keep_dat, keep_frd=False)
    dt = time.time() - t0
    tag = f"{case.key} {et} {nx}x{ny} [{load_mode}]"
    uex = exact_u_end(case.geom, case.material, case.load)
    umx = exact_u_mid(case.geom, case.material, case.load)
    smx = _f(exact_sigma(case.geom, case.material, case.load, 0.5 * case.geom.L))
    e_end = 100.0 * (res["u_end"] / uex - 1.0)
    e_mid = 100.0 * (res["u_mid"] / umx - 1.0)
    e_sig = 100.0 * (res["sigma_mid"] / smx - 1.0)
    react_err = abs(res["reaction_x"] + case.load) / case.load
    print(f"  {tag:<38s} {dt:6.1f}s  u_end={res['u_end']:.4e} "
          f"u_mid={res['u_mid']:.4e} s_mid={res['sigma_mid']:9.2f} "
          f"R={res['reaction_x']:10.4f} (rel {react_err:.1e})")
    if react_err > 1e-6:
        raise AssertionError(f"reaction check failed for {tag}: {res['reaction_x']}")
    return dict(res=res, u_end=_f(res["u_end"]), u_mid=_f(res["u_mid"]),
                sigma_mid=_f(res["sigma_mid"]), u_end_exact=_f(uex),
                u_mid_exact=_f(umx), sigma_mid_exact=_f(smx),
                e_end_pct=_f(e_end), e_mid_pct=_f(e_mid),
                e_sigma_pct=_f(e_sig), reaction_x=_f(res["reaction_x"]),
                reaction_rel_err=_f(react_err), nnodes=int(res["nnodes"]),
                nelem=int(res["nelem"]), n_ip=int(res["n_ip"]),
                profiles=[_profile_row(p) for p in res["profiles"]],
                probes=[_probe_row(p) for p in res["probes"]],
                wall_time_s=_f(res["wall_time_s"]), jobdir=res["jobdir"])


def compare_inhouse(runs, results_dir):
    """Compare ccx runs with in-house q4/cst/bar CSVs when they exist."""
    rows = []
    status = {}
    for source, fname, et in (("q4", "q4_runs.csv", "CPS4"),
                              ("cst", "cst_runs.csv", "CPS4")):
        path = results_dir / fname
        if not path.is_file():
            status[source] = "missing"
            continue
        try:
            data = load_csv(str(path))
        except Exception as exc:                       # pragma: no cover
            status[source] = f"unreadable: {exc}"
            continue
        n = 0
        for row in data:
            try:
                case_key = row["case"].strip()
                nx, ny = int(float(row["nx"])), int(float(row["ny"]))
                u_in = float(row["u_end"])
                s_in = float(row["sigma_mid"])
            except (KeyError, ValueError):
                continue
            r = runs.get((case_key, et, nx, ny))
            if r is None:
                continue
            rows.append(dict(
                source=source, case=case_key, mesh=f"{nx}x{ny}",
                u_end_ccx=r["u_end"], u_end_inhouse=u_in,
                rel_diff_u_end_pct=100.0 * (u_in - r["u_end"]) / r["u_end"],
                sigma_mid_ccx=r["sigma_mid"], sigma_mid_inhouse=s_in,
                rel_diff_sigma_mid_pct=100.0 * (s_in - r["sigma_mid"]) / r["sigma_mid"]))
            n += 1
        status[source] = f"{n} matched rows"

    path = results_dir / "bar_runs.csv"
    if not path.is_file():
        status["bar"] = "missing"
    else:
        try:
            data = load_csv(str(path))
            n = 0
            for row in data:
                try:
                    case_key = row["case"].strip()
                    n_elem = int(float(row["n_elem"]))
                    u_in = float(row["u_end"])
                    s_in = float(row["sigma_mid"])
                except (KeyError, ValueError):
                    continue
                cand = [k for k in runs
                        if k[0] == case_key and k[1] == "CPS8" and k[2] == n_elem]
                if not cand:
                    continue
                k = cand[0]
                r = runs[k]
                rows.append(dict(
                    source="bar", case=case_key,
                    mesh=f"1D n={n_elem} (ccx {n_elem}x{k[3]})",
                    u_end_ccx=r["u_end"], u_end_inhouse=u_in,
                    rel_diff_u_end_pct=100.0 * (u_in - r["u_end"]) / r["u_end"],
                    sigma_mid_ccx=r["sigma_mid"], sigma_mid_inhouse=s_in,
                    rel_diff_sigma_mid_pct=100.0 * (s_in - r["sigma_mid"]) / r["sigma_mid"]))
                n += 1
            status["bar"] = f"{n} matched rows"
        except Exception as exc:                        # pragma: no cover
            status["bar"] = f"unreadable: {exc}"
    return rows, status


def _probe_map(row):
    return {(round(p["x"], 9), round(p["y"], 9)): p for p in row["probes"]}


def convergence_check(runs, case_key):
    meshes = [f"{nx}x{ny}" for nx, ny in CONVERGENCE_MESHES]
    ref_key = (case_key, "CPS8", *REFERENCE_MESH)
    if ref_key not in runs:
        return None
    ref = runs[ref_key]
    out = {"meshes": meshes, "u_end": {}, "u_mid": {},
           "u_end_rel_diff_pct_vs_ref": {}, "u_mid_rel_diff_pct_vs_ref": {},
           "probe_sxx": []}
    for nx, ny in CONVERGENCE_MESHES:
        tag = f"{nx}x{ny}"
        r = runs.get((case_key, "CPS8", nx, ny))
        if r is None:
            continue
        out["u_end"][tag] = r["u_end"]
        out["u_mid"][tag] = r["u_mid"]
        out["u_end_rel_diff_pct_vs_ref"][tag] = 100.0 * (r["u_end"] - ref["u_end"]) / ref["u_end"]
        out["u_mid_rel_diff_pct_vs_ref"][tag] = 100.0 * (r["u_mid"] - ref["u_mid"]) / ref["u_mid"]
    if (case_key, "CPS8", *CONVERGENCE_MESHES[-1]) in runs:
        ref_probes = _probe_map(ref)
        for key in sorted(ref_probes):
            vals, diffs = {}, {}
            for nx, ny in CONVERGENCE_MESHES:
                r = runs.get((case_key, "CPS8", nx, ny))
                if r is None:
                    continue
                p = _probe_map(r).get(key)
                if p is None:
                    continue
                tag = f"{nx}x{ny}"
                vals[tag] = p["sxx"]
                diffs[tag] = 100.0 * (p["sxx"] - ref_probes[key]["sxx"]) / ref_probes[key]["sxx"]
            out["probe_sxx"].append(dict(x=key[0], y=key[1], values=vals,
                                         rel_diff_pct_vs_ref=diffs))
    out["selected_mesh"] = f"{REFERENCE_MESH[0]}x{REFERENCE_MESH[1]}"
    du = out["u_end_rel_diff_pct_vs_ref"].get("160x32")
    dm = out["u_mid_rel_diff_pct_vs_ref"].get("160x32")
    if du is not None:
        out["reason"] = ("240x48 selected as reference mesh; 160x32 -> 240x48 "
                         f"change of u_end is {du:+.3f}% (u_mid {dm:+.4f}%)")
    else:
        out["reason"] = "240x48 selected as reference mesh"
    return out


def load_mode_sensitivity(primary, variant, case_key):
    """Consistent uniform traction (primary) vs lumped equal nodal force."""
    cn = primary.get((case_key, "CPS8", *REFERENCE_MESH))
    eq = variant.get((case_key, "CPS8", *REFERENCE_MESH))
    if cn is None or eq is None:
        return None
    pcn, peq = _probe_map(cn), _probe_map(eq)
    probe_diff = []
    for key in sorted(pcn):
        if key not in peq:
            continue
        probe_diff.append(dict(
            x=key[0], y=key[1],
            sxx_consistent=pcn[key]["sxx"], sxx_equal_nodal=peq[key]["sxx"],
            diff_pct=100.0 * (peq[key]["sxx"] / pcn[key]["sxx"] - 1.0)))
    return dict(
        mesh=f"{REFERENCE_MESH[0]}x{REFERENCE_MESH[1]}",
        u_end_consistent=cn["u_end"], u_end_equal_nodal=eq["u_end"],
        u_end_diff_pct=100.0 * (eq["u_end"] / cn["u_end"] - 1.0),
        u_mid_consistent=cn["u_mid"], u_mid_equal_nodal=eq["u_mid"],
        u_mid_diff_pct=100.0 * (eq["u_mid"] / cn["u_mid"] - 1.0),
        sigma_mid_consistent=cn["sigma_mid"], sigma_mid_equal_nodal=eq["sigma_mid"],
        sigma_mid_diff_pct=100.0 * (eq["sigma_mid"] / cn["sigma_mid"] - 1.0),
        probe_sxx=probe_diff)


def build_reference_json(runs, sens, args):
    cases = {}
    for case in CASES:
        ref = runs.get((case.key, "CPS8", *REFERENCE_MESH))
        if ref is None:
            raise RuntimeError(f"reference run missing for {case.key}")
        cases[case.key] = dict(
            mesh=f"{REFERENCE_MESH[0]}x{REFERENCE_MESH[1]}",
            element_type="CPS8",
            load_mode="consistent_uniform_traction",
            u_end=ref["u_end"], u_mid=ref["u_mid"],
            u_end_exact_1d=ref["u_end_exact"], u_mid_exact_1d=ref["u_mid_exact"],
            sigma_mid=ref["sigma_mid"], sigma_mid_exact_1d=ref["sigma_mid_exact"],
            profile=[dict(x=p["x"], u_section_avg=p["u_section_avg"],
                          sigma_section_avg=p["sigma_section_avg"],
                          sigma_nom=p["sigma_nom"]) for p in ref["profiles"]],
            convergence_check=convergence_check(runs, case.key),
            load_mode_sensitivity=load_mode_sensitivity(runs, sens, case.key),
        )
    return dict(
        generated_by="scripts/run_ccx.py",
        ccx_version="CalculiX 2.23",
        model=dict(geometry="symmetric wedge h(x)=h0(1+beta*x/L), plane stress",
                   thickness_m=0.1, E_Pa=2.0e11, nu=0.3, P_N=1000.0,
                   element_types=["CPS4", "CPS8"],
                   fixed_end="all nodes x=0, ux=uy=0 (midside nodes included)",
                   loaded_end="consistent nodal loads of a uniform axial traction, "
                              "total P=1000 N (midside nodes included; same as "
                              "fem.assembly.end_load_vector(mode='consistent'))",
                   note="the legacy lumped equal-nodal-force variant is run on the "
                        "reference meshes as a sensitivity check "
                        "(results/ccx_reference_points_equal_nodal.csv)"),
        section_average_definition=dict(
            u_section_avg="trapezoidal average of ux over the node column x=xi "
                          "(int ux dy / h(x); sorted actual node y-coordinates, end "
                          "weights 0.5, inner weights 1, CPS8 midside nodes included)",
            sigma_section_avg="element sxx first averaged over the CalculiX integration "
                              "points (CPS4: 8 IPs from C3D8 expansion, CPS8: 27 IPs from "
                              "C3D20 expansion), then equally averaged over the element "
                              "columns adjacent to the node column (columns k-1 and k; one "
                              "column at x=0 and x=L)"),
        reference_mesh=f"{REFERENCE_MESH[0]}x{REFERENCE_MESH[1]}",
        reference_element_type="CPS8",
        cases=cases)


def render_note(json_obj, anchor_rows, compare_rows, compare_status, args):
    L = ["# CalculiX 参考", ""]
    L.append("CPS8 240x48，平面应力，厚度 0.1 m，一致面力。")
    L.append("")
    L.append("| case | u_end/m | u_mid/m | sigma_mid/Pa |")
    L.append("|---|---|---|---|")
    for case in CASES:
        c = json_obj["cases"][case.key]
        L.append(f"| {case.key} | {c['u_end']:.5e} | {c['u_mid']:.5e} | {c['sigma_mid']:.2f} |")
    L.append("")
    L.append("## 锚点")
    L.append("| case | mesh | elem | u_end/m | e_end/% | e_sigma/% |")
    L.append("|---|---|---|---|---|---|")
    for r in anchor_rows:
        if r["mesh"] == "240x48" or r["element_type"] == "CPS4":
            L.append(f"| {r['case']} | {r['mesh']} | {r['element_type']} | "
                     f"{float(r['u_end']):.5e} | {float(r['e_end_pct']):+.3f} | "
                     f"{float(r['e_sigma_pct']):+.4f} |")
    L.append("")
    L.append("## 对比")
    L.append(f"- ccx_vs_inhouse: {compare_status}")
    L.append("")
    L.append("## 文件")
    L.append("- results/ccx_anchor.csv, ccx_profiles.csv, ccx_reference.json, ccx_reference_points.csv")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="CalculiX CPS4/CPS8 plane-stress anchor")
    ap.add_argument("--keep-dat", action="store_true",
                    help="keep the (large) .dat files after parsing")
    ap.add_argument("--no-consistent", action="store_true",
                    help="skip the lumped equal-nodal-force sensitivity runs")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--exe", default=os.environ.get("CCX_EXE", C.CCX_EXE_DEFAULT))
    args = ap.parse_args(argv)

    if not C.ccx_available(args.exe):
        raise SystemExit(f"CalculiX executable not found: {args.exe}")
    os.environ["CCX_EXE"] = args.exe
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print("ccx", args.exe)
    print("runs", RUNS_ROOT)

    runs, sens = {}, {}
    for case in CASES:
        print(case.key, case.label)
        for et, nx, ny, role in RUN_PLAN:
            runs[(case.key, et, nx, ny)] = run_one(
                case, nx, ny, et, "consistent", args.keep_dat, args.timeout, RUNS_ROOT)
            runs[(case.key, et, nx, ny)]["role"] = role
        if not args.no_consistent:
            print("  lumped")
            for nx, ny in SENSITIVITY_MESHES:
                sens[(case.key, "CPS8", nx, ny)] = run_one(
                    case, nx, ny, "CPS8", "equal", args.keep_dat,
                    args.timeout, RUNS_ROOT)

    anchor_rows = []
    for case in CASES:
        for et, nx, ny, role in RUN_PLAN:
            r = runs[(case.key, et, nx, ny)]
            anchor_rows.append(dict(
                case=case.key, mesh=f"{nx}x{ny}", element_type=et,
                nnodes=r["nnodes"], nelem=r["nelem"],
                u_end=r["u_end"], u_mid=r["u_mid"],
                sigma_mid=r["sigma_mid"], sigma_mid_exact=r["sigma_mid_exact"],
                e_end_pct=r["e_end_pct"], e_mid_pct=r["e_mid_pct"],
                e_sigma_pct=r["e_sigma_pct"],
                note=f"{role}; load=consistent_uniform_traction; "
                     f"reaction_rel_err={r['reaction_rel_err']:.1e}"))
    save_csv(str(RESULTS / "ccx_anchor.csv"), anchor_rows, ANCHOR_FIELDS)

    profile_rows = []
    for case in CASES:
        for et, nx, ny, role in RUN_PLAN:
            r = runs[(case.key, et, nx, ny)]
            for p in r["profiles"]:
                profile_rows.append(dict(case=case.key, mesh=f"{nx}x{ny}",
                                         element_type=et, **p))
    save_csv(str(RESULTS / "ccx_profiles.csv"), profile_rows, PROFILE_FIELDS)

    point_rows, point_rows_eq = [], []
    for case in CASES:
        for et, nx, ny, role in RUN_PLAN:
            if et != "CPS8":
                continue
            point_rows.extend(runs[(case.key, et, nx, ny)]["probes"])
        for nx, ny in SENSITIVITY_MESHES:
            r = sens.get((case.key, "CPS8", nx, ny))
            if r is not None:
                point_rows_eq.extend(r["probes"])
    save_csv(str(RESULTS / "ccx_reference_points.csv"), point_rows, POINT_FIELDS)
    save_csv(str(RESULTS / "ccx_reference_points_equal_nodal.csv"),
             point_rows_eq, POINT_FIELDS)

    ref = build_reference_json(runs, sens, args)
    save_json(str(RESULTS / "ccx_reference.json"), ref)

    compare_rows, compare_status = compare_inhouse(runs, RESULTS)
    compare_fields = ["source", "case", "mesh", "u_end_ccx", "u_end_inhouse",
                      "rel_diff_u_end_pct", "sigma_mid_ccx", "sigma_mid_inhouse",
                      "rel_diff_sigma_mid_pct"]
    if compare_rows:
        save_csv(str(RESULTS / "ccx_vs_inhouse.csv"), compare_rows, compare_fields)
        print("ccx_vs_inhouse", len(compare_rows))
    else:
        print("ccx_vs_inhouse skipped")

    note = render_note(ref, anchor_rows, compare_rows, compare_status, args)
    with open(RESULTS / "ccx_note.md", "w", encoding="utf-8") as f:
        f.write(note)

    print("")
    print("reference")
    for case in CASES:
        c = ref["cases"][case.key]
        print(f"  {case.key}: u_end={c['u_end']:.6e} u_mid={c['u_mid']:.6e} "
              f"sigma_mid={c['sigma_mid']:.3f}")
    print("")
    print(f"time {time.time() - t_start:.1f}s")
    print("files: results/ccx_anchor.csv, ccx_profiles.csv, "
          "ccx_reference_points.csv, ccx_reference_points_equal_nodal.csv, "
          "ccx_reference.json, ccx_note.md")


if __name__ == "__main__":
    main()
