#!/usr/bin/env python3
"""1-D variable-section bar study."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fem.analytic import exact_energy, exact_sigma, exact_u, exact_u_end, exact_u_mid
from fem.bar import (
    BAR_VS_CCX_FIELDS,
    CRITERION_TEXT,
    ETA_A_MAX,
    ETA_B_MAX,
    METHODS,
    applicability,
    bar_vs_ccx_row,
    build_bar,
    ccx_case_data,
    ccx_posterior,
    criterion_row,
    criterion_text,
    element_stiffness_vector,
    load_ccx_reference,
    node_records,
    observed_order,
    run_record,
)
from fem.config import CASES
from fem.io import fnum, save_csv, save_json

RESULTS = ROOT / "results"
N_ELEM = (5, 10, 20, 40, 80, 160)

RUN_FIELDS = ["case", "n_elem", "method", "u_end", "u_mid", "u_end_exact",
              "u_mid_exact", "e_end_pct", "e_mid_pct", "sigma_mid",
              "sigma_mid_exact", "e_sigma_pct"]
NODE_FIELDS = ["case", "n_elem", "method", "x", "u_fem", "u_exact",
               "sigma_fem", "sigma_exact"]
CRIT_FIELDS = ["case", "eta", "angle_deg", "slenderness", "area_ratio",
               "label", "criterion", "recommendation", "u_mid_exact_um",
               "u_mid_ccx_um", "e_u_mid_pct_ccx", "sigma_mid_ccx",
               "e_sigma_mid_pct_ccx"]

CHECK_TOL = dict(reaction=1e-8, force=1e-8, equiv_node=1e-10,
                 energy=1e-10, gauss_mean=1e-12, beta0=1e-12)

# 2-D Q4 160x32 reference errors relative to the 1-D nominal solution
# (provided by the lead, 2026-10-22).  Used only to document the calibration
# of the eta thresholds; the CSV posterior columns come from
# results/ccx_reference.json when that file exists.
CALIBRATION_Q4 = {
    "S0": dict(eta=0.00, e_mid_pct=-0.50, e_end_pct=-0.18),
    "S1": dict(eta=0.20, e_mid_pct=0.11, e_end_pct=0.84),
    "S2": dict(eta=0.10, e_mid_pct=-0.31, e_end_pct=-0.05),
    "S3": dict(eta=0.40, e_mid_pct=1.37, e_end_pct=4.38),
    "S4": dict(eta=0.60, e_mid_pct=4.53, e_end_pct=9.54),
}


def fmt_row(row):
    """Format a record row for CSV ('' for NaN, integers as integers)."""
    out = {}
    for key, val in row.items():
        if isinstance(val, (bool, np.bool_)):
            out[key] = str(bool(val))
        elif isinstance(val, (int, np.integer)):
            out[key] = int(val)
        elif isinstance(val, (float, np.floating)):
            out[key] = fnum(float(val), "{:.10e}")
        else:
            out[key] = val
    return out


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    return obj


def md_table(header, rows):
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    ref = load_ccx_reference()

    runs = []
    nodes_rows = []
    crit_rows = []
    maxima = dict(reaction=0.0, force=0.0, equiv_node=0.0, energy=0.0,
                  gauss_mean=0.0, beta0=0.0)
    case_json = {}
    conv_all = {}
    runs_all = {}

    for case in CASES:
        geom, mat, P = case.geom, case.material, case.load
        app = applicability(geom)
        ue = exact_u_end(geom, mat, P)
        um = exact_u_mid(geom, mat, P)
        sm = float(exact_sigma(geom, mat, P, 0.5 * geom.L))
        energy = float(exact_energy(geom, mat, P))
        runs_case = {m: [] for m in METHODS}

        for n_elem in N_ELEM:
            for method in METHODS:
                model = build_bar(geom, mat, P, n_elem, method)
                row = run_record(case, n_elem, method)
                runs.append(row)
                runs_case[method].append(row)
                nodes_rows.extend(node_records(case, n_elem, method))

                maxima["reaction"] = max(
                    maxima["reaction"], abs(model.reaction_x + P) / P)
                maxima["force"] = max(maxima["force"], model.force_error)
                maxima["energy"] = max(
                    maxima["energy"],
                    abs(model.strain_energy - model.external_work) / model.external_work)
                if method == "equiv":
                    u_ex = exact_u(geom, mat, P, model.nodes)
                    maxima["equiv_node"] = max(
                        maxima["equiv_node"],
                        float(np.max(np.abs(model.U - u_ex)) / np.max(np.abs(u_ex))))
                if method == "gauss":
                    km = element_stiffness_vector(geom, mat, model.nodes, "mean")
                    maxima["gauss_mean"] = max(
                        maxima["gauss_mean"],
                        float(np.max(np.abs(model.ke - km) / km)))
                if abs(geom.beta) < 1e-14:
                    u_pl = P * geom.L / (mat.E * geom.A0)
                    maxima["beta0"] = max(
                        maxima["beta0"], abs(model.u_end - u_pl) / u_pl)

        conv = {}
        for method in METHODS:
            rows_m = runs_case[method]
            ns = [r["n_elem"] for r in rows_m]
            e_mid = [abs(r["e_mid_pct"]) for r in rows_m]
            e_end = [abs(r["e_end_pct"]) for r in rows_m]
            e_sig = [abs(r["e_sigma_pct"]) for r in rows_m]
            conv[method] = dict(
                n_elem=ns,
                e_mid_pct=[r["e_mid_pct"] for r in rows_m],
                e_end_pct=[r["e_end_pct"] for r in rows_m],
                e_sigma_pct=[r["e_sigma_pct"] for r in rows_m],
                order_mid=observed_order(ns, e_mid),
                order_end=observed_order(ns, e_end),
                order_sigma=observed_order(ns, e_sig),
            )
        conv_all[case.key] = conv
        runs_all[case.key] = runs_case

        app["case"] = case.key
        crit_rows.append(criterion_row(case, ref))
        case_json[case.key] = dict(
            label=case.label, note=case.note,
            geometry=dict(L=float(geom.L), h0=float(geom.h0), h1=float(geom.h1),
                          beta=float(geom.beta), width=float(geom.width),
                          A0=float(geom.A0), A1=float(geom.A1),
                          eta=float(geom.eta),
                          angle_deg=float(geom.taper_angle_deg),
                          slenderness=float(geom.slenderness),
                          area_ratio=float(geom.A1 / geom.A0),
                          h_min=float(geom.h_min()), h_max=float(geom.h_max())),
            material=dict(E=float(mat.E), nu=float(mat.nu)), load=float(P),
            exact=dict(u_end=float(ue), u_mid=float(um),
                       u_end_um=1.0e6 * float(ue),
                       u_mid_um=1.0e6 * float(um),
                       sigma_mid=sm, energy=energy),
            applicability=app,
            runs={m: {str(r["n_elem"]): dict(
                u_end=r["u_end"], u_mid=r["u_mid"],
                e_end_pct=r["e_end_pct"], e_mid_pct=r["e_mid_pct"],
                sigma_mid=r["sigma_mid"], e_sigma_pct=r["e_sigma_pct"])
                for r in runs_case[m]} for m in METHODS},
            convergence=conv,
            posterior=ccx_posterior(case, ref),
        )

    conv_checks = {}
    for case in CASES:
        conv = conv_all[case.key]
        if abs(case.geom.beta) < 1e-14:
            conv_checks[case.key] = dict(
                passed=True, note="beta=0：所有方法刚度恒为 E A/L，离散误差为 0，不做阶估计")
            continue
        mean_conv = conv["mean"]
        ns = np.asarray(mean_conv["n_elem"], dtype=int)
        e_mid = np.abs(np.asarray(mean_conv["e_mid_pct"], dtype=float))
        e_10 = float(e_mid[ns == 10][0])
        e_160 = float(e_mid[ns == 160][0])
        order = float(mean_conv["order_mid"]["order"])
        ok = bool(e_160 < e_10 and 1.4 <= order <= 2.9)
        conv_checks[case.key] = dict(
            passed=ok, e_mid_pct_n10=e_10, e_mid_pct_n160=e_160,
            order_mid=order,
            note="mean 方法 u_mid 误差应随 n_elem 下降且观测阶约 2（O(h^2)）")
        if not ok:
            raise AssertionError("mean convergence check failed for " + case.key)

    checks = {}
    for key, tol in CHECK_TOL.items():
        checks[key] = dict(max_rel_error=float(maxima[key]),
                           tolerance=float(tol),
                           passed=bool(maxima[key] <= tol))
    checks["convergence"] = conv_checks
    checks["all_passed"] = bool(
        all(checks[k]["passed"] for k in CHECK_TOL)
        and all(v["passed"] for v in conv_checks.values()))
    if not checks["all_passed"]:
        bad = {k: checks[k] for k in CHECK_TOL if not checks[k]["passed"]}
        raise AssertionError("self-check failed: " + repr(bad))

    vs_rows = [bar_vs_ccx_row(case, ref, n_elem=160, method="equiv")
               for case in CASES]
    vs_data = {case.key: ccx_case_data(case, ref) for case in CASES}
    vs_meta = next((d for d in vs_data.values() if d["found"]), None)

    save_csv(str(RESULTS / "bar_runs.csv"),
             [fmt_row(r) for r in runs], fieldnames=RUN_FIELDS)
    save_csv(str(RESULTS / "bar_nodes.csv"),
             [fmt_row(r) for r in nodes_rows], fieldnames=NODE_FIELDS)
    save_csv(str(RESULTS / "bar_criterion.csv"),
             [fmt_row(r) for r in crit_rows], fieldnames=CRIT_FIELDS)
    save_csv(str(RESULTS / "bar_vs_ccx.csv"),
             [fmt_row(r) for r in vs_rows], fieldnames=BAR_VS_CCX_FIELDS)

    summary = dict(
        title="一维杆结果",
        generated_by="scripts/run_bar.py",
        model=dict(L=1.0, E=2.0e11, nu=0.3, P=1000.0, width=0.1,
                   description="h(x)=h0(1+beta x/L), A=b h(x); 左端 u(0)=0, 右端合力 F=P"),
        n_elem=list(N_ELEM),
        methods=dict(
            equiv="k_e = 1 / int dx/(E A(x))（柔度积分，主方法；节点位移精确）",
            mean="k_e = E A(x_mid)/Le（算术平均面积）",
            gauss="k_e = E/Le^2 * int A dx（2 点 Gauss；对线性 A 与 mean 恒等）",
        ),
        postprocessing=dict(
            u_end="U[-1]",
            u_mid="节点位移在 x=L/2 处线性插值；n_elem 偶数时恰为节点值",
            element_force="N_e = k_e (u2-u1)；装配平衡使 N_e = P（所有方法）",
            stress="本构恢复 sigma_e = E (u2-u1)/Le；节点应力取相邻单元平均",
            error_pct="100*(fem-exact)/exact（有符号，%）",
            reaction="固定端 R=K U-F，要求 sum(R_x) = -P",
            energy="0.5 U^T K U = 0.5 P u_end",
        ),
        ccx_comparison=dict(
            source="results/ccx_reference.json",
            reference_mesh=None if vs_meta is None else vs_meta["mesh"],
            reference_element_type=None if vs_meta is None else vs_meta["element_type"],
            load_mode=None if vs_meta is None else vs_meta["load_mode"],
            n_elem=160, method="equiv",
            rel_definition="rel_*_pct = 100*(CCX - bar)/bar，正号=CCX 大于 1D",
            rows=[dict(r) for r in vs_rows],
        ),
        criterion=dict(
            definition="η = |beta| h0 / L = |dh/dx| = tan(锥角)",
            cone_angle="alpha = arctan(|beta| h0 / L)",
            thresholds=dict(A_max=float(ETA_A_MAX), B_max=float(ETA_B_MAX)),
            criterion=criterion_text(),
            labels=dict(A="可信", B="工程可用但需 2D 复核", C="不可信"),
            calibration_2d_q4_160x32=CALIBRATION_Q4,
            calibration_source="Q4 160x32 相对一维名义解",
            posterior_source="results/ccx_reference.json",
            posterior_found=bool(ref),
            posterior_sign="(CCX-1D)/1D，正号表示二维大于一维",
        ),
        checks=checks,
        cases=case_json,
        notes=[
            "equiv 在任意 n_elem 下节点位移等于一维解析解（串联柔度可加）；不把它当作离散误差来源。",
            "mean/gauss 采用刚度积分（算术平均面积），比精确柔度单元偏硬，位移偏小；离散误差 O(h^2)。",
            "2 点 Gauss 对线性 A(x) 严格精确，故 gauss 与 mean 数值恒等，不是独立方法。",
            "n_elem=5 为奇数，x=L/2 不是节点，u_mid/sigma_mid 含 O(h^2) 插值误差；收敛阶拟合使用 n>=10。",
            "应力采用本构恢复而非 N_e/A(x)：平衡已保证 N_e=P，N_e/A(x) 与解析名义应力恒等，无法反映离散误差。",
        ],
    )
    save_json(str(RESULTS / "bar_summary.json"), jsonable(summary))

    note = build_note(case_json, conv_all, runs_all, maxima, checks, ref,
                      crit_rows, vs_rows, vs_meta)
    (RESULTS / "bar_method_note.md").write_text(note, encoding="utf-8")

    print("results", RESULTS)
    print("checks", "ok" if checks["all_passed"] else "fail")
    for key in CHECK_TOL:
        print("  %-11s max=%.3e  tol=%.1e  %s" % (
            key, checks[key]["max_rel_error"], checks[key]["tolerance"],
            "ok" if checks[key]["passed"] else "FAIL"))
    header = ("case", "eta", "label", "u_end_um", "equiv_e_end%",
              "mean_e_end%", "mean_order")
    print("  ".join("%13s" % h for h in header))
    for case in CASES:
        cj = case_json[case.key]
        app = cj["applicability"]
        eq = runs_all[case.key]["equiv"][-1]
        mn = runs_all[case.key]["mean"][-1]
        order = conv_all[case.key]["mean"]["order_mid"]["order"]
        print("  ".join("%13s" % v for v in (
            case.key, "%.2f" % app["eta"], app["label"],
            "%.4f" % cj["exact"]["u_end_um"], "%.3e" % eq["e_end_pct"],
            "%.3e" % mn["e_end_pct"], "%.3f" % order)))
    print("ccx_ref", bool(ref))
    return 0


# method note
def build_note(cases, conv_all, runs_all, maxima, checks, ref, crit_rows,
               vs_rows, vs_meta):
    L = ["# 一维杆结果", ""]
    L.append("k_e = 1 / integral dx/(E A(x))；mean 用 E A(x_mid)/Le。")
    L.append("")
    L.append("| case | eta | label | u_mid/um | CCX u_mid/um | rel mid/% |")
    L.append("|---|---|---|---|---|---|")
    for r in crit_rows:
        u = r.get("u_mid_ccx_um")
        u = f"{float(u):.5f}" if u not in (None, "") else "-"
        e = r.get("e_u_mid_pct_ccx")
        e = f"{float(e):+.3f}" if e not in (None, "") else "-"
        L.append(f"| {r['case']} | {float(r['eta']):.2f} | {r['label']} | "
                 f"{float(r['u_mid_exact_um']):.5f} | {u} | {e} |")
    L.append("")
    L.append("## 自检")
    for k, tol in CHECK_TOL.items():
        v = maxima.get(k, float("nan"))
        L.append(f"- {k}: {v:.3e} / {tol:.0e}")
    L.append("")
    L.append("## 文件")
    L.append("- results/bar_runs.csv, bar_nodes.csv, bar_criterion.csv, bar_vs_ccx.csv")
    L.append("- 图/报告由 scripts/make_figures.py、scripts/make_report.py 生成")
    return "\n".join(L) + "\n"


def ratio_example():
    import math
    r = 1.1
    return (1.0 + r) * math.log(r) / (2.0 * (r - 1.0)) - 1.0


def series_example():
    r = 1.1
    d = r - 1.0
    return d * d / 12.0 - d * d * d / 12.0


if __name__ == "__main__":
    sys.exit(main())
