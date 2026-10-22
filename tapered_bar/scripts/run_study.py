"""Collect and compare method results."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from fem.config import CASES, MESH_LEVELS, STATIONS, INTERIOR_STATIONS
from fem.mesh import make_mesh
from fem.io import save_csv, save_json
from fem import analytic as an
from fem import bar as bar_mod
from fem import q4 as q4_mod
from fem import cst as cst_mod
from fem import post as post_mod

RES = os.path.join(PROJ, "results")
PROBE_XS = (0.1, 0.2, 0.5, 0.8, 0.9)
PROBE_FY = (0.0, 0.25, -0.25)
REF_MESH = (320, 64)


def probe_points(geom):
    pts = []
    for xf in PROBE_XS:
        x = xf * geom.L
        h = float(geom.h(x))
        for fy in PROBE_FY:
            pts.append((x, fy * h))
    return np.asarray(pts, dtype=float)


def norm_xy(x, y):
    return (round(float(x), 6), round(float(y), 6))


def _read_csv(path):
    import csv
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


# reference solution
def _parse_reference_json(raw):
    """Normalise several possible ccx_reference.json layouts."""
    cases = raw.get("cases", raw)
    out = {}
    for key, ent in cases.items():
        if not isinstance(ent, dict):
            continue
        prof = ent.get("profile") or ent.get("profiles") or []
        prof_rows = []
        if isinstance(prof, dict):
            xs = prof.get("x", [])
            ss = prof.get("sigma_section_avg", prof.get("sigma", []))
            us = prof.get("u_section_avg", prof.get("u", []))
            for k in range(len(xs)):
                prof_rows.append(dict(
                    x=float(xs[k]),
                    sigma_section_avg=float(ss[k]) if k < len(ss) else np.nan,
                    u_section_avg=float(us[k]) if k < len(us) else np.nan,
                ))
        else:
            for pr in prof:
                prof_rows.append(dict(
                    x=float(pr["x"]),
                    sigma_section_avg=_as_float(
                        pr.get("sigma_section_avg", pr.get("sigma"))),
                    u_section_avg=_as_float(
                        pr.get("u_section_avg", pr.get("u"))),
                ))
        out[key] = dict(
            u_end=float(ent["u_end"]),
            u_mid=float(ent["u_mid"]),
            profile=prof_rows,
            mesh=ent.get("mesh"),
            element_type=ent.get("element_type", ent.get("element")),
            source=ent.get("source", "CalculiX"),
        )
    return out


def compute_inhouse_reference():
    """Fine Q4 reference (used only if the CCX anchor is unavailable)."""
    print("ref fallback Q4", f"{REF_MESH[0]}x{REF_MESH[1]}")
    out = {}
    for case in CASES:
        g = case.geom
        m = make_mesh(g, *REF_MESH)
        res = q4_mod.analyze_case(m, case)
        rr = res["run_row"]
        prof = [dict(x=row["x"], sigma_section_avg=row["sigma_section_avg"],
                     u_section_avg=row["u_section_avg"])
                for row in res["profile_rows"]]
        out[case.key] = dict(u_end=rr["u_end"], u_mid=rr["u_mid"],
                             profile=prof, mesh=f"{REF_MESH[0]}x{REF_MESH[1]}",
                             element_type="Q4-inhouse",
                             source="in-house Q4 (fallback)")
        # probe table
        pts = probe_points(g)
        rows = []
        for x, y in pts:
            pt = q4_mod.evaluate_point(m, case, res["U"], x, y)
            rows.append(dict(case=case.key, x=x, y=y, ux=pt["ux"],
                             uy=pt["uy"], sxx=pt["sxx"], syy=pt["syy"],
                             sxy=pt["sxy"]))
        out[case.key]["points"] = rows
    return out


def load_reference():
    jpath = os.path.join(RES, "ccx_reference.json")
    cpath = os.path.join(RES, "ccx_reference_points.csv")
    if os.path.exists(jpath):
        with open(jpath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        ref = _parse_reference_json(raw)
        source = ref[next(iter(ref))].get("source", "CalculiX")
    else:
        cache = os.path.join(RES, "inhouse_reference.json")
        if os.path.exists(cache):
            with open(cache, "r", encoding="utf-8") as f:
                ref = json.load(f)
        else:
            ref = compute_inhouse_reference()
            save_json(cache, ref)
        source = "in-house Q4 (fallback)"

    # reference probe values: prefer dedicated CSV, else embedded points
    points = {}
    if os.path.exists(cpath):
        import csv
        raw_rows = list(csv.DictReader(open(cpath, "r", encoding="utf-8-sig")))

        def mesh_nx(tag):
            try:
                return int(str(tag).split("x")[0])
            except (ValueError, AttributeError):
                return -1

        # The CSV may contain the anchor meshes as well; keep the finest mesh
        # per case so that the reference is the converged CPS8 solution.
        best = {}
        for row in raw_rows:
            key = row["case"]
            best[key] = max(best.get(key, -1), mesh_nx(row.get("mesh")))
        for row in raw_rows:
            key = row["case"]
            if mesh_nx(row.get("mesh")) != best[key]:
                continue
            points.setdefault(key, {})[norm_xy(float(row["x"]), float(row["y"]))] = dict(
                ux=_as_float(row.get("ux")),
                uy=_as_float(row.get("uy")),
                sxx=_as_float(row.get("sxx")),
                syy=_as_float(row.get("syy")),
                sxy=_as_float(row.get("sxy")),
                mesh=row.get("mesh"),
                element_type=row.get("element_type"),
            )
    else:
        for key, ent in ref.items():
            for pr in ent.get("points", []):
                points.setdefault(key, {})[norm_xy(pr["x"], pr["y"])] = pr
    return ref, points, source


# metrics
def rel_pct(a, b):
    if b == 0 or not np.isfinite(b):
        return np.nan
    return 100.0 * (float(a) - float(b)) / float(b)


def probe_rms(rows, x_max=1.0):
    values = [p["sxx"] for p in rows if p["x"] <= x_max * 1.0000001]
    refs = [p["sxx_ref"] for p in rows if p["x"] <= x_max * 1.0000001]
    return rms_rel_pct(values, refs)


def rms_rel_pct(values, refs):
    values = np.asarray(values, dtype=float)
    refs = np.asarray(refs, dtype=float)
    good = np.isfinite(values) & np.isfinite(refs)
    if good.sum() == 0:
        return np.nan
    return float(100.0 * np.sqrt(np.mean((values[good] - refs[good]) ** 2))
                 / np.sqrt(np.mean(refs[good] ** 2)))


def ref_profile_at(ref_ent, xs):
    prof = ref_ent.get("profile", [])
    if not prof:
        return np.full(len(xs), np.nan)
    px = np.array([p["x"] for p in prof], dtype=float)
    ps = np.array([p["sigma_section_avg"] for p in prof], dtype=float)
    order = np.argsort(px)
    return np.interp(np.asarray(xs, dtype=float), px[order], ps[order])


def profile_gradient(x, sigma):
    x = np.asarray(x, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    g = np.full_like(sigma, np.nan)
    if sigma.size >= 3:
        g[1:-1] = (sigma[2:] - sigma[:-2]) / (x[2:] - x[:-2])
    return g


def fit_order(hs, errs):
    hs = np.asarray(hs, dtype=float)
    errs = np.abs(np.asarray(errs, dtype=float))
    good = np.isfinite(errs) & (errs > 0)
    if good.sum() < 2:
        return dict(slope=np.nan, intercept=np.nan, r2=np.nan, n=int(good.sum()))
    # use the three finest meshes for the asymptotic slope
    idx = np.flatnonzero(good)[-3:]
    if idx.size < 2:
        return dict(slope=np.nan, intercept=np.nan, r2=np.nan, n=int(idx.size))
    lx = np.log(hs[idx])
    ly = np.log(errs[idx])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return dict(slope=float(slope), intercept=float(intercept),
                r2=float(r2), n=int(idx.size))


# collection
def collect():
    ref, ref_points, ref_source = load_reference()
    runs, profiles, probes, conv = [], [], [], []
    t0 = time.time()

    for case in CASES:
        g, mat, P = case.geom, case.material, case.load
        ue_1d = an.exact_u_end(g, mat, P)
        um_1d = an.exact_u_mid(g, mat, P)
        sig_nom_st = an.exact_sigma(g, mat, P, g.L * STATIONS)
        ref_ent = ref.get(case.key, {})
        ue_ref = ref_ent.get("u_end", np.nan)
        um_ref = ref_ent.get("u_mid", np.nan)
        sig_ref_st = ref_profile_at(ref_ent, g.L * STATIONS)
        grad_ref = profile_gradient(g.L * STATIONS, sig_ref_st)
        grad_nom = an.exact_sigma_gradient(g, mat, P, g.L * STATIONS)
        case_probes = ref_points.get(case.key, {})
        print(case.key, ref_source, f"u_end={ue_ref:.6e}")

        for n_elem in (10, 160):
            for method in ("equiv", "mean"):
                bm = bar_mod.solve_case(case, n_elem, method)
                sig_probe = an.exact_sigma(g, mat, P, [p[0] for p in probe_points(g)])
                pr = []
                for (px, py), sx in zip(probe_points(g), sig_probe):
                    refp = case_probes.get(norm_xy(px, py), {})
                    pr.append(dict(case=case.key, method="BAR", diag="-",
                                   nx=n_elem, ny=0, x=px, y=py, sx=sx,
                                   sx_ref=refp.get("sxx", np.nan)))
                e_sig_probe = rms_rel_pct([p["sx"] for p in pr],
                                          [p["sx_ref"] for p in pr])
                runs.append(dict(
                    case=case.key, method=f"BAR-{method}", diag="-",
                    nx=n_elem, ny=0, nnodes=n_elem + 1, ndof=n_elem + 1,
                    u_end=bm.u_end, u_mid=bm.u_mid,
                    e_end_1d_pct=rel_pct(bm.u_end, ue_1d),
                    e_mid_1d_pct=rel_pct(bm.u_mid, um_1d),
                    e_end_ref_pct=rel_pct(bm.u_end, ue_ref),
                    e_mid_ref_pct=rel_pct(bm.u_mid, um_ref),
                    e_sigma_probe_ref_pct=e_sig_probe,
                    e_sigma_probe_int_ref_pct=e_sig_probe,
                    e_sigma_profile_ref_pct=0.0,
                    grad_fidelity_ref_pct=np.nan,
                    shear_energy_ratio=np.nan,
                    reaction_x=bm.reaction_x,
                    reaction_rel_err=abs(bm.reaction_x + P) / P,
                ))
                profiles.extend(dict(
                    case=case.key, method=f"BAR-{method}", diag="-",
                    nx=n_elem, ny=0, x=float(x), u=float(u),
                    sigma=float(sx), sigma_nom=float(sn),
                    dsigma_dx_fem=np.nan, dsigma_dx_nom=float(gn),
                ) for x, u, sx, sn, gn in zip(
                    bm.nodes, bm.U,
                    an.exact_sigma(g, mat, P, bm.nodes),
                    an.exact_sigma(g, mat, P, bm.nodes),
                    an.exact_sigma_gradient(g, mat, P, bm.nodes)))
        # convergence entries for bar equiv
        for n_elem in (5, 10, 20, 40, 80, 160):
            bm = bar_mod.solve_case(case, n_elem, "equiv")
            conv.append(dict(case=case.key, method="BAR-equiv",
                             n=n_elem, h=1.0 / n_elem,
                             e_u_mid_ref_pct=rel_pct(bm.u_mid, um_ref),
                             e_u_end_ref_pct=rel_pct(bm.u_end, ue_ref),
                             e_sigma_probe_ref_pct=np.nan,
                             e_sigma_profile_ref_pct=0.0))

        for nx, ny in MESH_LEVELS:
            h = 1.0 / nx
            m = make_mesh(g, nx, ny)
            pts = probe_points(g)
            pres = q4_mod.analyze_case(m, case, probe_points=pts)
            prun = pres["run_row"]
            q4_u_end = post_mod.end_average_u(m, pres["U"])
            q4_u_mid = post_mod.mid_average_u(m, pres["U"])
            prof = pres["profile_rows"]
            sig_st = np.array([row["sigma_section_avg"] for row in prof])
            e_sig_prof = rms_rel_pct(sig_st[1:-1], sig_ref_st[1:-1])
            gfem = np.array([row["dsigma_dx_fem"] for row in prof])
            grad_fid = rms_rel_pct(gfem[1:-1], grad_ref[1:-1])
            # probes
            prow = []
            for row in pres["probe_rows"]:
                refp = case_probes.get(norm_xy(row["x"], row["y"]), {})
                prow.append(dict(case=case.key, method="Q4", diag="-",
                                 nx=nx, ny=ny, x=row["x"], y=row["y"],
                                 sxx=row["sxx"], sxx_ref=refp.get("sxx", np.nan),
                                 ux=row["ux"], uy=row["uy"]))
            e_sig_probe = rms_rel_pct([p["sxx"] for p in prow],
                                      [p["sxx_ref"] for p in prow])
            profiles.extend(dict(
                case=case.key, method="Q4", diag="-", nx=nx, ny=ny,
                x=row["x"], u=row["u_section_avg"],
                sigma=row["sigma_section_avg"], sigma_nom=row["sigma_nom"],
                dsigma_dx_fem=row["dsigma_dx_fem"],
                dsigma_dx_nom=row["dsigma_dx_nom"]) for row in prof)
            runs.append(dict(
                case=case.key, method="Q4", diag="-", nx=nx, ny=ny,
                nnodes=m.nnodes, ndof=m.ndof,
                u_end=q4_u_end, u_mid=q4_u_mid,
                e_end_1d_pct=rel_pct(q4_u_end, ue_1d),
                e_mid_1d_pct=rel_pct(q4_u_mid, um_1d),
                e_end_ref_pct=rel_pct(q4_u_end, ue_ref),
                e_mid_ref_pct=rel_pct(q4_u_mid, um_ref),
                e_sigma_probe_ref_pct=e_sig_probe,
                e_sigma_probe_int_ref_pct=probe_rms(prow, x_max=0.8),
                e_sigma_profile_ref_pct=e_sig_prof,
                grad_fidelity_ref_pct=grad_fid,
                shear_energy_ratio=prun["shear_energy_ratio"],
                reaction_x=prun["reaction_x"],
                reaction_rel_err=abs(prun["reaction_x"] + P) / P,
            ))
            conv.append(dict(case=case.key, method="Q4", n=nx, h=h,
                             e_u_mid_ref_pct=rel_pct(q4_u_mid, um_ref),
                             e_u_end_ref_pct=rel_pct(q4_u_end, ue_ref),
                             e_sigma_probe_ref_pct=e_sig_probe,
                             e_sigma_probe_int_ref_pct=probe_rms(prow, x_max=0.8),
                             e_sigma_profile_ref_pct=e_sig_prof,
                             grad_fidelity_ref_pct=grad_fid))
            probes.extend(prow)

            # CST (both diagonals for local metrics, pos for convergence)
            cres = {}
            for diag in ("pos", "neg"):
                r = cst_mod.solve_case(case, nx, ny, diag, mesh=m)
                cst_u_end = post_mod.end_average_u(m, r["U"])
                cst_u_mid = post_mod.mid_average_u(m, r["U"])
                prof_c = cst_mod.section_profile(r)
                s_st = np.asarray(prof_c["sigma_section_avg_raw"])
                e_prof = rms_rel_pct(s_st[1:-1], sig_ref_st[1:-1])
                cprobe = cst_mod.probe_rows(r, case)
                cp = []
                for row in cprobe:
                    refp = case_probes.get(norm_xy(row["x"], row["y"]), {})
                    cp.append(dict(case=case.key, method="CST", diag=diag,
                                   nx=nx, ny=ny, x=row["x"], y=row["y"],
                                   sxx=row["sxx"],
                                   sxx_ref=refp.get("sxx", np.nan),
                                   ux=row["ux"], uy=row["uy"]))
                e_probe = rms_rel_pct([p["sxx"] for p in cp],
                                      [p["sxx_ref"] for p in cp])
                cres[diag] = r
                runs.append(dict(
                    case=case.key, method="CST", diag=diag, nx=nx, ny=ny,
                    nnodes=m.nnodes, ndof=m.ndof,
                    u_end=cst_u_end, u_mid=cst_u_mid,
                    e_end_1d_pct=rel_pct(cst_u_end, ue_1d),
                    e_mid_1d_pct=rel_pct(cst_u_mid, um_1d),
                    e_end_ref_pct=rel_pct(cst_u_end, ue_ref),
                    e_mid_ref_pct=rel_pct(cst_u_mid, um_ref),
                    e_sigma_probe_ref_pct=e_probe,
                    e_sigma_probe_int_ref_pct=probe_rms(cp, x_max=0.8),
                    e_sigma_profile_ref_pct=e_prof,
                    grad_fidelity_ref_pct=np.nan,
                    shear_energy_ratio=np.nan,
                    reaction_x=r["reaction"],
                    reaction_rel_err=abs(r["reaction"] + P) / P,
                ))
                profiles.extend(dict(
                    case=case.key, method="CST", diag=diag, nx=nx, ny=ny,
                    x=float(prof_c["x"][k]),
                    u=float(prof_c["u_section_avg"][k]),
                    sigma=float(s_st[k]),
                    sigma_nom=float(sig_nom_st[k]),
                    dsigma_dx_fem=np.nan,
                    dsigma_dx_nom=float(grad_nom[k]))
                    for k in range(len(prof_c["x"])))
                probes.extend(cp)
                if diag == "pos":
                    conv.append(dict(
                        case=case.key, method="CST-pos", n=nx, h=h,
                        e_u_mid_ref_pct=rel_pct(cst_u_mid, um_ref),
                        e_u_end_ref_pct=rel_pct(cst_u_end, ue_ref),
                        e_sigma_probe_ref_pct=e_probe,
                        e_sigma_probe_int_ref_pct=probe_rms(cp, x_max=0.8),
                        e_sigma_profile_ref_pct=e_prof,
                        grad_fidelity_ref_pct=np.nan))
            # diagonal metrics (needs both)
            dm = cst_mod.diagonal_metrics(cres["pos"], cres["neg"],
                                          sig_nom_st)
            dm.update(dict(case=case.key, nx=nx, ny=ny, h=h))
            runs.append(dict(case=case.key, method="CST-DIAG", diag="pos-neg",
                             nx=nx, ny=ny, nnodes=m.nnodes, ndof=m.ndof,
                             u_end=np.nan, u_mid=np.nan,
                             e_end_1d_pct=np.nan, e_mid_1d_pct=np.nan,
                             e_end_ref_pct=np.nan, e_mid_ref_pct=np.nan,
                             e_sigma_probe_ref_pct=np.nan,
                             e_sigma_profile_ref_pct=np.nan,
                             grad_fidelity_ref_pct=np.nan,
                             shear_energy_ratio=np.nan,
                             reaction_x=np.nan, reaction_rel_err=np.nan,
                             **{k: v for k, v in dm.items()
                                if k not in ("case", "nx", "ny", "h")}))
        print(f"  {time.time() - t0:.1f}s")

    fits = []
    for case in CASES:
        for method in ("Q4", "CST-pos", "BAR-equiv"):
            for metric in ("e_u_mid_ref_pct", "e_u_end_ref_pct",
                           "e_sigma_probe_ref_pct", "e_sigma_profile_ref_pct",
                           "grad_fidelity_ref_pct"):
                rows = [r for r in conv if r["case"] == case.key
                        and r["method"] == method and metric in r]
                if not rows:
                    continue
                rows.sort(key=lambda r: r["h"], reverse=True)
                fit = fit_order([r["h"] for r in rows],
                                [r.get(metric, np.nan) for r in rows])
                fits.append(dict(case=case.key, method=method, metric=metric,
                                 **fit))
    def _union_fields(rows):
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        return fields

    save_csv(os.path.join(RES, "study_convergence.csv"), fits)
    save_csv(os.path.join(RES, "study_runs.csv"), runs,
             fieldnames=_union_fields(runs))
    save_csv(os.path.join(RES, "study_profiles.csv"), profiles,
             fieldnames=_union_fields(profiles))
    save_csv(os.path.join(RES, "study_probes.csv"), probes,
             fieldnames=_union_fields(probes))

    tables = dict(
        reference_source=ref_source,
        reference_meshes={k: dict(mesh=v.get("mesh"),
                                  element_type=v.get("element_type"))
                          for k, v in ref.items()},
        cases={c.key: dict(label=c.label, beta=c.geom.beta,
                           h0=c.geom.h0, h1=c.geom.h1,
                           eta=c.geom.eta,
                           taper_angle_deg=c.geom.taper_angle_deg,
                           u_end_1d=an.exact_u_end(c.geom, c.material, c.load),
                           u_mid_1d=an.exact_u_mid(c.geom, c.material, c.load),
                           u_end_ref=ref.get(c.key, {}).get("u_end"),
                           u_mid_ref=ref.get(c.key, {}).get("u_mid"))
                for c in CASES},
        runs=runs, convergence_fits=fits,
        diagonal=[r for r in runs if r["method"] == "CST-DIAG"],
        bar_criterion=_read_csv(os.path.join(RES, "bar_criterion.csv")),
    )
    save_json(os.path.join(RES, "study_tables.json"), tables)
    print("study", f"{time.time() - t0:.1f}s")
    if ref_source.startswith("in-house"):
        print("warning: no ccx reference; using in-house Q4 fallback")
    return tables


if __name__ == "__main__":
    collect()
