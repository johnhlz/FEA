"""Tests for the 1-D variable-section bar element."""
from __future__ import annotations

import csv
import json
import math
import pathlib
import sys
import tempfile
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fem.analytic import (
    bar_equiv_stiffness,
    bar_gauss_stiffness,
    exact_compliance,
    exact_sigma,
    exact_u,
    exact_u_end,
    exact_u_mid,
)
from fem.bar import (
    BAR_VS_CCX_FIELDS,
    CRITERION_TEXT,
    ETA_A_MAX,
    ETA_B_MAX,
    METHODS,
    applicability,
    bar_connectivity,
    bar_nodes,
    bar_vs_ccx_row,
    build_bar,
    ccx_case_data,
    ccx_posterior,
    classify_eta,
    criterion_row,
    element_stiffness,
    element_stiffness_vector,
    load_ccx_reference,
    node_records,
    observed_order,
    run_record,
)
from fem.config import CASES, CASES_BY_KEY, Geometry, Material


# 1. stiffness limits and closed form
def test_equiv_stiffness_beta_zero_limit():
    mat = Material()
    geom0 = Geometry(L=1.0, h0=0.2, width=0.1, beta=0.0)
    k0 = element_stiffness(geom0, mat, 0.0, 0.1, "equiv")
    k_exact = mat.E * geom0.A0 / 0.1
    assert abs(k0 - k_exact) / k_exact < 1e-12

    # small beta -> E A0 / Le (the log-ratio loses relative accuracy for
    # beta below ~1e-8, so the limit test uses beta=1e-3)
    geom1 = Geometry(L=1.0, h0=0.2, width=0.1, beta=1e-3)
    k1 = element_stiffness(geom1, mat, 0.0, 0.1, "equiv")
    assert abs(k1 - k_exact) / k_exact < 2e-4


def test_equiv_stiffness_closed_form():
    case = CASES_BY_KEY["S1"]
    geom, mat = case.geom, case.material
    for x1, x2 in ((0.0, 0.1), (0.3, 0.4), (0.7, 0.85)):
        expected = (mat.E * geom.A0 * geom.beta
                    / (geom.L * math.log((1.0 + geom.beta * x2 / geom.L)
                                         / (1.0 + geom.beta * x1 / geom.L))))
        got = element_stiffness(geom, mat, x1, x2, "equiv")
        assert abs(got - expected) / expected < 1e-14


def test_compliance_series_additivity():
    for key in ("S1", "S2", "S4"):
        case = CASES_BY_KEY[key]
        geom, mat = case.geom, case.material
        n = 13
        xs = bar_nodes(geom, n)
        total = exact_compliance(geom, mat, 0.0, geom.L)
        parts = sum(exact_compliance(geom, mat, xs[i], xs[i + 1])
                    for i in range(n))
        assert abs(parts - total) / total < 1e-13
        ke = element_stiffness_vector(geom, mat, xs, "equiv")
        k_series = 1.0 / float(np.sum(1.0 / ke))
        k_total = bar_equiv_stiffness(geom, mat, 0.0, geom.L)
        assert abs(k_series - k_total) / k_total < 1e-13


# 2. equiv nodal exactness (mesh independent)
def test_equiv_node_displacements_exact_all_meshes():
    for case in CASES:
        geom, mat, P = case.geom, case.material, case.load
        u_end_exact = exact_u_end(geom, mat, P)
        for n in (1, 2, 3, 5, 7, 10, 13, 40):
            model = build_bar(geom, mat, P, n, "equiv")
            u_ex = exact_u(geom, mat, P, model.nodes)
            err = float(np.max(np.abs(model.U - u_ex))
                        / np.max(np.abs(u_ex)))
            assert err < 1e-10, (case.key, n, err)
            assert abs(model.u_end - u_end_exact) / u_end_exact < 1e-10


# 3. equilibrium / reactions / energy
def test_reaction_equals_minus_P():
    for case in CASES:
        geom, mat, P = case.geom, case.material, case.load
        for n in (1, 2, 5, 10, 40):
            for method in METHODS:
                model = build_bar(geom, mat, P, n, method)
                assert abs(model.reaction_x + P) / P < 1e-8, (
                    case.key, n, method, model.reaction_x)


def test_element_force_is_uniform_and_equals_P():
    for case in CASES:
        geom, mat, P = case.geom, case.material, case.load
        for n in (1, 5, 10, 40):
            for method in METHODS:
                model = build_bar(geom, mat, P, n, method)
                assert model.force_error < 1e-8, (case.key, n, method)
                forces = model.element_axial_forces
                assert abs(float(np.mean(forces)) - P) / P < 1e-8


def test_energy_consistency():
    for case in CASES:
        geom, mat, P = case.geom, case.material, case.load
        for n in (1, 5, 20):
            for method in METHODS:
                model = build_bar(geom, mat, P, n, method)
                assert abs(model.strain_energy - model.external_work) / model.external_work < 1e-10


def test_beta_zero_degenerates_to_PL_over_EA():
    case = CASES_BY_KEY["S0"]
    geom, mat, P = case.geom, case.material, case.load
    u_pl = P * geom.L / (mat.E * geom.A0)
    for method in METHODS:
        for n in (1, 5, 10, 40):
            model = build_bar(geom, mat, P, n, method)
            assert abs(model.u_end - u_pl) / u_pl < 1e-12, (method, n)


# 4. mean vs equiv: O(h^2) discretisation error
def test_mean_error_is_second_order_and_equiv_is_exact():
    for key in ("S1", "S2", "S4"):
        case = CASES_BY_KEY[key]
        ns = [10, 20, 40, 80, 160]
        errs = [abs(run_record(case, n, "mean")["e_mid_pct"]) for n in ns]
        assert errs[-1] < errs[0], (key, errs)
        order = float(observed_order(ns, errs, 10)["order"])
        assert 1.4 <= order <= 2.8, (key, order)
        for n in ns:
            assert abs(run_record(case, n, "equiv")["e_mid_pct"]) < 1e-10


def test_gauss_equals_mean_for_linear_area():
    for case in CASES:
        geom, mat = case.geom, case.material
        for x1, x2 in ((0.0, 0.1), (0.2, 0.37), (0.5, 0.9)):
            kg = element_stiffness(geom, mat, x1, x2, "gauss")
            km = element_stiffness(geom, mat, x1, x2, "mean")
            assert abs(kg - km) / km < 1e-14, (case.key, x1, x2)
            # 2-point Gauss integrates A exactly: k = E * trapz(A) / Le^2
            a_int = 0.5 * (geom.A(x1) + geom.A(x2)) * (x2 - x1)
            expected = mat.E * a_int / (x2 - x1) ** 2
            assert abs(kg - expected) / expected < 1e-14
            # local implementation must agree with the shared analytic helper
            ks = bar_gauss_stiffness(geom, mat, x1, x2)
            assert abs(kg - ks) / ks < 1e-12, (case.key, x1, x2, kg, ks)


# 5. stress recovery
def test_stress_recovery_exact_for_constant_section():
    case = CASES_BY_KEY["S0"]
    geom, mat, P = case.geom, case.material, case.load
    for method in METHODS:
        model = build_bar(geom, mat, P, 10, method)
        s_ex = exact_sigma(geom, mat, P, model.nodes)
        rel = np.abs(model.nodal_stresses - s_ex) / np.abs(s_ex)
        assert float(np.max(rel)) < 1e-12, (method, float(np.max(rel)))


def test_stress_recovery_converges_second_order():
    case = CASES_BY_KEY["S1"]
    ns = [10, 20, 40, 80, 160]
    errs = [abs(run_record(case, n, "equiv")["e_sigma_pct"]) for n in ns]
    assert errs[-1] < errs[0]
    order = float(observed_order(ns, errs, 10)["order"])
    assert 1.4 <= order <= 2.8, order


# 6. applicability criterion
def test_criterion_geometry_and_labels():
    assert 0.0 < ETA_A_MAX < ETA_B_MAX
    expected = {"S0": "A", "S1": "A", "S2": "A", "S3": "B", "S4": "C"}
    for case in CASES:
        geom = case.geom
        assert abs(geom.eta - abs(geom.beta) * geom.h0 / geom.L) < 1e-15
        assert abs(math.tan(math.radians(geom.taper_angle_deg)) - geom.eta) < 1e-12
        app = applicability(geom)
        assert app["label"] == expected[case.key], (case.key, app["label"])
        assert abs(app["area_ratio"] - geom.A1 / geom.A0) < 1e-14
        assert app["criterion"] == CRITERION_TEXT
        assert app["recommendation"]
        assert abs(app["eta_sq_pct"] - 100.0 * geom.eta ** 2) < 1e-12
    assert classify_eta(ETA_A_MAX) == "A"
    assert classify_eta(ETA_A_MAX + 1e-9) == "B"
    assert classify_eta(ETA_B_MAX) == "B"
    assert classify_eta(ETA_B_MAX + 1e-9) == "C"
    assert classify_eta(-0.05) == "A"


# 7. optional CCX posterior columns
def test_ccx_reference_missing_is_nan_not_error():
    assert load_ccx_reference(str(ROOT / "results" / "no_such_file.json")) == {}
    case = CASES_BY_KEY["S1"]
    row = criterion_row(case, {})
    for key in ("u_mid_ccx_um", "e_u_mid_pct_ccx",
                "sigma_mid_ccx", "e_sigma_mid_pct_ccx"):
        assert math.isnan(row[key]), key
    assert abs(row["u_mid_exact_um"]
               - 1e6 * exact_u_mid(case.geom, case.material, case.load)) < 1e-9


def test_ccx_reference_parsing_variants():
    case = CASES_BY_KEY["S1"]
    geom, mat, P = case.geom, case.material, case.load
    u1d = exact_u_mid(geom, mat, P)
    s1d = float(exact_sigma(geom, mat, P, 0.5 * geom.L))
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="bar_ccx_"))

    # variant 1: flat {case: {u_mid, sigma_mid}}
    p1 = tmp / "ref1.json"
    p1.write_text(json.dumps({"S1": {"u_mid": 1.01 * u1d,
                                     "sigma_mid": 0.99 * s1d}}), encoding="utf-8")
    post = ccx_posterior(case, load_ccx_reference(str(p1)))
    assert post["ccx_found"]
    assert abs(post["e_u_mid_pct_ccx"] - 1.0) < 1e-9
    assert abs(post["e_sigma_mid_pct_ccx"] + 1.0) < 1e-9

    # variant 2: wrapper {"cases": ...} + list-of-dict profile fallback
    p2 = tmp / "ref2.json"
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    prof = [{"x": x, "u_section_avg": 1.02 * u1d * (x / 0.5),
             "sigma_section_avg": 0.98 * s1d} for x in xs]
    p2.write_text(json.dumps({"cases": {"S1": {"profile": prof}}}),
                  encoding="utf-8")
    post2 = ccx_posterior(case, load_ccx_reference(str(p2)))
    assert post2["ccx_found"]
    assert abs(post2["u_mid_ccx"] - 1.02 * u1d) / u1d < 1e-9
    assert abs(post2["sigma_mid_ccx"] - 0.98 * s1d) / s1d < 1e-9
    assert post2["e_u_mid_pct_ccx"] > 0.0
    assert post2["e_sigma_mid_pct_ccx"] < 0.0


def test_new_ccx_schema_and_bar_vs_ccx_row():
    case = CASES_BY_KEY["S1"]
    geom, mat, P = case.geom, case.material, case.load
    u1d_mid = exact_u_mid(geom, mat, P)
    u1d_end = exact_u_end(geom, mat, P)
    s1d = float(exact_sigma(geom, mat, P, 0.5 * geom.L))
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="bar_ccx_new_"))
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    prof = [{"x": x,
             "u_section_avg": 1.02 * float(exact_u(geom, mat, P, x)),
             "sigma_section_avg": 0.98 * s1d,
             "sigma_nom": s1d} for x in xs]
    payload = {"generated_by": "unit-test", "reference_mesh": "240x48",
               "reference_element_type": "CPS8",
               "cases": {"S1": {"mesh": "240x48", "element_type": "CPS8",
                                "load_mode": "consistent_uniform_traction",
                                "u_mid": 1.01 * u1d_mid,
                                "sigma_mid": 0.99 * s1d,
                                "profile": prof}}}
    p = tmp / "ref_new.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    ref = load_ccx_reference(str(p))
    assert list(ref.keys()) == ["S1"]  # metadata + "cases" wrapper unwrapped
    data = ccx_case_data(case, ref)
    assert data["found"]
    assert abs(data["u_mid"] - 1.01 * u1d_mid) / u1d_mid < 1e-12
    assert abs(data["sigma_mid"] - 0.99 * s1d) / s1d < 1e-12
    # u_end is absent as a scalar -> interpolated from the profile at x=L
    assert abs(data["u_end"] - 1.02 * u1d_end) / u1d_end < 1e-10
    assert data["mesh"] == "240x48" and data["element_type"] == "CPS8"
    post = ccx_posterior(case, ref)
    assert abs(post["e_u_mid_pct_ccx"] - 1.0) < 1e-9
    assert abs(post["e_sigma_mid_pct_ccx"] + 1.0) < 1e-3

    row = bar_vs_ccx_row(case, ref, n_elem=160, method="equiv")
    assert list(row.keys()) == BAR_VS_CCX_FIELDS
    assert row["case"] == "S1" and row["label"] == "A"
    assert abs(row["rel_end_pct"] - 2.0) < 1e-6
    assert abs(row["rel_mid_pct"] - 1.0) < 1e-6
    assert abs(row["rel_sigma_pct"] + 1.0) < 0.01
    assert abs(row["u_mid_bar"] - u1d_mid) / u1d_mid < 1e-9


def test_ccx_reference_real_file_if_present():
    ref = load_ccx_reference()
    if not ref:
        return
    for case in CASES:
        data = ccx_case_data(case, ref)
        assert data["found"], case.key
        assert np.isfinite(data["u_end"])
        assert np.isfinite(data["u_mid"])
        assert np.isfinite(data["sigma_mid"])


def test_bar_vs_ccx_csv_if_present():
    p = ROOT / "results" / "bar_vs_ccx.csv"
    if not p.is_file():
        return
    with open(p, "r", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(CASES)
    assert list(rows[0].keys()) == BAR_VS_CCX_FIELDS
    for row in rows:
        assert row["case"] in {c.key for c in CASES}
        assert row["label"] in {"A", "B", "C"}


# 8. CSV schema / helper functions
def test_csv_schema_matches_interfaces():
    case = CASES_BY_KEY["S1"]
    row = run_record(case, 10, "equiv")
    assert list(row.keys()) == [
        "case", "n_elem", "method", "u_end", "u_mid", "u_end_exact",
        "u_mid_exact", "e_end_pct", "e_mid_pct", "sigma_mid",
        "sigma_mid_exact", "e_sigma_pct"]
    nrow = node_records(case, 10, "equiv")[0]
    assert list(nrow.keys()) == [
        "case", "n_elem", "method", "x", "u_fem", "u_exact",
        "sigma_fem", "sigma_exact"]
    crow = criterion_row(case, {})
    required = ["case", "eta", "angle_deg", "slenderness", "area_ratio",
                "label", "criterion", "recommendation", "u_mid_exact_um"]
    assert list(crow.keys())[:len(required)] == required
    for extra in ("u_mid_ccx_um", "e_u_mid_pct_ccx", "sigma_mid_ccx",
                  "e_sigma_mid_pct_ccx"):
        assert extra in crow


def test_mesh_helpers():
    geom = CASES_BY_KEY["S1"].geom
    for n in (1, 5, 10, 160):
        nodes = bar_nodes(geom, n)
        assert nodes.size == n + 1
        assert abs(nodes[0]) < 1e-15 and abs(nodes[-1] - geom.L) < 1e-15
        conn = bar_connectivity(n)
        assert conn.shape == (n, 2)
        assert np.array_equal(conn[:, 1], conn[:, 0] + 1)
        model = build_bar(geom, CASES_BY_KEY["S1"].material, 1000.0, n)
        assert model.K.shape == (n + 1, n + 1)
        assert np.allclose(model.K.toarray(), model.K.toarray().T)


def test_observed_order_estimator():
    ns = np.array([10, 20, 40, 80, 160], dtype=float)
    errs = 5.0 / ns ** 2
    res = observed_order(ns, errs, 10)
    assert abs(res["order"] - 2.0) < 1e-9
    assert len(res["pairs"]) == 4
    for pair in res["pairs"]:
        assert abs(pair["order"] - 2.0) < 1e-9
    res0 = observed_order(ns, np.zeros_like(errs), 10)
    assert math.isnan(res0["order"])
    # round-off level errors are treated as numerical zero
    tiny = observed_order(ns, 1e-13 / ns, 10)
    assert math.isnan(tiny["order"])
    assert tiny["err_max"] < 1e-9


# plain-python runner (no pytest required)
def _run_all():
    names = sorted(name for name, obj in globals().items()
                   if name.startswith("test_") and callable(obj))
    failures = []
    for name in names:
        try:
            globals()[name]()
        except Exception as exc:
            failures.append(name)
            print("FAIL", name, ":", repr(exc))
            traceback.print_exc()
        else:
            print("PASS", name)
    print("ok %d/%d" % (len(names) - len(failures), len(names)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
