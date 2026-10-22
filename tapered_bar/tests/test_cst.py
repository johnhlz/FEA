"""CST element and post-processing self-tests."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem.config import CASES, CASES_BY_KEY, MAT, LOAD, MESH_LEVELS, STATIONS
from fem.mesh import make_mesh
from fem import cst
from fem import analytic as an

DIAGS = ("pos", "neg")


def linear_field(mesh, a):
    """Nodal displacement of ux = a0 + a1 x + a2 y, uy = a3 + a4 x + a5 y."""
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    U = np.zeros(mesh.ndof)
    U[0::2] = a[0] + a[1] * x + a[2] * y
    U[1::2] = a[3] + a[4] * x + a[5] * y
    return U


def ref_B(xy):
    """Independent textbook B matrix for one triangle (3x6, positive area)."""
    (x1, y1), (x2, y2), (x3, y3) = xy
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
    return np.array([[b1, 0, b2, 0, b3, 0],
                     [0, c1, 0, c2, 0, c3],
                     [c1, b1, c2, b2, c3, b3]]) / det


def _max_rel(a, ref):
    scale = np.max(np.abs(ref))
    return float(np.max(np.abs(np.asarray(a) - ref)) / scale) if scale > 0 else 0.0


# element level
def test_B_matrix_matches_reference():
    """Batched B / ke against an independent textbook implementation."""
    case = CASES_BY_KEY["S4"]
    mesh = make_mesh(case.geom, 10, 2)
    D = case.material.D_plane_stress()
    t = case.geom.width
    for diag in DIAGS:
        tris = cst.tri_conn(mesh, diag)
        xy, det, area = cst.element_geometry(mesh, tris)
        B = cst.cst_B_batch(xy, det)
        ke = cst.cst_ke_batch(B, area, D, t)
        for e in range(tris.shape[0]):
            Br = ref_B(xy[e])
            assert _max_rel(B[e], Br) < 1e-14, (diag, e)
            ker = t * abs(area[e]) * Br.T @ D @ Br
            assert _max_rel(ke[e], ker) < 1e-13, (diag, e)
            # symmetric + rigid-body modes in the null space
            assert np.allclose(ke[e], ke[e].T, rtol=1e-13, atol=1e-6)
            scale = np.max(np.abs(ke[e]))
            assert np.max(np.abs(ke[e] @ np.array([1.0, 0, 1, 0, 1, 0]))) < 1e-10 * scale
            assert np.max(np.abs(ke[e] @ np.array([0, 1.0, 0, 1, 0, 1]))) < 1e-10 * scale
        ev = np.linalg.eigvalsh(ke)
        assert ev.min() > -1e-10 * ev.max(), (diag, ev.min(), ev.max())


def test_triangle_areas_positive():
    for case in CASES:
        for nx, ny in MESH_LEVELS:
            mesh = make_mesh(case.geom, nx, ny)
            qa = mesh.quad_areas()
            for diag in DIAGS:
                tris = cst.tri_conn(mesh, diag)
                xy, det, area = cst.element_geometry(mesh, tris)
                assert np.all(det > 0.0), (case.key, nx, ny, diag)
                assert np.all(area > 0.0), (case.key, nx, ny, diag)
                pair_sum = area.reshape(-1, 2).sum(axis=1)
                assert np.allclose(pair_sum, qa, rtol=1e-12, atol=0.0), \
                    (case.key, nx, ny, diag)


def test_constant_strain_patch():
    """Linear displacement field -> constant stress D*eps in every triangle."""
    a = (1.0e-4, 1.0e-3, 2.0e-4, -2.0e-4, -3.0e-4, 5.0e-4)
    eps_ref = np.array([a[1], a[5], a[2] + a[4]])
    for case in CASES:
        D = case.material.D_plane_stress()
        sig_ref = D @ eps_ref
        for nx, ny in ((10, 2), (40, 8)):
            mesh = make_mesh(case.geom, nx, ny)
            U = linear_field(mesh, a)
            for diag in DIAGS:
                tris = cst.tri_conn(mesh, diag)
                xy, det, area = cst.element_geometry(mesh, tris)
                B = cst.cst_B_batch(xy, det)
                eps, sigma = cst.cst_stress_batch(B, D, U, tris)
                assert _max_rel(eps, np.tile(eps_ref, (tris.shape[0], 1))) < 1e-10, \
                    (case.key, nx, ny, diag)
                assert _max_rel(sigma, np.tile(sig_ref, (tris.shape[0], 1))) < 1e-10, \
                    (case.key, nx, ny, diag)


def test_rigid_body_translation():
    """Uniform translation: eps = 0, sigma = 0 (to round-off)."""
    for case in CASES:
        mesh = make_mesh(case.geom, 20, 4)
        D = case.material.D_plane_stress()
        U = np.zeros(mesh.ndof)
        U[0::2] = 1.0
        U[1::2] = -2.0
        for diag in DIAGS:
            tris = cst.tri_conn(mesh, diag)
            xy, det, area = cst.element_geometry(mesh, tris)
            B = cst.cst_B_batch(xy, det)
            eps, sigma = cst.cst_stress_batch(B, D, U, tris)
            assert np.max(np.abs(eps)) < 1e-12, (case.key, diag)
            # 1e-3 Pa is ~12 orders of magnitude below the O(1e11) stress scale
            assert np.max(np.abs(sigma)) < 1e-3, (case.key, diag)


def test_reaction_equals_minus_P():
    for case in CASES:
        for nx, ny in ((10, 2), (40, 8)):
            mesh = make_mesh(case.geom, nx, ny)
            for diag in DIAGS:
                r = cst.solve_cst(case.geom, case.material, case.load,
                                  nx, ny, diag, mesh=mesh)
                rel = abs(r["reaction"] + case.load) / case.load
                assert rel < 1e-10, (case.key, nx, ny, diag, rel)


def test_area_weighted_recovery_constant_field():
    """Constant stress field: sum(A s)/sum(A) must reproduce it exactly."""
    eps0 = 1.0e-3
    nu = MAT.nu
    for case in CASES:
        D = case.material.D_plane_stress()
        sig_ref = D @ np.array([eps0, -nu * eps0, 0.0])
        assert abs(sig_ref[1]) < 1e-6 and abs(sig_ref[2]) < 1e-6
        mesh = make_mesh(case.geom, 20, 4)
        x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
        U = np.zeros(mesh.ndof)
        U[0::2] = eps0 * x
        U[1::2] = -nu * eps0 * y
        for diag in DIAGS:
            tris = cst.tri_conn(mesh, diag)
            xy, det, area = cst.element_geometry(mesh, tris)
            B = cst.cst_B_batch(xy, det)
            eps, sigma = cst.cst_stress_batch(B, D, U, tris)
            n_e = tris.shape[0]
            assert _max_rel(sigma, np.tile(sig_ref, (n_e, 1))) < 1e-12
            rec_w, conn_area = cst.recover_nodal(mesh, tris, sigma, area,
                                                 weighted=True)
            rec_a, _ = cst.recover_nodal(mesh, tris, sigma, area,
                                         weighted=False)
            assert _max_rel(rec_w, np.tile(sig_ref, (mesh.nnodes, 1))) < 1e-12
            assert _max_rel(rec_a, np.tile(sig_ref, (mesh.nnodes, 1))) < 1e-12
            # spot-check the weighted formula on a boundary and an interior node
            for node in (mesh.node_index(10, 0), mesh.node_index(10, 2)):
                mask = np.any(tris == node, axis=1)
                manual = (area[mask, None] * sigma[mask]).sum(axis=0) \
                    / area[mask].sum()
                assert _max_rel(rec_w[node], manual) < 1e-12
                assert abs(conn_area[node] - area[mask].sum()) \
                    / area[mask].sum() < 1e-12


def test_s0_constant_section_stress():
    """beta=0: the raw section average equals P/A0 at x >= 0.1 (machine precision)."""
    case = CASES_BY_KEY["S0"]
    sigma_nom = case.load / (case.geom.width * case.geom.h0)
    for diag in DIAGS:
        for nx, ny in MESH_LEVELS:
            r = cst.solve_cst(case.geom, case.material, case.load, nx, ny, diag)
            p = cst.section_profile(r)
            sel = p["x"] >= 0.1 - 1e-12
            rel = np.max(np.abs(p["sigma_section_avg_raw"][sel] - sigma_nom)) \
                / sigma_nom
            assert rel < 1e-9, (diag, nx, ny, rel)


def test_section_profile_structure():
    case = CASES_BY_KEY["S1"]
    mesh = make_mesh(case.geom, 40, 8)
    r = cst.solve_case(case, 40, 8, "pos", mesh=mesh)
    p = cst.section_profile(r)
    assert p["x"].size == STATIONS.size == 11
    assert np.allclose(p["x"], STATIONS, atol=1e-12)
    assert np.allclose(p["x_m"], STATIONS * case.geom.L, atol=1e-12)
    for k, i in enumerate(p["column_index"]):
        if 0 < i < mesh.nx:
            assert p["n_tris"][k] == 4 * mesh.ny
            assert p["n_nodes"][k] == mesh.ny + 1
        else:
            assert p["n_tris"][k] == 2 * mesh.ny
            assert p["n_nodes"][k] == mesh.ny + 1


def test_solve_api_consistency():
    case = CASES_BY_KEY["S2"]
    mesh = make_mesh(case.geom, 40, 8)
    r1 = cst.solve_case(case, 40, 8, "neg")
    r2 = cst.solve_mesh(mesh, mesh.tris_neg, case)
    assert r1["diag"] == "neg" == r2["diag"]
    assert abs(r1["u_end"] - r2["u_end"]) / r1["u_end"] < 1e-15
    assert abs(r1["u_mid"] - r2["u_mid"]) / r1["u_mid"] < 1e-15
    try:
        cst.tri_conn(mesh, "diagonal")
    except ValueError:
        pass
    else:
        raise AssertionError("tri_conn must reject unknown diagonals")


# mirror symmetry and diagonal sensitivity
def test_mirror_invariance_of_global_quantities():
    """pos/neg are mirror images: u_end, u_mid, section-averaged sigma are equal."""
    for case in CASES:
        for nx, ny in ((10, 2), (40, 8)):
            mesh = make_mesh(case.geom, nx, ny)
            rp = cst.solve_cst(case.geom, case.material, case.load, nx, ny,
                               "pos", mesh=mesh)
            rn = cst.solve_cst(case.geom, case.material, case.load, nx, ny,
                               "neg", mesh=mesh)
            pp = cst.section_profile(rp)
            nom = an.exact_sigma(case.geom, case.material, case.load,
                                 pp["x_m"])
            dm = cst.diagonal_metrics(rp, rn, nom)
            assert dm["d_u_pct"] < 1e-9, (case.key, nx, ny, dm["d_u_pct"])
            assert dm["d_u_mid_pct"] < 1e-9
            assert dm["d_sigma_axial_pct"] < 1e-9, \
                (case.key, nx, ny, dm["d_sigma_axial_pct"])
            # on the mirror plane y=0 the two solutions coincide
            rows_p = cst.probe_rows(rp, case)
            rows_n = cst.probe_rows(rn, case)
            for a, b in zip(rows_p, rows_n):
                if abs(a["y"]) > 1e-12 * case.geom.L:
                    continue
                den = max(abs(a["sxx"]), abs(b["sxx"]), 1e-300)
                assert 100 * abs(a["sxx"] - b["sxx"]) / den < 1e-8, \
                    (case.key, nx, ny, a["x"])


def test_diagonal_local_sensitivity_decreases():
    """Local diagonal artifacts decay under refinement; global differences stay ~0."""
    for key in ("S1", "S4"):
        case = CASES_BY_KEY[key]
        d_el, probe_max, d_u = [], [], []
        for nx, ny in ((10, 2), (40, 8), (160, 32)):
            mesh = make_mesh(case.geom, nx, ny)
            rp = cst.solve_cst(case.geom, case.material, case.load, nx, ny,
                               "pos", mesh=mesh)
            rn = cst.solve_cst(case.geom, case.material, case.load, nx, ny,
                               "neg", mesh=mesh)
            pp = cst.section_profile(rp)
            nom = an.exact_sigma(case.geom, case.material, case.load,
                                 pp["x_m"])
            dm = cst.diagonal_metrics(rp, rn, nom)
            d_el.append(dm["d_sigma_element_pct"])
            d_u.append(dm["d_u_pct"])
            rows_p = cst.probe_rows(rp, case)
            rows_n = cst.probe_rows(rn, case)
            rel = [100 * abs(a["sxx"] - b["sxx"])
                   / max(abs(a["sxx"]), abs(b["sxx"]), 1e-300)
                   for a, b in zip(rows_p, rows_n)
                   if abs(a["y"]) > 1e-12 * case.geom.L]
            probe_max.append(max(rel))
        d_el = np.asarray(d_el)
        assert np.all(np.diff(d_el) < 0.0), (key, d_el)
        assert d_el[-1] < 0.6 * d_el[0], (key, d_el)
        assert probe_max[-1] < 0.1 * probe_max[0], (key, probe_max)
        assert all(v < 1e-9 for v in d_u), (key, d_u)


def test_beta0_u_mid_convergence():
    """beta=0: u_mid converges under refinement; residual offset is 2-D clamp effect."""
    case = CASES_BY_KEY["S0"]
    u_mid_exact_1d = an.exact_u_mid(case.geom, case.material, case.load)
    errs, u_mid = [], []
    for nx, ny in MESH_LEVELS:
        r = cst.solve_case(case, nx, ny, "pos")
        u_mid.append(r["u_mid"])
        errs.append(abs((r["u_mid"] - u_mid_exact_1d) / u_mid_exact_1d))
    errs = np.asarray(errs)
    u_mid = np.asarray(u_mid)
    assert np.all(np.diff(errs) < 0.0), errs
    assert errs[-1] < 1e-2, errs[-1]
    assert abs(u_mid[-1] - u_mid[-2]) / u_mid[-1] < 1e-3
    # pos and neg give the same u_mid to round-off
    u_mid_neg = cst.solve_case(case, MESH_LEVELS[-1][0],
                               MESH_LEVELS[-1][1], "neg")["u_mid"]
    assert abs(u_mid[-1] - u_mid_neg) / u_mid[-1] < 1e-12


def test_probe_points_on_mesh():
    """Probe coordinates are the requested x and y = 0, +-h(x)/4."""
    case = CASES_BY_KEY["S3"]
    mesh = make_mesh(case.geom, 10, 2)
    r = cst.solve_case(case, 10, 2, "pos", mesh=mesh)
    rows = cst.probe_rows(r, case)
    xs = (0.1, 0.2, 0.5, 0.8, 0.9)
    assert len(rows) == len(xs) * 3
    for k, xf in enumerate(xs):
        h = case.geom.h(xf * case.geom.L)
        tri = rows[3 * k: 3 * k + 3]
        assert abs(tri[0]["x"] - xf * case.geom.L) < 1e-15
        assert abs(tri[0]["y"]) < 1e-15
        assert abs(tri[1]["y"] - h / 4) < 1e-15
        assert abs(tri[2]["y"] + h / 4) < 1e-15
        for row in tri:
            assert set(row) == set(("case", "nx", "ny", "diag", "x", "y",
                                    "ux", "uy", "sxx", "syy", "sxy"))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("cst ok", len(fns))
