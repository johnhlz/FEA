"""Tests for the Q4 plane-stress element."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem import analytic as an
from fem.assembly import end_load_vector, fixed_dofs_left
from fem.config import CASES, CASES_BY_KEY, MESH_LEVELS
from fem.mesh import make_mesh
from fem.q4 import (
    b_matrix,
    element_nodes,
    evaluate_point,
    gauss_stresses,
    jacobian,
    postprocess_case,
    q4_batch,
    q4_element,
    shape_q4,
    solve_q4,
)

LINEAR = (2.0, 0.5, -0.2, 0.1, -0.3, 0.7)   # a0,a1,a2,b0,b1,b2


def _linear_field(xy, scale=1e-3):
    a = LINEAR
    ue = np.empty(8)
    for k in range(4):
        x, y = xy[k]
        ue[2 * k] = scale * (a[0] + a[1] * x + a[2] * y)
        ue[2 * k + 1] = scale * (a[3] + a[4] * x + a[5] * y)
    return ue


# element-level tests
def test_shape_functions_partition():
    for xi, eta in ((0.0, 0.0), (0.3, -0.4), (-0.7, 0.2)):
        N, dN = shape_q4(xi, eta)
        assert abs(N.sum() - 1.0) < 1e-14
        assert np.allclose(dN.sum(axis=0), 0.0, atol=1e-14)
    for k, (xi, eta) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        N, _ = shape_q4(xi, eta)
        expect = np.zeros(4)
        expect[k] = 1.0
        assert np.allclose(N, expect, atol=1e-15)


def test_jacobian_positive_and_rectangle_constant():
    for case in CASES:
        mesh = make_mesh(case.geom, 20, 4)
        data = q4_batch(element_nodes(mesh), case.material.D_plane_stress(),
                        case.geom.width)
        detJ = data["detJ"]
        assert np.all(detJ > 0.0)
        if abs(case.geom.beta) < 1e-14:
            # rectangular elements: constant Jacobian
            assert (detJ.max() - detJ.min()) / detJ.max() < 1e-12
        else:
            # tapered trapezoids: Jacobian must vary within the element
            assert (detJ.max() - detJ.min()) / detJ.max() > 1e-4


def test_constant_strain_patch():
    """Linear displacement field -> constant stress equal to D*eps."""
    case = CASES_BY_KEY["S1"]
    mesh = make_mesh(case.geom, 10, 2)
    D = case.material.D_plane_stress()
    a = LINEAR
    eps = np.array([a[1], a[5], a[2] + a[4]]) * 1e-3
    sig = D @ eps
    for conn in mesh.quads:
        xy = mesh.nodes[conn]
        _, info = q4_element(xy, D, case.geom.width)
        ue = _linear_field(xy)
        for B in info["B"]:
            s = D @ B @ ue
            assert np.max(np.abs(s - sig)) / np.max(np.abs(sig)) < 1e-10
        # boundary nodes also reproduce the field exactly
        for k in range(4):
            N, _ = shape_q4([-1, 1, 1, -1][k], [-1, -1, 1, 1][k])
            ux = N @ ue[0::2]
            uy = N @ ue[1::2]
            x, y = xy[k]
            assert abs(ux - 1e-3 * (a[0] + a[1] * x + a[2] * y)) < 1e-18
            assert abs(uy - 1e-3 * (a[3] + a[4] * x + a[5] * y)) < 1e-18


def test_rigid_body_translation():
    case = CASES_BY_KEY["S4"]
    mesh = make_mesh(case.geom, 10, 2)
    D = case.material.D_plane_stress()
    scale = case.material.E * 1e-3
    for conn in mesh.quads[:5]:
        xy = mesh.nodes[conn]
        _, info = q4_element(xy, D, case.geom.width)
        ue = np.tile([1e-3, -2e-3], 4)
        for B in info["B"]:
            sigma = D @ B @ ue
            assert np.max(np.abs(sigma)) / scale < 1e-10


def test_rectangle_2x2_gauss_is_exact():
    """For a rectangle the 2x2 rule integrates ke exactly (compare 3x3)."""
    case = CASES_BY_KEY["S0"]
    mesh = make_mesh(case.geom, 10, 2)
    xy = mesh.nodes[mesh.quads[0]]
    D = case.material.D_plane_stress()
    ke2, _ = q4_element(xy, D, case.geom.width)
    gp = np.sqrt(0.6)
    pts = [(-gp, -gp), (0.0, -gp), (gp, -gp),
           (-gp, 0.0), (0.0, 0.0), (gp, 0.0),
           (-gp, gp), (0.0, gp), (gp, gp)]
    wts = [25 / 81, 40 / 81, 25 / 81, 40 / 81, 64 / 81, 40 / 81,
           25 / 81, 40 / 81, 25 / 81]
    ke3 = np.zeros((8, 8))
    for (xi, eta), w in zip(pts, wts):
        N, dN = shape_q4(xi, eta)
        J, detJ, invJ, G = jacobian(xy, dN)
        B = b_matrix(G[:, 0], G[:, 1])
        ke3 += w * detJ * case.geom.width * (B.T @ D @ B)
    assert np.max(np.abs(ke2 - ke3)) / np.max(np.abs(ke3)) < 1e-10


def test_trapezoid_2x2_differs_from_3x3_but_converges():
    """Trapezoid integrand is rational: 2x2 and 3x3 differ (full, not exact)."""
    case = CASES_BY_KEY["S1"]
    mesh = make_mesh(case.geom, 10, 2)
    xy = mesh.nodes[mesh.quads[3]]
    D = case.material.D_plane_stress()
    ke2, _ = q4_element(xy, D, case.geom.width)
    gp = np.sqrt(0.6)
    pts = [(-gp, -gp), (0.0, -gp), (gp, -gp),
           (-gp, 0.0), (0.0, 0.0), (gp, 0.0),
           (-gp, gp), (0.0, gp), (gp, gp)]
    wts = [25 / 81, 40 / 81, 25 / 81, 40 / 81, 64 / 81, 40 / 81,
           25 / 81, 40 / 81, 25 / 81]
    ke3 = np.zeros((8, 8))
    for (xi, eta), w in zip(pts, wts):
        N, dN = shape_q4(xi, eta)
        J, detJ, invJ, G = jacobian(xy, dN)
        B = b_matrix(G[:, 0], G[:, 1])
        ke3 += w * detJ * case.geom.width * (B.T @ D @ B)
    rel = np.max(np.abs(ke2 - ke3)) / np.max(np.abs(ke3))
    assert 1e-12 < rel < 1e-2      # differs, but only mildly (mild taper)


# assembled-model tests
def test_reaction_equals_minus_P():
    P = 1000.0
    for case in CASES:
        for nx, ny in ((10, 2), (20, 4)):
            mesh = make_mesh(case.geom, nx, ny)
            res = solve_q4(mesh, case)
            reaction = float(res["reactions"][2 * mesh.left_nodes].sum())
            assert abs(reaction + P) / P < 1e-8
            # both load modes satisfy global equilibrium
            for mode in ("consistent", "lumped"):
                F = end_load_vector(mesh, case.load, mode=mode)
                assert abs(F.sum() - P) < 1e-9 * P


def test_beta0_mid_deviation_decreases():
    """S0: |u_mid - 1.25e-7| must shrink with refinement."""
    case = CASES_BY_KEY["S0"]
    devs = []
    for nx, ny in ((10, 2), (20, 4), (40, 8), (80, 16)):
        mesh = make_mesh(case.geom, nx, ny)
        res = solve_q4(mesh, case)
        u_mid = float(res["U"][0::2][mesh.column_nodes(mesh.mid_index)].mean())
        devs.append(abs(u_mid - 1.25e-7) / 1.25e-7)
    assert all(devs[i + 1] < devs[i] for i in range(len(devs) - 1)), devs
    assert devs[-1] < 1e-2


def test_mid_convergence_80_160():
    """80x16 vs 160x32 mid-section displacement difference < 0.05%."""
    for key in ("S0", "S1"):
        case = CASES_BY_KEY[key]
        vals = {}
        for nx, ny in ((80, 16), (160, 32)):
            mesh = make_mesh(case.geom, nx, ny)
            res = solve_q4(mesh, case)
            vals[nx] = float(res["U"][0::2][mesh.column_nodes(
                mesh.mid_index)].mean())
        gap = abs(vals[160] - vals[80]) / abs(vals[160])
        assert gap < 5e-4, f"{key}: gap {gap:.3e}"


def test_probe_point_evaluation():
    """Point evaluation at a node reproduces the nodal displacements."""
    case = CASES_BY_KEY["S1"]
    for nx, ny in ((10, 2), (20, 4), (40, 8)):
        mesh = make_mesh(case.geom, nx, ny)
        res = solve_q4(mesh, case)
        U = res["U"]
        for x in (0.1, 0.5, 0.9):
            for y in (0.0, 0.25 * case.geom.h(x), -0.25 * case.geom.h(x)):
                pt = evaluate_point(mesh, case, U, x, y)
                # displacement must be continuous and match the FE field
                assert np.isfinite(pt["sxx"]) and np.isfinite(pt["uy"])
                assert pt["n_elems"] >= 1
        # exact node check at (0.5, 0)
        pt = evaluate_point(mesh, case, U, 0.5, 0.0)
        node = mesh.node_index(mesh.mid_index, mesh.ny // 2)
        assert abs(pt["ux"] - U[2 * node]) < 1e-20
        assert abs(pt["uy"] - U[2 * node + 1]) < 1e-20


def test_postprocess_schema_and_accuracy():
    case = CASES_BY_KEY["S1"]
    mesh = make_mesh(case.geom, 40, 8)
    res = solve_q4(mesh, case)
    post = postprocess_case(mesh, case, res["U"], res["reactions"],
                            data=res["data"])
    assert len(post["profile_rows"]) == 11
    assert len(post["gradient_rows"]) == 9
    rr = post["run_row"]
    assert rr["sigma_profile_rms_pct"] < 0.05
    assert rr["grad_profile_rms_pct"] < 5.0
    assert post["extras"]["detJ_min"] > 0.0
    # section stress at x=L/2 close to P/A
    sigma_mid_nom = float(an.exact_sigma(case.geom, case.material, case.load,
                                         0.5))
    assert abs(rr["sigma_mid"] - sigma_mid_nom) / sigma_mid_nom < 2e-3
    # u_end > u_mid > 0
    assert rr["u_end"] > rr["u_mid"] > 0.0


def test_gradient_fidelity_improves():
    case = CASES_BY_KEY["S1"]
    errs = []
    for nx, ny in ((20, 4), (40, 8), (80, 16)):
        mesh = make_mesh(case.geom, nx, ny)
        res = solve_q4(mesh, case)
        post = postprocess_case(mesh, case, res["U"], res["reactions"],
                                data=res["data"])
        errs.append(post["run_row"]["grad_profile_rms_pct"])
    assert errs[0] > errs[1] > errs[2]
    # our finest in-house gradient error stays small (stencil floor ~0.6%)
    assert errs[-1] < 2.0


def test_shear_energy_ratio_ordering():
    """The shear/axial energy ratio should grow with the taper index eta."""
    ratios = {}
    for key in ("S0", "S1", "S3", "S4"):
        case = CASES_BY_KEY[key]
        mesh = make_mesh(case.geom, 40, 8)
        res = solve_q4(mesh, case)
        post = postprocess_case(mesh, case, res["U"], res["reactions"],
                                data=res["data"])
        ratios[key] = post["run_row"]["shear_energy_ratio"]
    assert ratios["S0"] < ratios["S1"] < ratios["S3"] < ratios["S4"], ratios
    assert ratios["S0"] < 0.01


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("q4 ok")
