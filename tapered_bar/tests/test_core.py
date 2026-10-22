"""Core invariants: analytic solution, mesh geometry, constitutive matrix."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem.config import CASES, Geometry, MAT, LOAD, MESH_LEVELS, STATIONS
from fem.mesh import make_mesh, mesh_quality
from fem import analytic as an


def test_analytic_limits():
    g = Geometry(beta=0.0)
    A = g.A0
    u = an.exact_u_end(g, MAT, LOAD)
    assert abs(u - LOAD * g.L / (MAT.E * A)) / u < 1e-13
    # small beta expansion: u(L) -> P L/(EA) * (1 - beta/2 + beta^2/3 ...)
    for beta in (1e-2, 1e-4):
        gb = Geometry(beta=beta)
        ub = an.exact_u_end(gb, MAT, LOAD)
        ref = LOAD * gb.L / (MAT.E * gb.A0) * (1 - beta / 2 + beta ** 2 / 3)
        assert abs(ub - ref) / ref < 1e-6
    # monotone and positive for all cases
    for case in CASES:
        g = case.geom
        assert g.h_min() > 0
        xs = np.linspace(0, g.L, 51)
        u = an.exact_u(g, MAT, LOAD, xs)
        assert np.all(np.diff(u) > 0)
        assert abs(u[0]) < 1e-30


def test_compliance_additive():
    for case in CASES:
        g = case.geom
        total = an.exact_compliance(g, MAT, 0.0, g.L)
        parts = sum(an.exact_compliance(g, MAT, a, b)
                    for a, b in zip(np.linspace(0, g.L, 11)[:-1],
                                    np.linspace(0, g.L, 11)[1:]))
        assert abs(total - parts) / total < 1e-13
        # exact stiffness must reproduce the exact end displacement under P
        k = an.bar_equiv_stiffness(g, MAT, 0.0, g.L)
        u_end = LOAD / k
        assert abs(u_end - an.exact_u_end(g, MAT, LOAD)) / u_end < 1e-13


def test_sigma_gradient_finite_difference():
    for case in CASES:
        g = case.geom
        x = 0.37 * g.L
        hstep = 1e-6 * g.L
        fd = (an.exact_sigma(g, MAT, LOAD, x + hstep)
              - an.exact_sigma(g, MAT, LOAD, x - hstep)) / (2 * hstep)
        gx = an.exact_sigma_gradient(g, MAT, LOAD, x)
        assert abs(fd - gx) / max(abs(gx), 1e-30) < 1e-8


def test_stiffness_integrals():
    """For linear A(x), 2-point Gauss equals the arithmetic mean-area stiffness,"""
    for case in CASES:
        g = case.geom
        k_eq = an.bar_equiv_stiffness(g, MAT, 0.0, g.L)
        k_mean = an.bar_mean_stiffness(g, MAT, 0.0, g.L)
        k_gauss = an.bar_gauss_stiffness(g, MAT, 0.0, g.L)
        assert abs(k_gauss - k_mean) / k_mean < 1e-12
        assert abs(k_eq - k_mean) / k_mean < 1e-2 if abs(g.beta) < 0.1 else True
        # series decomposition of the exact compliance is exact
        xmid = 0.5 * g.L
        k_eq2 = 1.0 / (1.0 / an.bar_equiv_stiffness(g, MAT, 0, xmid)
                       + 1.0 / an.bar_equiv_stiffness(g, MAT, xmid, g.L))
        assert abs(k_eq - k_eq2) / k_eq < 1e-13


def test_end_load_vectors():
    """Consistent traction loads: total = P, corners half of interior nodes."""
    from fem.config import CASES
    from fem.mesh import make_mesh
    from fem.assembly import end_load_vector
    for case in CASES:
        g = case.geom
        m = make_mesh(g, 20, 4)
        F = end_load_vector(m, case.load, mode="consistent")
        assert abs(F[0::2].sum() - case.load) / case.load < 1e-12
        assert abs(F[1::2].sum()) < 1e-9
        ux = F[2 * m.right_nodes]
        assert abs(ux[0] - 0.5 * ux[1]) < 1e-12 * ux.max()
        Fl = end_load_vector(m, case.load, mode="lumped")
        assert abs(Fl[0::2].sum() - case.load) / case.load < 1e-12
        assert abs(Fl[2 * m.right_nodes[0]] - case.load / (m.ny + 1)) < 1e-12


def test_mesh_geometry():
    for case in CASES:
        g = case.geom
        for nx, ny in MESH_LEVELS:
            m = make_mesh(g, nx, ny)
            q = mesh_quality(m)
            expect_area = g.L * g.h0 * (1 + g.beta / 2)  # ∫h dx
            assert abs(q["area_sum"] - expect_area) / expect_area < 1e-12
            assert q["min_quad_signed_area"] > 0
            assert q["min_tri_area_pos"] > 0
            assert q["min_tri_area_neg"] > 0
            assert m.nnodes == (nx + 1) * (ny + 1)
            assert m.nquads == nx * ny
            assert m.tris_pos.shape == (2 * nx * ny, 3)
            assert m.tris_neg.shape == (2 * nx * ny, 3)
            # columns at the common stations
            for x in STATIONS:
                i = int(round(x * nx))
                assert abs(m.column_x(i) - x * g.L) < 1e-12
            # top boundary follows the taper exactly
            for i in (0, nx // 2, nx):
                top = m.node_index(i, ny)
                assert abs(m.nodes[top, 1] - g.h(m.column_x(i)) / 2) < 1e-14


def test_constitutive():
    D = MAT.D_plane_stress()
    assert np.allclose(D, D.T)
    assert np.all(np.linalg.eigvalsh(D) > 0)
    eps = np.array([1.0, 0.0, 0.0])
    sig = D @ eps
    assert abs(sig[0] - MAT.E / (1 - MAT.nu ** 2)) < 1e-6
    assert abs(sig[1] - MAT.nu * MAT.E / (1 - MAT.nu ** 2)) < 1e-6
    assert abs(sig[2]) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("core ok")
