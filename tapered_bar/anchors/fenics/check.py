"""FEniCS plane-stress check."""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import basix.ufl
import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
from mpi4py import MPI
import ufl

from fem.analytic import exact_sigma, exact_u_end
from fem.config import CASES_BY_KEY, Material

E = 2.0e11
NU = 0.3
P = 1000.0
B = 0.1
L = 1.0
NX, NY = 80, 16          # structured triangle mesh resolution


def make_tapered_mesh(beta: float, h0: float, nx: int = NX, ny: int = NY):
    """Structured triangle mesh of the symmetric wedge bar (x,y=h(x)*eta)."""
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_rectangle(
        comm, np.array([[0.0, -0.5], [L, 0.5]]), [nx, ny],
        cell_type=dolfinx.mesh.CellType.triangle)
    x = mesh.geometry.x
    h = h0 * (1.0 + beta * x[:, 0] / L)
    x[:, 1] = x[:, 1] * h
    return mesh


def plane_stress_problem(mesh, h_end: float):
    V = dolfinx.fem.functionspace(mesh, basix.ufl.element("Lagrange", "triangle", 2, shape=(2,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    eps = lambda w: ufl.sym(ufl.grad(w))          # noqa: E731
    I = ufl.Identity(2)
    # plane-stress tensor form: sigma = E/(1-nu^2) * ((1-nu) eps + nu tr(eps) I)
    sigma = lambda w: E / (1.0 - NU ** 2) * ((1.0 - NU) * eps(w) + NU * ufl.tr(eps(w)) * I)  # noqa: E731

    # right boundary facets (x = L)
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    mid = dolfinx.mesh.compute_midpoints(mesh, fdim, facets)[:, :2]
    right = facets[np.isclose(mid[:, 0], L, atol=1e-10)]
    mt = dolfinx.mesh.meshtags(mesh, fdim, right, np.full(right.size, 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=1)
    traction = P / (B * h_end)                     # uniform axial end traction

    a = ufl.inner(sigma(u), eps(v)) * B * ufl.dx
    lform = ufl.inner(ufl.as_vector((traction, 0.0)), v) * B * ds

    # clamped left end
    left_dofs = dolfinx.fem.locate_dofs_geometrical(
        V, lambda p: np.isclose(p[0], 0.0, atol=1e-12))
    u_bc = dolfinx.fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = dolfinx.fem.dirichletbc(u_bc, left_dofs)

    problem = dolfinx.fem.petsc.LinearProblem(
        a, lform, bcs=[bc], petsc_options_prefix="fenics_anchor_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    return V, uh, ds


def u_end_from_solution(uh, ds, h_end: float) -> float:
    """Section-average ux at x=L (line integral / height)."""
    val = dolfinx.fem.assemble_scalar(dolfinx.fem.form(uh[0] * ds))
    return float(val / h_end)


def sigma_xx_mid(uh, x_lo: float = 0.35, x_hi: float = 0.65) -> float:
    """Area-weighted mean sigma_xx over cells with centroid x in [x_lo, x_hi]."""
    mesh = uh.function_space.mesh
    Q = dolfinx.fem.functionspace(mesh, ("DG", 0))
    eps = ufl.sym(ufl.grad(uh))
    sxx = E / (1.0 - NU ** 2) * ((1.0 - NU) * eps[0, 0] + NU * (eps[0, 0] + eps[1, 1]))
    expr = dolfinx.fem.Expression(sxx, Q.element.interpolation_points)
    sxx_dg = dolfinx.fem.Function(Q)
    sxx_dg.interpolate(expr)
    dofmaps = getattr(mesh.geometry, "dofmaps", None)
    cmap = dofmaps[0] if dofmaps else mesh.geometry.dofmap
    coords = mesh.geometry.x[cmap][:, :, :2].mean(axis=1)
    mask = (coords[:, 0] >= x_lo) & (coords[:, 0] <= x_hi)
    return float(np.mean(sxx_dg.x.array[mask]))


def run_case(case_key: str):
    case = CASES_BY_KEY[case_key]
    geom = case.geom
    mesh = make_tapered_mesh(geom.beta, geom.h0)
    h_end = float(geom.h(geom.L))
    V, uh, ds = plane_stress_problem(mesh, h_end)
    u_end = u_end_from_solution(uh, ds, h_end)
    out = dict(case=case_key, beta=float(geom.beta), h0=float(geom.h0),
               mesh=f"{NX}x{NY}", element="P2 triangles (plane stress)",
               nnodes=int(mesh.geometry.x.shape[0]),
               u_end=u_end, u_end_exact_1d=float(exact_u_end(geom, case.material, case.load)),
               sigma_mid=sigma_xx_mid(uh),
               sigma_exact_1d_mid=float(exact_sigma(geom, case.material, case.load, 0.5 * L)))
    return out


def main():
    out = dict(
        environment=dict(python=sys.version.split()[0], dolfinx=dolfinx.__version__,
                         ufl=ufl.__version__),
        model=dict(E_Pa=E, nu=NU, thickness_m=B, P_N=P,
                   note="2-D plane stress, P2 triangles, uniform end traction "
                        "t=P/(b h(L)) applied as a boundary integral, left end clamped"),
        cases={})

    s0 = run_case("S0")
    out["cases"]["S0"] = s0
    out["cases"]["S0"]["u_end_theory"] = 2.5e-7
    out["cases"]["S0"]["sigma_theory"] = 5.0e4
    assert abs(s0["u_end"] / 2.5e-7 - 1.0) < 0.01, ("S0 u_end", s0["u_end"])
    assert abs(s0["sigma_mid"] / 5.0e4 - 1.0) < 0.005, ("S0 sigma", s0["sigma_mid"])

    s1 = run_case("S1")
    ref_path = ROOT / "results" / "ccx_reference.json"
    ccx_ref = None
    if ref_path.is_file():
        ccx = json.loads(ref_path.read_text())
        c = ccx["cases"].get("S1")
        if c:
            ccx_ref = float(c["u_end"])
            s1["ccx_cps8_reference_u_end"] = ccx_ref
            s1["ccx_reference_mesh"] = c.get("mesh")
            s1["u_end_vs_ccx_pct"] = 100.0 * (s1["u_end"] / ccx_ref - 1.0)
    s1["u_end_vs_1d_pct"] = 100.0 * (s1["u_end"] / s1["u_end_exact_1d"] - 1.0)
    out["cases"]["S1"] = s1
    assert abs(s1["u_end"] / s1["u_end_exact_1d"] - 1.0) < 0.01, ("S1 u_end 1D", s1["u_end"])
    if ccx_ref is not None:
        assert abs(s1["u_end"] / ccx_ref - 1.0) < 0.005, ("S1 vs CCX", s1["u_end"], ccx_ref)
    out["status"] = "ok"

    out_path = pathlib.Path(__file__).resolve().parent / "fenics_check.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
