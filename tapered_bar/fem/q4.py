"""Q4 bilinear plane-stress element."""
from __future__ import annotations

import numpy as np

from . import analytic
from .assembly import (
    assemble_sparse,
    dirichlet_solve,
    dofs_2d,
    end_load_vector,
    fixed_dofs_left,
)
from .config import INTERIOR_STATIONS, STATIONS, Case
from .mesh import Mesh

NEN = 4           # nodes per element
NDOF_E = 2 * NEN  # 8 element dofs

_SQ3 = 1.0 / np.sqrt(3.0)
#: 2x2 Gauss points (xi, eta) and weights (all 1.0), tensor product of
#: 1D rules with points +-1/sqrt(3); exact for polynomials up to degree 3.
GAUSS_POINTS_2X2 = np.array(
    [[-_SQ3, -_SQ3], [_SQ3, -_SQ3], [_SQ3, _SQ3], [-_SQ3, _SQ3]], dtype=float
)
GAUSS_WEIGHTS_2X2 = np.ones(4, dtype=float)


def shape_q4(xi: float, eta: float):
    """Bilinear shape functions and their (xi, eta) derivatives."""
    xi = float(xi)
    eta = float(eta)
    N = 0.25 * np.array(
        [(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]
    )
    dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, np.column_stack([dxi, deta])


def jacobian(nodes_xy, dN):
    """Jacobian, determinant, inverse and physical shape gradients."""
    nodes_xy = np.asarray(nodes_xy, dtype=float)
    dN = np.asarray(dN, dtype=float)
    J = nodes_xy.T @ dN
    detJ = float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
    if detJ <= 0.0:
        raise ValueError(f"non-positive Jacobian detJ={detJ!r}")
    invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ
    # v_ref = J^T g_phys  =>  g_phys = invJ^T v_ref
    G = dN @ invJ
    return J, detJ, invJ, G


def b_matrix(dNdx, dNdy) -> np.ndarray:
    """Strain-displacement matrix B (3 x 8)."""
    dNdx = np.asarray(dNdx, dtype=float)
    dNdy = np.asarray(dNdy, dtype=float)
    B = np.zeros((3, NDOF_E))
    B[0, 0::2] = dNdx
    B[1, 1::2] = dNdy
    B[2, 0::2] = dNdy
    B[2, 1::2] = dNdx
    return B


def q4_element(nodes_xy, D, thickness: float = 1.0, n_gauss: int = 2):
    """Single Q4 element stiffness with 2x2 Gauss full integration."""
    if n_gauss != 2:
        raise ValueError("Q4 module is restricted to 2x2 full integration")
    nodes_xy = np.asarray(nodes_xy, dtype=float)
    D = np.asarray(D, dtype=float)
    ke = np.zeros((NDOF_E, NDOF_E))
    Bs = np.empty((4, 3, NDOF_E))
    detJs = np.empty(4)
    gxy = np.empty((4, 2))
    for g, (xi, eta) in enumerate(GAUSS_POINTS_2X2):
        N, dN = shape_q4(xi, eta)
        J, detJ, invJ, G = jacobian(nodes_xy, dN)
        B = b_matrix(G[:, 0], G[:, 1])
        w = GAUSS_WEIGHTS_2X2[g]
        ke += w * detJ * thickness * (B.T @ D @ B)
        Bs[g] = B
        detJs[g] = detJ
        gxy[g] = N @ nodes_xy
    info = dict(B=Bs, detJ=detJs, gauss_xy=gxy, weights=GAUSS_WEIGHTS_2X2,
                local_gauss=GAUSS_POINTS_2X2, D=D, thickness=thickness)
    return ke, info


# Vectorised element batch (all elements at once)
def q4_batch(nodes_all, D, thickness: float = 1.0, n_gauss: int = 2):
    """Vectorised 2x2 Gauss stiffness/stress data for a batch of Q4 elements."""
    if n_gauss != 2:
        raise ValueError("Q4 module is restricted to 2x2 full integration")
    nodes_all = np.asarray(nodes_all, dtype=float)
    if nodes_all.ndim != 3 or nodes_all.shape[1:] != (NEN, 2):
        raise ValueError("nodes_all must have shape (ne, 4, 2)")
    D = np.asarray(D, dtype=float)
    ne = nodes_all.shape[0]

    ke = np.zeros((ne, NDOF_E, NDOF_E))
    Bs = np.empty((ne, 4, 3, NDOF_E))
    detJs = np.empty((ne, 4))
    gxy = np.empty((ne, 4, 2))

    for g, (xi, eta) in enumerate(GAUSS_POINTS_2X2):
        N, dN = shape_q4(xi, eta)
        # J[n,i,j] = sum_a nodes[n,a,i] dN[a,j]
        J = np.einsum("nai,aj->nij", nodes_all, dN)
        detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        if np.any(detJ <= 0.0):
            bad = int(np.argmax(detJ <= 0.0))
            raise ValueError(f"non-positive Jacobian in element {bad}")
        invJ = np.empty((ne, 2, 2))
        invJ[:, 0, 0] = J[:, 1, 1]
        invJ[:, 0, 1] = -J[:, 0, 1]
        invJ[:, 1, 0] = -J[:, 1, 0]
        invJ[:, 1, 1] = J[:, 0, 0]
        invJ /= detJ[:, None, None]
        # G[n,a,i] = sum_j dN[a,j] invJ[n,j,i]  (g = invJ^T v_ref)
        G = np.einsum("aj,nji->nai", dN, invJ)
        B = np.zeros((ne, 3, NDOF_E))
        B[:, 0, 0::2] = G[:, :, 0]
        B[:, 1, 1::2] = G[:, :, 1]
        B[:, 2, 0::2] = G[:, :, 1]
        B[:, 2, 1::2] = G[:, :, 0]
        DB = np.einsum("ij,njk->nik", D, B)
        ke += GAUSS_WEIGHTS_2X2[g] * detJ[:, None, None] * thickness * np.einsum(
            "nci,ncj->nij", B, DB
        )
        Bs[:, g] = B
        detJs[:, g] = detJ
        gxy[:, g] = np.einsum("a,nai->ni", N, nodes_all)

    return dict(ke=ke, B=Bs, detJ=detJs, gauss_xy=gxy,
                weights=GAUSS_WEIGHTS_2X2, D=D, thickness=thickness)


# Assembly / solution using the shared assembly core
def element_nodes(mesh: Mesh) -> np.ndarray:
    """(ne, 4, 2) coordinates of all quad elements."""
    return mesh.nodes[mesh.quads]


def element_displacements(mesh: Mesh, U) -> np.ndarray:
    """(ne, 8) element displacement vectors from the global dof vector."""
    U = np.asarray(U, dtype=float)
    conn = mesh.quads
    d = np.empty((mesh.nquads, NDOF_E), dtype=int)
    d[:, 0::2] = 2 * conn
    d[:, 1::2] = 2 * conn + 1
    return U[d]


def assemble_q4(mesh: Mesh, D, thickness: float):
    """Sparse global stiffness of a Q4 mesh (2x2 Gauss full integration)."""
    data = q4_batch(element_nodes(mesh), D, thickness)
    K = assemble_sparse(mesh.ndof, mesh.quads, list(data["ke"]))
    return K, data


def consistent_end_load_vector(mesh: Mesh, P: float) -> np.ndarray:
    """Consistent nodal load of a uniform axial traction on the right end."""
    return end_load_vector(mesh, P)


def solve_q4(mesh: Mesh, case: Case, load_vector=None):
    """Assemble and solve the plane-stress Q4 model for one case / mesh."""
    D = case.material.D_plane_stress()
    K, data = assemble_q4(mesh, D, case.geom.width)
    if load_vector is None:
        F = end_load_vector(mesh, case.load)
    else:
        F = np.asarray(load_vector, dtype=float).copy()
    U, reactions = dirichlet_solve(K, F, fixed_dofs_left(mesh))
    return dict(U=U, reactions=reactions, F=F, K=K, data=data, D=D)


def gauss_stresses(mesh: Mesh, U, D, thickness: float, data=None):
    """Stress sigma = D B u_e at every 2x2 Gauss point of every element."""
    if data is None:
        data = q4_batch(element_nodes(mesh), D, thickness)
    B = data["B"]
    ue = element_displacements(mesh, U)
    strain = np.einsum("ngck,nk->ngc", B, ue)
    sigma = np.einsum("cd,ngd->ngc", np.asarray(D, dtype=float), strain)
    return sigma


# Point location / evaluation at arbitrary physical points
def inverse_map(nodes_xy, x: float, y: float, xi0: float = 0.0,
                eta0: float = 0.0, maxit: int = 30, tol: float = 1e-13):
    """Invert the bilinear isoparametric map by Newton iteration."""
    nodes_xy = np.asarray(nodes_xy, dtype=float)
    target = np.array([float(x), float(y)])
    xi, eta = float(xi0), float(eta0)
    scale = max(1.0, float(np.abs(target).max()))
    for _ in range(maxit):
        N, dN = shape_q4(xi, eta)
        xy = N @ nodes_xy
        r = xy - target
        if float(np.linalg.norm(r)) <= tol * scale:
            return xi, eta
        J, detJ, invJ, G = jacobian(nodes_xy, dN)
        delta = invJ @ r
        xi -= float(delta[0])
        eta -= float(delta[1])
    raise ValueError(f"inverse_map did not converge for point ({x}, {y})")


def find_point_element(mesh: Mesh, x: float, y: float, tol: float = 1e-8):
    """Locate the Q4 element containing physical point (x, y)."""
    nx, ny = mesh.nx, mesh.ny
    x_cols = np.array([mesh.column_x(i) for i in range(nx + 1)])
    x = float(np.clip(x, x_cols[0], x_cols[-1]))
    hx = float(mesh.geom.h(x))
    i_hi = int(np.searchsorted(x_cols, x, side="right") - 1)
    i_hi = min(max(i_hi, 0), nx - 1)
    cols = [i for i in (i_hi - 1, i_hi, i_hi + 1) if 0 <= i < nx]
    eta_param = y / hx if hx > 0.0 else 0.0
    j_hi = int(np.floor((eta_param + 0.5) * ny))
    j_hi = min(max(j_hi, 0), ny - 1)
    rows = [j for j in (j_hi - 1, j_hi, j_hi + 1) if 0 <= j < ny]

    def _try(ei):
        conn = mesh.quads[ei]
        xy = mesh.nodes[conn]
        i = ei // ny
        xl, xr = x_cols[i], x_cols[i + 1]
        xi0 = 2.0 * (x - xl) / (xr - xl) - 1.0
        eta0 = 2.0 * y / hx if hx > 0.0 else 0.0
        try:
            xi, eta = inverse_map(xy, x, y, xi0=xi0, eta0=eta0)
        except (ValueError, np.linalg.LinAlgError):
            return None
        if abs(xi) <= 1.0 + tol and abs(eta) <= 1.0 + tol:
            return (int(ei), float(min(max(xi, -1.0), 1.0)),
                    float(min(max(eta, -1.0), 1.0)))
        return None

    for i in cols:
        for j in rows:
            hit = _try(i * ny + j)
            if hit is not None:
                return hit
    for ei in range(mesh.nquads):          # robust fallback
        hit = _try(ei)
        if hit is not None:
            return hit
    raise ValueError(f"point ({x}, {y}) not inside the mesh")


def elements_containing_point(mesh: Mesh, x: float, y: float, tol: float = 1e-8):
    """All Q4 elements whose closure contains the physical point (x, y)."""
    nx, ny = mesh.nx, mesh.ny
    x_cols = np.array([mesh.column_x(i) for i in range(nx + 1)])
    x = float(np.clip(x, x_cols[0], x_cols[-1]))
    hx = float(mesh.geom.h(x))
    i_hi = int(np.searchsorted(x_cols, x, side="right") - 1)
    i_hi = min(max(i_hi, 0), nx - 1)
    cols = [i for i in (i_hi - 1, i_hi, i_hi + 1) if 0 <= i < nx]
    eta_param = y / hx if hx > 0.0 else 0.0
    j_hi = int(np.floor((eta_param + 0.5) * ny))
    j_hi = min(max(j_hi, 0), ny - 1)
    rows = [j for j in range(max(j_hi - 1, 0), min(j_hi + 2, ny))]

    def _try(ei):
        conn = mesh.quads[ei]
        xy = mesh.nodes[conn]
        i = ei // ny
        xl, xr = x_cols[i], x_cols[i + 1]
        xi0 = 2.0 * (x - xl) / (xr - xl) - 1.0
        # eta guess for the structured tapered element (eta -> y is affine)
        eta0 = eta_param * ny - (ei % ny) + 0.5
        try:
            xi, eta = inverse_map(xy, x, y, xi0=xi0, eta0=eta0)
        except (ValueError, np.linalg.LinAlgError):
            return None
        if abs(xi) <= 1.0 + tol and abs(eta) <= 1.0 + tol:
            return (int(ei), float(min(max(xi, -1.0), 1.0)),
                    float(min(max(eta, -1.0), 1.0)))
        return None

    hits = []
    for i in cols:
        for j in rows:
            hit = _try(i * ny + j)
            if hit is not None:
                hits.append(hit)
    if not hits:
        for ei in range(mesh.nquads):
            hit = _try(ei)
            if hit is not None:
                hits.append(hit)
    # unique by element index
    seen = set()
    uniq = []
    for hit in hits:
        if hit[0] not in seen:
            seen.add(hit[0])
            uniq.append(hit)
    if not uniq:
        raise ValueError(f"point ({x}, {y}) not inside the mesh")
    return uniq


def _point_value(mesh, case, U, ei, xi, eta):
    """Displacement / stress at reference coordinates of one element."""
    D = case.material.D_plane_stress()
    conn = mesh.quads[ei]
    nodes_xy = mesh.nodes[conn]
    N, dN = shape_q4(xi, eta)
    J, detJ, invJ, G = jacobian(nodes_xy, dN)
    B = b_matrix(G[:, 0], G[:, 1])
    ue = np.asarray(U, dtype=float)[dofs_2d(conn)]
    strain = B @ ue
    sigma = D @ strain
    return dict(
        elem=int(ei), xi=float(xi), eta=float(eta), detJ=float(detJ),
        ux=float(N @ ue[0::2]), uy=float(N @ ue[1::2]),
        sxx=float(sigma[0]), syy=float(sigma[1]), sxy=float(sigma[2]),
    )


def evaluate_point(mesh: Mesh, case: Case, U, x: float, y: float, data=None,
                   average_shared: bool = True):
    """Pointwise displacement and stress at an arbitrary physical point."""
    hits = elements_containing_point(mesh, x, y)
    vals = [_point_value(mesh, case, U, ei, xi, eta) for ei, xi, eta in hits]
    if average_shared and len(vals) > 1:
        out = dict(
            elem=vals[0]["elem"], xi=vals[0]["xi"], eta=vals[0]["eta"],
            x=float(x), y=float(y),
            detJ=float(np.mean([v["detJ"] for v in vals])),
            ux=float(np.mean([v["ux"] for v in vals])),
            uy=float(np.mean([v["uy"] for v in vals])),
            sxx=float(np.mean([v["sxx"] for v in vals])),
            syy=float(np.mean([v["syy"] for v in vals])),
            sxy=float(np.mean([v["sxy"] for v in vals])),
            n_elems=len(vals),
        )
    else:
        out = dict(vals[0])
        out.update(x=float(x), y=float(y), n_elems=len(vals))
    return out


# Post-processing: section profiles, gradient fidelity, energy ratio
def _station_elements(mesh: Mesh):
    """Element indices adjacent to each node column i = 0..nx."""
    ne = mesh.nquads
    elem_col = np.arange(ne) // mesh.ny
    out = []
    for i in range(mesh.nx + 1):
        if i == 0:
            idx = np.flatnonzero(elem_col == 0)
        elif i == mesh.nx:
            idx = np.flatnonzero(elem_col == mesh.nx - 1)
        else:
            idx = np.flatnonzero((elem_col == i - 1) | (elem_col == i))
        out.append(idx)
    return out


def postprocess_case(mesh: Mesh, case: Case, U, reactions, data=None):
    """All per-mesh post-processing for the Q4 study."""
    geom, mat, P = case.geom, case.material, case.load
    D = mat.D_plane_stress()
    if data is None:
        data = q4_batch(element_nodes(mesh), D, geom.width)
    sigma_gp = gauss_stresses(mesh, U, D, geom.width, data=data)  # (ne,4,3)

    # nx is a multiple of 10, so column i = round(s*nx) is exactly x = s*L.
    ux = np.asarray(U, dtype=float)[0::2]
    adj = _station_elements(mesh)
    station_idx = np.array([int(round(float(s) * mesh.nx)) for s in STATIONS])
    x_st = np.array([mesh.column_x(i) for i in station_idx])
    if not np.allclose(x_st, geom.L * STATIONS, rtol=0.0, atol=1e-12):
        raise AssertionError("mesh node columns do not match STATIONS")
    u_sec = np.array([ux[mesh.column_nodes(i)].mean() for i in station_idx])
    sigma_sec = np.array([sigma_gp[adj[i], :, 0].mean() for i in station_idx])

    sigma_nom = analytic.exact_sigma(geom, mat, P, x_st)
    grad_nom = analytic.exact_sigma_gradient(geom, mat, P, x_st)

    dx = x_st[1] - x_st[0]
    grad_fd = np.empty_like(x_st)
    grad_fd[1:-1] = (sigma_sec[2:] - sigma_sec[:-2]) / (2.0 * dx)
    # endpoints: second-order one-sided differences (profile CSV is complete)
    grad_fd[0] = (-3.0 * sigma_sec[0] + 4.0 * sigma_sec[1]
                  - sigma_sec[2]) / (2.0 * dx)
    grad_fd[-1] = (3.0 * sigma_sec[-1] - 4.0 * sigma_sec[-2]
                   + sigma_sec[-3]) / (2.0 * dx)

    # interior station mask: x = 0.1 ... 0.9
    mask = np.isin(np.round(x_st, 10), np.round(geom.L * INTERIOR_STATIONS, 10))
    if mask.sum() != INTERIOR_STATIONS.size:
        raise AssertionError("interior station mask incomplete")

    s_fem = sigma_sec[mask]
    s_nom = sigma_nom[mask]
    sigma_rms_pct = 100.0 * np.sqrt(np.mean((s_fem - s_nom) ** 2)) / np.sqrt(
        np.mean(s_nom ** 2)
    )

    g_fem = grad_fd[mask]
    g_nom = grad_nom[mask]
    if abs(geom.beta) < 1e-14:
        grad_rms_pct = float("nan")  # nominal gradient vanishes identically
    else:
        grad_rms_pct = 100.0 * np.sqrt(np.mean((g_fem - g_nom) ** 2)) / np.sqrt(
            np.mean(g_nom ** 2)
        )

    # 4th-order 5-point derivative on the same profile: pure stencil-truncation
    # study.  The INTERFACES gradient stays the 2nd-order central difference.
    grad5 = np.full_like(x_st, np.nan)
    for k in range(2, x_st.size - 2):
        grad5[k] = (-sigma_sec[k + 2] + 8.0 * sigma_sec[k + 1]
                    - 8.0 * sigma_sec[k - 1] + sigma_sec[k - 2]) / (12.0 * dx)
    mask5 = np.isin(np.round(x_st, 10),
                    np.round(geom.L * np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), 10))
    if abs(geom.beta) < 1e-14:
        grad5_rms_pct = float("nan")
    else:
        grad5_rms_pct = 100.0 * np.sqrt(np.mean(
            (grad5[mask5] - grad_nom[mask5]) ** 2)) / np.sqrt(
            np.mean(grad_nom[mask5] ** 2))

    u_end = float(ux[mesh.right_nodes].mean())
    u_mid = float(ux[mesh.column_nodes(mesh.mid_index)].mean())
    u_end_exact = analytic.exact_u_end(geom, mat, P)
    u_mid_exact = analytic.exact_u_mid(geom, mat, P)
    e_end_pct = 100.0 * (u_end - u_end_exact) / u_end_exact
    e_mid_pct = 100.0 * (u_mid - u_mid_exact) / u_mid_exact

    mid_st = int(np.flatnonzero(np.isclose(STATIONS, 0.5))[0])  # x = L/2 station
    sigma_mid = float(sigma_sec[mid_st])
    sigma_mid_exact = float(analytic.exact_sigma(geom, mat, P, x_st[mid_st]))
    e_sigma_mid_pct = 100.0 * (sigma_mid - sigma_mid_exact) / sigma_mid_exact

    reaction_x = float(np.asarray(reactions)[2 * mesh.left_nodes].sum())
    reaction_rel_err = abs(reaction_x + P) / P

    w = data["weights"]                       # (4,)
    vol = w[None, :] * data["detJ"] * geom.width   # (ne,4)
    sx = sigma_gp[:, :, 0]
    sy = sigma_gp[:, :, 1]
    txy = sigma_gp[:, :, 2]
    trans_energy = float(np.sum((sy ** 2 + txy ** 2) * vol))
    axial_energy = float(np.sum(sx ** 2 * vol))
    shear_energy_ratio = trans_energy / axial_energy

    run_row = dict(
        case=case.key, nx=mesh.nx, ny=mesh.ny,
        nnodes=mesh.nnodes, ndof=mesh.ndof,
        u_end=u_end, u_mid=u_mid,
        u_end_exact=u_end_exact, u_mid_exact=u_mid_exact,
        e_end_pct=e_end_pct, e_mid_pct=e_mid_pct,
        sigma_mid=sigma_mid, sigma_mid_exact=sigma_mid_exact,
        e_sigma_mid_pct=e_sigma_mid_pct,
        sigma_profile_rms_pct=sigma_rms_pct,
        grad_profile_rms_pct=grad_rms_pct,
        shear_energy_ratio=shear_energy_ratio,
        reaction_x=reaction_x,
        note="q4_bilinear_2x2_full_integration",
    )

    profile_rows = []
    for k in range(x_st.size):
        profile_rows.append(dict(
            case=case.key, nx=mesh.nx, ny=mesh.ny, x=float(x_st[k]),
            u_section_avg=float(u_sec[k]),
            sigma_section_avg=float(sigma_sec[k]),
            sigma_nom=float(sigma_nom[k]),
            dsigma_dx_fem=float(grad_fd[k]),
            dsigma_dx_nom=float(grad_nom[k]),
        ))

    gradient_rows = []
    for k in np.flatnonzero(mask):
        gradient_rows.append(dict(
            case=case.key, nx=mesh.nx, ny=mesh.ny, x=float(x_st[k]),
            dsigma_dx_fem=float(grad_fd[k]),
            dsigma_dx_nom=float(grad_nom[k]),
        ))

    # Area-weighted (trapezoidal) section means; the INTERFACES arithmetic node
    # mean carries an O(1/ny) estimator bias relative to the true section mean.
    def _trap_mean(vals):
        ww = np.ones(vals.size)
        ww[0] = ww[-1] = 0.5
        return float(np.dot(ww, vals) / ww.sum())

    u_mid_weighted = _trap_mean(ux[mesh.column_nodes(mesh.mid_index)])
    u_end_weighted = _trap_mean(ux[mesh.right_nodes])

    extras = dict(
        u_mid_weighted=u_mid_weighted,
        u_end_weighted=u_end_weighted,
        u_mid_arith_bias_pct=float(
            100.0 * (u_mid - u_mid_weighted) / u_mid_weighted
        ),
        detJ_min=float(data["detJ"].min()),
        detJ_max=float(data["detJ"].max()),
        detJ_ratio=float(data["detJ"].max() / data["detJ"].min()),
        reaction_rel_err=float(reaction_rel_err),
        reaction_x=reaction_x,
        axial_energy=axial_energy,
        transverse_energy=trans_energy,
        grad_5point_rms_pct=float(grad5_rms_pct),
        grad_endpoint_scheme="one-sided second-order differences at x=0 and x=L",
        x_stations=[float(v) for v in x_st],
        sigma_section_avg=[float(v) for v in sigma_sec],
        u_section_avg=[float(v) for v in u_sec],
        sigma_profile_rms_pct=float(sigma_rms_pct),
        grad_profile_rms_pct=float(grad_rms_pct),
    )
    return dict(run_row=run_row, profile_rows=profile_rows,
                gradient_rows=gradient_rows, extras=extras)


def analyze_case(mesh: Mesh, case: Case, probe_points=None, load_vector=None):
    """Solve one case/mesh and return full post-processing (reusable API)."""
    res = solve_q4(mesh, case, load_vector=load_vector)
    post = postprocess_case(mesh, case, res["U"], res["reactions"],
                            data=res["data"])
    out = dict(res)
    out.update(post)
    if probe_points is not None:
        rows = []
        for x, y in probe_points:
            pt = evaluate_point(mesh, case, res["U"], x, y)
            rows.append(dict(case=case.key, nx=mesh.nx, ny=mesh.ny,
                             x=pt["x"], y=pt["y"], ux=pt["ux"], uy=pt["uy"],
                             sxx=pt["sxx"], syy=pt["syy"], sxy=pt["sxy"],
                             elem=pt["elem"], xi=pt["xi"], eta=pt["eta"]))
        out["probe_rows"] = rows
    return out


__all__ = [
    "NEN", "NDOF_E", "GAUSS_POINTS_2X2", "GAUSS_WEIGHTS_2X2",
    "shape_q4", "jacobian", "b_matrix", "q4_element", "q4_batch",
    "element_nodes", "element_displacements", "assemble_q4",
    "consistent_end_load_vector", "solve_q4", "gauss_stresses", "inverse_map",
    "find_point_element", "evaluate_point", "postprocess_case", "analyze_case",
]
