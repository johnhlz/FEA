"""3-node constant-strain triangle."""
from __future__ import annotations

import numpy as np

from .config import STATIONS
from .mesh import Mesh, make_mesh
from .assembly import (assemble_sparse, dirichlet_solve, fixed_dofs_left,
                       end_load_vector, axial_force_reaction)


# element level
def tri_conn(mesh: Mesh, diag: str) -> np.ndarray:
    """Connectivity (ne, 3) of the requested diagonal triangulation."""
    if diag == "pos":
        return mesh.tris_pos
    if diag == "neg":
        return mesh.tris_neg
    raise ValueError("diag must be 'pos' or 'neg', got %r" % (diag,))


def element_geometry(mesh: Mesh, tris: np.ndarray):
    """Return (xy, det, area) with det = 2*signed area and area = 0.5*det."""
    xy = mesh.nodes[tris]                      # (ne, 3, 2)
    x, y = xy[..., 0], xy[..., 1]
    det = ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
           - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    return xy, det, 0.5 * det


def cst_B_batch(xy: np.ndarray, det: np.ndarray) -> np.ndarray:
    """Constant strain-displacement matrices, shape (ne, 3, 6)."""
    x, y = xy[..., 0], xy[..., 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]],
                 axis=1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]],
                 axis=1)
    inv = 1.0 / det
    B = np.zeros((xy.shape[0], 3, 6))
    B[:, 0, 0::2] = b * inv[:, None]
    B[:, 1, 1::2] = c * inv[:, None]
    B[:, 2, 0::2] = c * inv[:, None]
    B[:, 2, 1::2] = b * inv[:, None]
    return B


def cst_ke_batch(B: np.ndarray, area: np.ndarray, D: np.ndarray,
                 thickness: float) -> np.ndarray:
    """Explicit element stiffness t*|A|*B^T D B, shape (ne, 6, 6)."""
    A = np.abs(area)
    return thickness * A[:, None, None] * np.einsum("nki,kl,nlj->nij", B, D, B)


def cst_stress_batch(B: np.ndarray, D: np.ndarray, U: np.ndarray,
                     tris: np.ndarray):
    """Return (eps_e, sigma_e) for all triangles; both shape (ne, 3)."""
    ne = tris.shape[0]
    dof = np.empty((ne, 6), dtype=int)
    dof[:, 0::2] = 2 * tris
    dof[:, 1::2] = 2 * tris + 1
    ue = np.asarray(U)[dof]
    eps = np.einsum("nij,nj->ni", B, ue)
    sigma = np.einsum("ij,nj->ni", D, eps)
    return eps, sigma


def solve_cst(geom, mat, P: float, nx: int, ny: int, diag: str = "pos",
              mesh: Mesh | None = None, tris: np.ndarray | None = None) -> dict:
    """Assemble and solve one CST case on the structured tapered mesh."""
    if mesh is None:
        mesh = make_mesh(geom, nx, ny)
    if tris is None:
        tris = tri_conn(mesh, diag)
    xy, det, area = element_geometry(mesh, tris)
    if not np.all(det > 0.0):
        raise ValueError("non-positive triangle area for diag=%s" % diag)
    D = mat.D_plane_stress()
    B = cst_B_batch(xy, det)
    ke = cst_ke_batch(B, area, D, geom.width)
    K = assemble_sparse(mesh.ndof, tris, ke)
    F = end_load_vector(mesh, P)
    U, reactions = dirichlet_solve(K, F, fixed_dofs_left(mesh))
    eps_e, sigma_e = cst_stress_batch(B, D, U, tris)

    rec_w, conn_area = recover_nodal(mesh, tris, sigma_e, area, weighted=True)
    rec_a, _ = recover_nodal(mesh, tris, sigma_e, area, weighted=False)

    ux = U[0::2]
    uy = U[1::2]
    right = mesh.right_nodes
    mid_col = mesh.column_nodes(mesh.mid_index)
    return dict(
        geom=geom, mat=mat, P=P, nx=nx, ny=ny, diag=diag,
        mesh=mesh, tris=tris, U=U, ux=ux, uy=uy,
        reactions=reactions, F=F, K=K,
        xy=xy, det=det, area=area, B=B, D=D, eps_e=eps_e, sigma_e=sigma_e,
        node_area=conn_area,
        recovered=rec_w, recovered_arith=rec_a,
        u_end=float(ux[right].mean()),
        u_mid=float(ux[mid_col].mean()),
        reaction=axial_force_reaction(U, F, reactions, mesh),
    )


def solve_case(case, nx: int, ny: int, diag: str = "pos",
               mesh: Mesh | None = None) -> dict:
    """Reusable entry point: solve one Case on an (nx, ny) mesh."""
    return solve_cst(case.geom, case.material, case.load, nx, ny, diag, mesh)


def solve_mesh(mesh: Mesh, tris: np.ndarray, case) -> dict:
    """Reusable entry point for a user-supplied mesh/connectivity."""
    if np.array_equal(tris, mesh.tris_pos):
        diag = "pos"
    elif np.array_equal(tris, mesh.tris_neg):
        diag = "neg"
    else:
        diag = "custom"
    return solve_cst(case.geom, case.material, case.load, mesh.nx, mesh.ny,
                     diag, mesh=mesh, tris=tris)


# stress recovery
def recover_nodal(mesh: Mesh, tris: np.ndarray, sigma_e: np.ndarray,
                  areas: np.ndarray, weighted: bool = True):
    """Nodal stress by averaging the incident element stresses."""
    w = np.abs(np.asarray(areas, dtype=float)) if weighted \
        else np.ones(tris.shape[0], dtype=float)
    nn = mesh.nnodes
    den = np.zeros(nn)
    np.add.at(den, tris.ravel(), np.repeat(w, 3))
    acc = np.zeros((nn, 3))
    for k in range(3):
        np.add.at(acc, tris[:, k], w[:, None] * sigma_e)
    good = den > 0.0
    out = np.zeros((nn, 3))
    out[good] = acc[good] / den[good, None]
    return out, den


# section profiles
def tri_element_columns(mesh: Mesh, tris: np.ndarray) -> np.ndarray:
    """Element column index (0..nx-1) of every triangle."""
    return (tris // (mesh.ny + 1)).min(axis=1)


def section_sets(mesh: Mesh, tris: np.ndarray, stations=STATIONS):
    """Per-station (x_frac, column_index, tri_mask, node_ids)."""
    tri_col = tri_element_columns(mesh, tris)
    out = []
    for xf in stations:
        i = int(round(float(xf) * mesh.nx))
        i = min(max(i, 0), mesh.nx)
        if i == 0:
            cols = [0]
        elif i == mesh.nx:
            cols = [mesh.nx - 1]
        else:
            cols = [i - 1, i]
        mask = np.isin(tri_col, cols)
        out.append((float(xf), i, mask, mesh.column_nodes(i)))
    return out


def section_profile(res: dict, stations=STATIONS) -> dict:
    """Section averages and dispersion at the reporting stations."""
    mesh, tris = res["mesh"], res["tris"]
    sigma_x = res["sigma_e"][:, 0]
    rec_x = res["recovered"][:, 0]
    rec_a_x = res["recovered_arith"][:, 0]
    ux = res["ux"]
    cols = dict(
        x=[], column_index=[], x_m=[], u_section_avg=[],
        sigma_section_avg_raw=[], sigma_section_avg_recovered=[],
        cv_raw=[], cv_recovered=[], cv_recovered_arith=[],
        n_tris=[], n_nodes=[],
    )
    for xf, i, mask, nodes in section_sets(mesh, tris, stations):
        s_e = sigma_x[mask]
        raw = float(s_e.mean())
        rec = rec_x[nodes]
        recm = float(rec.mean())
        reca = rec_a_x[nodes]
        recam = float(reca.mean())
        cols["x"].append(xf)
        cols["column_index"].append(i)
        cols["x_m"].append(mesh.column_x(i))
        cols["u_section_avg"].append(float(ux[nodes].mean()))
        cols["sigma_section_avg_raw"].append(raw)
        cols["sigma_section_avg_recovered"].append(recm)
        cols["cv_raw"].append(float(s_e.std() / abs(raw)) if raw != 0 else np.nan)
        cols["cv_recovered"].append(float(rec.std() / abs(recm)) if recm != 0 else np.nan)
        cols["cv_recovered_arith"].append(
            float(reca.std() / abs(recam)) if recam != 0 else np.nan)
        cols["n_tris"].append(int(mask.sum()))
        cols["n_nodes"].append(int(nodes.size))
    return {k: np.asarray(v) for k, v in cols.items()}


def profile_rms_pct(sigma, x_frac, sigma_nom, lo: float = 0.1,
                    hi: float = 0.9) -> float:
    """100 * RMS(sigma - sigma_nom) / RMS(sigma_nom) for x in [lo, hi] (in L units)."""
    sel = (np.asarray(x_frac) >= lo - 1e-12) & (np.asarray(x_frac) <= hi + 1e-12)
    d = np.asarray(sigma)[sel] - np.asarray(sigma_nom)[sel]
    return float(100.0 * np.sqrt(np.mean(d * d))
                 / np.sqrt(np.mean(np.asarray(sigma_nom)[sel] ** 2)))


# point probes
def probe_points(geom, xs=(0.1, 0.2, 0.5, 0.8, 0.9),
                 fy=(0.0, 0.25, -0.25)):
    """Off-axis probe points: x = xs*L, y = fy*h(x).  Returns (n, 2)."""
    pts = []
    for xf in xs:
        x = float(xf) * geom.L
        h = float(geom.h(x))
        for f in fy:
            pts.append((x, f * h))
    return np.asarray(pts, dtype=float)


def evaluate_at_points(res: dict, points, tol: float = 1e-12) -> dict:
    """Evaluate the CST solution at arbitrary points."""
    mesh, tris = res["mesh"], res["tris"]
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    xy, det, area = element_geometry(mesh, tris)
    x1, y1 = xy[:, 0, 0], xy[:, 0, 1]
    x2, y2 = xy[:, 1, 0], xy[:, 1, 1]
    x3, y3 = xy[:, 2, 0], xy[:, 2, 1]
    cent = xy.mean(axis=1)
    sig = res["sigma_e"]
    ux, uy = res["ux"], res["uy"]
    tris = res["tris"]
    w_all = np.abs(area)
    n = pts.shape[0]
    out = dict(ux=np.empty(n), uy=np.empty(n), sxx=np.empty(n),
               syy=np.empty(n), sxy=np.empty(n),
               tri_index=np.empty(n, dtype=int),
               n_containing=np.empty(n, dtype=int))
    for p, (px, py) in enumerate(pts):
        L1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / det
        L2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / det
        L3 = 1.0 - L1 - L2
        inside = np.where((L1 >= -tol) & (L2 >= -tol) & (L3 >= -tol))[0]
        if inside.size == 0:
            raise ValueError("point %r is outside the mesh" % (p,))
        # representative triangle (closest centroid) for the displacement
        if inside.size == 1:
            j = int(inside[0])
        else:
            d2 = np.sum((cent[inside] - np.array([px, py])) ** 2, axis=1)
            j = int(inside[np.argmin(d2)])
        L = np.array([L1[j], L2[j], L3[j]])
        Lc = np.clip(L, 0.0, 1.0)
        Lc /= Lc.sum()
        conn = tris[j]
        out["ux"][p] = Lc[0] * ux[conn[0]] + Lc[1] * ux[conn[1]] + Lc[2] * ux[conn[2]]
        out["uy"][p] = Lc[0] * uy[conn[0]] + Lc[1] * uy[conn[1]] + Lc[2] * uy[conn[2]]
        w = w_all[inside]
        out["sxx"][p] = float(np.dot(w, sig[inside, 0]) / w.sum())
        out["syy"][p] = float(np.dot(w, sig[inside, 1]) / w.sum())
        out["sxy"][p] = float(np.dot(w, sig[inside, 2]) / w.sum())
        out["tri_index"][p] = j
        out["n_containing"][p] = inside.size
    return out


# diagonal sensitivity metrics
def diagonal_metrics(res_pos: dict, res_neg: dict, sigma_nom,
                     stations=STATIONS, interior=(0.1, 0.9)) -> dict:
    """Compare the two diagonal triangulations on the same mesh."""
    geom = res_pos["geom"]
    mesh = res_pos["mesh"]
    p = section_profile(res_pos, stations)
    n = section_profile(res_neg, stations)
    ue_p, ue_n = res_pos["u_end"], res_neg["u_end"]
    um_p, um_n = res_pos["u_mid"], res_neg["u_mid"]

    def _rel(a, b):
        den = 0.5 * (abs(a) + abs(b))
        return float(100.0 * abs(a - b) / den) if den > 0 else np.nan

    lo, hi = interior
    sel = (p["x"] >= lo - 1e-12) & (p["x"] <= hi + 1e-12)
    nom = np.asarray(sigma_nom, dtype=float)
    diff = p["sigma_section_avg_raw"][sel] - n["sigma_section_avg_raw"][sel]
    d_sigma_axial_pct = float(
        100.0 * np.sqrt(np.mean(diff ** 2)) / np.sqrt(np.mean(nom[sel] ** 2)))

    imid = int(np.argmin(np.abs(p["x"] - 0.5)))
    sm_p = float(p["sigma_section_avg_raw"][imid])
    sm_n = float(n["sigma_section_avg_raw"][imid])

    # element-wise (same element index => same quad, different half)
    sp = res_pos["sigma_e"][:, 0]
    sn = res_neg["sigma_e"][:, 0]
    cent = mesh.nodes[res_pos["tris"]].mean(axis=1)
    nom_e = res_pos["P"] / geom.A(cent[:, 0])
    d_sigma_element_pct = float(
        100.0 * np.sqrt(np.mean((sp - sn) ** 2))
        / np.sqrt(np.mean(nom_e ** 2)))

    # nodal (area-weighted recovered)
    rp = res_pos["recovered"][:, 0]
    rn = res_neg["recovered"][:, 0]
    mean_n = float(rp.mean())
    d_sigma_nodal_pct = float(
        100.0 * np.sqrt(np.mean((rp - rn) ** 2)) / abs(mean_n))
    max_abs_diff = float(np.max(np.abs(rp - rn)))
    max_nodal_recovered_diff_pct = float(100.0 * max_abs_diff / abs(mean_n))

    return dict(
        u_end_pos=ue_p, u_end_neg=ue_n, u_mid_pos=um_p, u_mid_neg=um_n,
        d_u_pct=_rel(ue_p, ue_n), d_u_mid_pct=_rel(um_p, um_n),
        sigma_mid_pos=sm_p, sigma_mid_neg=sm_n,
        d_sigma_mid_pct=_rel(sm_p, sm_n),
        sigma_end_pos=float(p["sigma_section_avg_raw"][-1]),
        sigma_end_neg=float(n["sigma_section_avg_raw"][-1]),
        d_sigma_axial_pct=d_sigma_axial_pct,
        d_sigma_element_pct=d_sigma_element_pct,
        d_sigma_nodal_pct=d_sigma_nodal_pct,
        max_nodal_abs_diff=max_abs_diff,
        mean_nodal=mean_n,
        max_nodal_recovered_diff_pct=max_nodal_recovered_diff_pct,
        cv_max_raw_pos=float(np.nanmax(p["cv_raw"][sel])),
        cv_max_raw_neg=float(np.nanmax(n["cv_raw"][sel])),
        cv_max_recovered_pos=float(np.nanmax(p["cv_recovered"][sel])),
        cv_max_recovered_neg=float(np.nanmax(n["cv_recovered"][sel])),
    )


def probe_rows(res: dict, case, xs=(0.1, 0.2, 0.5, 0.8, 0.9)) -> list:
    """Probe rows for cst_probe.csv at x=xs*L, y=0,+h/4,-h/4."""
    geom = res["geom"]
    xs = tuple(xs)
    pts = probe_points(geom, xs=xs)
    vals = evaluate_at_points(res, pts)
    rows = []
    for p, (x, y) in enumerate(pts):
        rows.append(dict(
            case=case.key, nx=res["nx"], ny=res["ny"], diag=res["diag"],
            x=x, y=y,
            ux=float(vals["ux"][p]), uy=float(vals["uy"][p]),
            sxx=float(vals["sxx"][p]), syy=float(vals["syy"][p]),
            sxy=float(vals["sxy"][p]),
        ))
    return rows


__all__ = [
    "tri_conn", "element_geometry", "cst_B_batch", "cst_ke_batch",
    "cst_stress_batch", "solve_cst", "solve_case", "solve_mesh",
    "recover_nodal", "tri_element_columns", "section_sets",
    "section_profile", "profile_rms_pct", "probe_points",
    "evaluate_at_points", "diagonal_metrics", "probe_rows",
]
