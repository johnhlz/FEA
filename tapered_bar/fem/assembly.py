"""Sparse assembly and Dirichlet solve."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from .mesh import Mesh


def dofs_2d(conn: np.ndarray) -> np.ndarray:
    """Interleaved 2D dof vector [ux0, uy0, ux1, uy1, ...] for a connectivity row."""
    conn = np.asarray(conn, dtype=int)
    return np.stack([2 * conn, 2 * conn + 1], axis=1).ravel()


def assemble_sparse(ndof: int, conns: np.ndarray,
                    ke_list, dtype=float) -> csr_matrix:
    """Assemble element stiffness matrices into a CSR matrix."""
    rows, cols, vals = [], [], []
    for conn, ke in zip(conns, ke_list):
        d = dofs_2d(conn)
        n = d.size
        rows.append(np.repeat(d, n))
        cols.append(np.tile(d, n))
        vals.append(np.asarray(ke, dtype=dtype).ravel())
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    K.sum_duplicates()
    return K


def dirichlet_solve(K: csr_matrix, F: np.ndarray, fixed_dofs: np.ndarray):
    """Eliminate fixed dofs, solve K U = F, return (U, reactions)."""
    ndof = K.shape[0]
    fixed = np.asarray(fixed_dofs, dtype=int)
    free = np.setdiff1d(np.arange(ndof), fixed)
    U = np.zeros(ndof, dtype=float)
    F = np.asarray(F, dtype=float)
    if free.size:
        Kff = K[free][:, free].tocsc()
        U[free] = spsolve(Kff, F[free])
    reactions = np.asarray(K @ U - F).ravel()
    return U, reactions


def fixed_dofs_left(mesh: Mesh) -> np.ndarray:
    """Both displacements of every node on the fixed end x=0."""
    return dofs_2d(mesh.left_nodes)


def end_load_vector(mesh: Mesh, P: float, mode: str = "consistent") -> np.ndarray:
    """Nodal load vector on the loaded end x=L for total axial force P."""
    F = np.zeros(mesh.ndof)
    right = mesh.right_nodes
    if mode == "lumped":
        F[2 * right] = P / right.size
        return F
    if mode != "consistent":
        raise ValueError("mode must be 'consistent' or 'lumped'")
    ys = mesh.nodes[right, 1]
    hL = float(mesh.geom.h(mesh.geom.L))
    seg = np.diff(ys)
    loads = P * seg / hL          # uniform traction resultant per edge segment
    F[2 * right[:-1]] += 0.5 * loads
    F[2 * right[1:]] += 0.5 * loads
    return F


def axial_force_reaction(U: np.ndarray, F: np.ndarray, reactions: np.ndarray,
                         mesh: Mesh) -> float:
    """Total axial reaction at the fixed end (should be -P)."""
    left = mesh.left_nodes
    return float(reactions[2 * left].sum())


__all__ = ["dofs_2d", "assemble_sparse", "dirichlet_solve",
           "fixed_dofs_left", "end_load_vector", "axial_force_reaction"]
