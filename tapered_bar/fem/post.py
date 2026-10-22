"""Section-average post-processing."""
from __future__ import annotations

import numpy as np

from .mesh import Mesh


def column_trapezoid_weights(mesh: Mesh, nodes=None):
    """Return (sorted_nodes, weights) for a node column of the structured mesh."""
    if nodes is None:
        raise ValueError("nodes required")
    nodes = np.asarray(nodes, dtype=int)
    ys = mesh.nodes[nodes, 1]
    order = np.argsort(ys)
    nodes = nodes[order]
    w = np.ones(nodes.size, dtype=float)
    if nodes.size > 1:
        w[0] = w[-1] = 0.5
    return nodes, w


def section_average_u(mesh: Mesh, U: np.ndarray, column: int) -> float:
    """Area-weighted average of ux over the node column i=column."""
    nodes = mesh.column_nodes(int(column))
    nodes, w = column_trapezoid_weights(mesh, nodes)
    ux = np.asarray(U, dtype=float)[2 * nodes]
    return float(np.sum(w * ux) / np.sum(w))


def end_average_u(mesh: Mesh, U: np.ndarray) -> float:
    return section_average_u(mesh, U, mesh.nx)


def mid_average_u(mesh: Mesh, U: np.ndarray) -> float:
    return section_average_u(mesh, U, mesh.mid_index)


__all__ = ["column_trapezoid_weights", "section_average_u",
           "end_average_u", "mid_average_u"]
