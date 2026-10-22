"""Structured wedge meshes."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import Geometry


@dataclass
class Mesh:
    geom: Geometry
    nx: int
    ny: int
    nodes: np.ndarray          # (nn, 2)
    quads: np.ndarray          # (ne, 4) int, CCW
    tris_pos: np.ndarray       # (2*ne, 3) int
    tris_neg: np.ndarray       # (2*ne, 3) int

    @property
    def nnodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def nquads(self) -> int:
        return self.quads.shape[0]

    @property
    def ndof(self) -> int:
        return 2 * self.nnodes

    def node_index(self, i: int, j: int) -> int:
        return i * (self.ny + 1) + j

    def column_nodes(self, i: int) -> np.ndarray:
        return np.arange(i * (self.ny + 1), (i + 1) * (self.ny + 1))

    def column_x(self, i: int) -> float:
        return self.nodes[self.node_index(i, 0), 0]

    @property
    def left_nodes(self) -> np.ndarray:
        return self.column_nodes(0)

    @property
    def right_nodes(self) -> np.ndarray:
        return self.column_nodes(self.nx)

    @property
    def mid_index(self) -> int:
        if self.nx % 2:
            raise ValueError("mid column requires even nx")
        return self.nx // 2

    def element_center(self, conn: np.ndarray) -> np.ndarray:
        return self.nodes[conn].mean(axis=0)

    def element_centers(self, conns: np.ndarray) -> np.ndarray:
        return self.nodes[conns].mean(axis=1)

    def quad_areas(self) -> np.ndarray:
        xy = self.nodes[self.quads]                       # (ne,4,2)
        x, y = xy[:, :, 0], xy[:, :, 1]
        return 0.5 * np.abs(
            np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
        )

    def tri_areas(self, tris: np.ndarray) -> np.ndarray:
        xy = self.nodes[tris]
        x, y = xy[:, :, 0], xy[:, :, 1]
        return 0.5 * np.abs(
            (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
            - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
        )

    def quad_signed_areas(self) -> np.ndarray:
        xy = self.nodes[self.quads]
        x, y = xy[:, :, 0], xy[:, :, 1]
        return 0.5 * (
            np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
        )

    def summary(self) -> dict:
        return dict(nx=self.nx, ny=self.ny, nnodes=self.nnodes,
                    nquads=self.nquads, ntris=2 * self.nquads,
                    ndof=self.ndof, n_dof_1d=self.nx + 1)

    def section_height(self) -> float:
        return float(self.geom.h(self.column_x(0)))


def make_mesh(geom: Geometry, nx: int, ny: int) -> Mesh:
    if nx < 1 or ny < 1:
        raise ValueError("nx, ny must be positive")
    xs = np.linspace(0.0, geom.L, nx + 1)
    eta = np.linspace(-0.5, 0.5, ny + 1)
    nodes = np.empty(((nx + 1) * (ny + 1), 2))
    for i, x in enumerate(xs):
        h = geom.h(x)
        j0 = i * (ny + 1)
        nodes[j0:j0 + ny + 1, 0] = x
        nodes[j0:j0 + ny + 1, 1] = eta * h

    quads = np.empty((nx * ny, 4), dtype=int)
    e = 0
    for i in range(nx):
        for j in range(ny):
            n1 = i * (ny + 1) + j
            n2 = (i + 1) * (ny + 1) + j
            n3 = (i + 1) * (ny + 1) + j + 1
            n4 = i * (ny + 1) + j + 1
            quads[e] = (n1, n2, n3, n4)
            e += 1

    tris_pos = np.empty((2 * nx * ny, 3), dtype=int)
    tris_neg = np.empty((2 * nx * ny, 3), dtype=int)
    e = 0
    for q in quads:
        n1, n2, n3, n4 = q
        tris_pos[e] = (n1, n2, n3)
        tris_pos[e + 1] = (n1, n3, n4)
        tris_neg[e] = (n1, n2, n4)
        tris_neg[e + 1] = (n2, n3, n4)
        e += 2
    return Mesh(geom=geom, nx=nx, ny=ny, nodes=nodes, quads=quads,
                tris_pos=tris_pos, tris_neg=tris_neg)


def mesh_quality(mesh: Mesh) -> dict:
    qa = mesh.quad_signed_areas()
    ta_p = mesh.tri_areas(mesh.tris_pos)
    ta_n = mesh.tri_areas(mesh.tris_neg)
    return dict(
        min_quad_signed_area=float(qa.min()),
        max_quad_signed_area=float(qa.max()),
        min_tri_area_pos=float(ta_p.min()),
        min_tri_area_neg=float(ta_n.min()),
        area_sum=float(sum(qa)),
    )


__all__ = ["Mesh", "make_mesh", "mesh_quality"]
