"""Independent CST check."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem.config import CASES, MAT
from fem.mesh import make_mesh
from fem.assembly import (assemble_sparse, dirichlet_solve, fixed_dofs_left,
                          end_load_vector, axial_force_reaction)
from fem import analytic as an


def cst_ke(xy, mat):
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]
    A = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    B = np.array([
        [y2 - y3, 0, y3 - y1, 0, y1 - y2, 0],
        [0, x3 - x2, 0, x1 - x3, 0, x2 - x1],
        [x3 - x2, y2 - y3, x1 - x3, y3 - y1, x2 - x1, y1 - y2],
    ]) / (2 * A)
    ke = mat.D_plane_stress() @ B * 1.0
    ke = A * B.T @ mat.D_plane_stress() @ B
    return ke, B, A


def solve(mesh, tris, case):
    g = case.geom
    t = g.width
    ke_list, Bs, As = [], [], []
    for conn in tris:
        xy = mesh.nodes[conn]
        ke, B, A = cst_ke(xy, case.material)
        ke_list.append(t * ke)
        Bs.append(B)
        As.append(A)
    K = assemble_sparse(mesh.ndof, tris, ke_list)
    F = end_load_vector(mesh, case.load)
    U, R = dirichlet_solve(K, F, fixed_dofs_left(mesh))
    ux = U[0::2]
    u_end = float(ux[mesh.right_nodes].mean())
    u_mid = float(ux[mesh.column_nodes(mesh.mid_index)].mean())
    reac = axial_force_reaction(U, F, R, mesh)
    sig = np.array([case.material.D_plane_stress() @ B @ U[np.stack([2*c, 2*c+1], axis=1).ravel()]
                    for B, c in zip(Bs, tris)])
    # area-weighted nodal recovery
    nn = mesh.nnodes
    num = np.zeros((nn, 3))
    den = np.zeros(nn)
    for e, conn in enumerate(tris):
        A = As[e]
        for n in conn:
            num[n] += A * sig[e]
            den[n] += A
    nodal = num / den[:, None]
    # raw element column average and recovered node column average
    nx, ny = mesh.nx, mesh.ny
    raw = np.zeros(nx)
    rec = np.zeros(nx)
    # triangles: 2 per quad, two triangles share a quad; map tri to column by centroid
    centers = mesh.element_centers(tris)
    for i in range(nx):
        sel = np.abs(centers[:, 0] - mesh.column_x(i)) < 1e-9
        raw[i] = sig[sel, 0].mean()
        rec[i] = nodal[mesh.column_nodes(i), 0].mean()
    x = np.array([mesh.column_x(i) for i in range(nx)])
    return dict(u_end=u_end, u_mid=u_mid, reaction=reac, x=x,
                raw=raw, rec=rec, nodal=nodal, sig=sig)


if __name__ == "__main__":
    print(f"{'case':>4} {'mesh':>8} {'diag':>4} {'u_end':>12} {'e_end%':>8} "
          f"{'u_mid':>12} {'e_mid%':>8} {'raw_rms%':>9} {'rec_rms%':>9}")
    for case in CASES:
        g = case.geom
        ee = an.exact_u_end(g, MAT, case.load)
        em = an.exact_u_mid(g, MAT, case.load)
        for nx, ny in [(40, 8), (160, 32)]:
            m = make_mesh(g, nx, ny)
            for name, tris in (("pos", m.tris_pos), ("neg", m.tris_neg)):
                r = solve(m, tris, case)
                sn = an.exact_sigma(g, MAT, case.load, r["x"][1:-1])
                raw_rms = 100 * np.sqrt(np.mean((r["raw"][1:-1] - sn) ** 2)) / np.sqrt(np.mean(sn ** 2))
                rec_rms = 100 * np.sqrt(np.mean((r["rec"][1:-1] - sn) ** 2)) / np.sqrt(np.mean(sn ** 2))
                print(f"{case.key:>4} {nx:>3}x{ny:<3} {name:>4} {r['u_end']:.6e} "
                      f"{100*(r['u_end']-ee)/ee:+8.3f} {r['u_mid']:.6e} "
                      f"{100*(r['u_mid']-em)/em:+8.3f} {raw_rms:9.3f} {rec_rms:9.3f}")
