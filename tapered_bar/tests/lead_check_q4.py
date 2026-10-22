"""Independent Q4 check."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem.config import CASES, MAT, MESH_LEVELS
from fem.mesh import make_mesh
from fem.assembly import (assemble_sparse, dirichlet_solve, fixed_dofs_left,
                          end_load_vector, axial_force_reaction)
from fem import analytic as an

GP = 1.0 / np.sqrt(3.0)
GAUSS = [(-GP, -GP), (GP, -GP), (GP, GP), (-GP, GP)]


def q4_ke_B(xy, mat):
    D = mat.D_plane_stress()
    ke = np.zeros((8, 8))
    gauss_data = []
    for xi, eta in GAUSS:
        dNxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4.0
        dNeta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4.0
        J = np.array([[dNxi @ xy[:, 0], dNxi @ xy[:, 1]],
                      [dNeta @ xy[:, 0], dNeta @ xy[:, 1]]])
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        dNdx = invJ[0, 0] * dNxi + invJ[0, 1] * dNeta
        dNdy = invJ[1, 0] * dNxi + invJ[1, 1] * dNeta
        B = np.zeros((3, 8))
        B[0, 0::2] = dNdx
        B[1, 1::2] = dNdy
        B[2, 0::2] = dNdy
        B[2, 1::2] = dNdx
        ke += B.T @ D @ B * detJ * mat and 0  # placeholder replaced below
        gauss_data.append((B, detJ, (xi, eta)))
    return ke, gauss_data


def q4_ke(xy, mat):
    D = mat.D_plane_stress()
    ke = np.zeros((8, 8))
    for xi, eta in GAUSS:
        dNxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4.0
        dNeta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4.0
        J = np.array([[dNxi @ xy[:, 0], dNxi @ xy[:, 1]],
                      [dNeta @ xy[:, 0], dNeta @ xy[:, 1]]])
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        dNdx = invJ[0, 0] * dNxi + invJ[0, 1] * dNeta
        dNdy = invJ[1, 0] * dNxi + invJ[1, 1] * dNeta
        B = np.zeros((3, 8))
        B[0, 0::2] = dNdx
        B[1, 1::2] = dNdy
        B[2, 0::2] = dNdy
        B[2, 1::2] = dNdx
        ke += B.T @ D @ B * detJ
    return ke


def solve(mesh, case, return_stress=True):
    g = case.geom
    thickness = g.width
    ke_list = []
    for conn in mesh.quads:
        xy = mesh.nodes[conn]
        ke = q4_ke(xy, case.material) * thickness
        ke_list.append(ke)
    K = assemble_sparse(mesh.ndof, mesh.quads, ke_list)
    F = end_load_vector(mesh, case.load)
    U, R = dirichlet_solve(K, F, fixed_dofs_left(mesh))
    ux = U[0::2]
    uy = U[1::2]
    u_end = float(ux[mesh.right_nodes].mean())
    u_mid = float(ux[mesh.column_nodes(mesh.mid_index)].mean())
    reac = axial_force_reaction(U, F, R, mesh)
    # stress at gauss points, per element, and section averages
    D = case.material.D_plane_stress()
    sigma_cols = [[] for _ in range(mesh.nx)]
    shear_energy = 0.0
    axial_energy = 0.0
    for ei, (conn, ke) in enumerate(zip(mesh.quads, ke_list)):
        xy = mesh.nodes[conn]
        dof = np.stack([2 * conn, 2 * conn + 1], axis=1).ravel()
        ue = U[dof]
        i = ei // mesh.ny
        hx = g.h(mesh.column_x(i))
        for xi, eta in GAUSS:
            dNxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4.0
            dNeta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4.0
            J = np.array([[dNxi @ xy[:, 0], dNxi @ xy[:, 1]],
                          [dNeta @ xy[:, 0], dNeta @ xy[:, 1]]])
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)
            dNdx = invJ[0, 0] * dNxi + invJ[0, 1] * dNeta
            dNdy = invJ[1, 0] * dNxi + invJ[1, 1] * dNeta
            B = np.zeros((3, 8))
            B[0, 0::2] = dNdx
            B[1, 1::2] = dNdy
            B[2, 0::2] = dNdy
            B[2, 1::2] = dNdx
            s = D @ B @ ue
            sigma_cols[i].append(s)
            dV = detJ * thickness
            axial_energy += s[0] ** 2 * dV
            shear_energy += (s[1] ** 2 + s[2] ** 2) * dV
    if not return_stress:
        return dict(u_end=u_end, u_mid=u_mid, reaction=reac)
    sig_cols = [np.array(col) for col in sigma_cols]
    sigma_x_avg = np.array([col[:, 0].mean() for col in sig_cols])
    x_cols = np.array([mesh.column_x(i) for i in range(mesh.nx)])
    return dict(u_end=u_end, u_mid=u_mid, reaction=reac,
                x_cols=x_cols, sigma_x_avg=sigma_x_avg,
                shear_energy_ratio=shear_energy / axial_energy,
                sigma_all=np.concatenate(sig_cols, axis=0))


if __name__ == "__main__":
    print(f"{'case':>4} {'mesh':>8} {'u_end':>12} {'e_end%':>8} "
          f"{'u_mid':>12} {'e_mid%':>8} {'sig_rms%':>9} {'shear%':>8} {'R/P':>8}")
    for case in CASES:
        g = case.geom
        ee = an.exact_u_end(g, MAT, case.load)
        em = an.exact_u_mid(g, MAT, case.load)
        for nx, ny in [(40, 8), (160, 32)]:
            m = make_mesh(g, nx, ny)
            r = solve(m, case)
            x = r["x_cols"]
            sn = an.exact_sigma(g, MAT, case.load, x[1:-1])
            sf = r["sigma_x_avg"][1:-1]
            sig_rms = 100 * np.sqrt(np.mean((sf - sn) ** 2)) / np.sqrt(np.mean(sn ** 2))
            print(f"{case.key:>4} {nx:>3}x{ny:<3} {r['u_end']:.6e} "
                  f"{100*(r['u_end']-ee)/ee:+8.3f} {r['u_mid']:.6e} "
                  f"{100*(r['u_mid']-em)/em:+8.3f} {sig_rms:9.3f} "
                  f"{100*r['shear_energy_ratio']:8.4f} {r['reaction']/case.load:+8.4f}")
