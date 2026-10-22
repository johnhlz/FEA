"""1-D nominal solution and exact bar stiffness."""
from __future__ import annotations

import numpy as np
from .config import Geometry, Material


def _arr(x):
    return np.asarray(x, dtype=float)


def exact_u(geom: Geometry, mat: Material, P: float, x):
    """Exact 1D axial displacement of the tapered bar (u(0)=0)."""
    x = _arr(x)
    A0 = geom.A0
    if abs(geom.beta) < 1e-14:
        return P * x / (mat.E * A0)
    return P * geom.L / (mat.E * A0 * geom.beta) * np.log1p(geom.beta * x / geom.L)


def exact_sigma(geom: Geometry, mat: Material, P: float, x):
    """Nominal axial stress sigma = P / A(x) (mat is unused, kept for API symmetry)."""
    return P / geom.A(x)


def exact_eps(geom: Geometry, mat: Material, P: float, x):
    return exact_sigma(geom, mat, P, x) / mat.E


def exact_sigma_gradient(geom: Geometry, mat: Material, P: float, x):
    """d sigma / dx = -P h'(x) / (b h(x)^2)."""
    h = geom.h(x)
    return -P * geom.dhdx / (geom.width * h * h)


def exact_u_end(geom: Geometry, mat: Material, P: float) -> float:
    return float(exact_u(geom, mat, P, geom.L))


def exact_u_mid(geom: Geometry, mat: Material, P: float) -> float:
    return float(exact_u(geom, mat, P, 0.5 * geom.L))


def exact_energy(geom: Geometry, mat: Material, P: float) -> float:
    """1D strain energy 0.5 * P * u(L)."""
    return 0.5 * P * exact_u_end(geom, mat, P)


def exact_compliance(geom: Geometry, mat: Material, x1: float, x2: float) -> float:
    """Integral of dx/(E A(x)) over [x1, x2] (exact)."""
    L, beta, A0 = geom.L, geom.beta, geom.A0
    x1, x2 = float(x1), float(x2)
    if abs(beta) < 1e-14:
        return (x2 - x1) / (mat.E * A0)
    return L / (mat.E * A0 * beta) * np.log(
        (1.0 + beta * x2 / L) / (1.0 + beta * x1 / L)
    )


def bar_equiv_stiffness(geom: Geometry, mat: Material, x1: float, x2: float) -> float:
    """Exact 2-node axial stiffness of a linearly tapered bar segment."""
    return 1.0 / exact_compliance(geom, mat, x1, x2)


def bar_mean_stiffness(geom: Geometry, mat: Material, x1: float, x2: float) -> float:
    """Naive mean-area stiffness E A(x_mid)/(x2-x1) for comparison."""
    xm = 0.5 * (x1 + x2)
    return mat.E * geom.A(xm) / (x2 - x1)


def bar_gauss_stiffness(geom: Geometry, mat: Material, x1: float, x2: float,
                        n_gauss: int = 2) -> float:
    """Gauss-integrated E/L^2 * int A dx over the segment (mean area)."""
    xi, w = np.polynomial.legendre.leggauss(n_gauss)
    xm, half = 0.5 * (x1 + x2), 0.5 * (x2 - x1)
    xs = xm + half * xi
    # int_{x1}^{x2} A(x) dx = half * sum_q w_q A(x_q)
    return float(mat.E / (x2 - x1) ** 2 * (half * np.sum(w * geom.A(xs))))


def nominal_profile(geom: Geometry, mat: Material, P: float, x):
    x = _arr(x)
    return dict(
        x=x,
        h=geom.h(x),
        A=geom.A(x),
        u=exact_u(geom, mat, P, x),
        sigma=exact_sigma(geom, mat, P, x),
        eps=exact_eps(geom, mat, P, x),
        dsigma_dx=exact_sigma_gradient(geom, mat, P, x),
    )


__all__ = [
    "exact_u", "exact_sigma", "exact_eps", "exact_sigma_gradient",
    "exact_u_end", "exact_u_mid", "exact_energy", "exact_compliance",
    "bar_equiv_stiffness", "bar_mean_stiffness", "bar_gauss_stiffness",
    "nominal_profile",
]
