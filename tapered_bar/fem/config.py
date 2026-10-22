"""Geometry, material, cases and mesh settings."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

E_DEFAULT = 2.0e11      # Pa
NU_DEFAULT = 0.3
P_DEFAULT = 1000.0      # N
WIDTH_DEFAULT = 0.1     # m, out-of-plane thickness b
L_DEFAULT = 1.0         # m
H0_DEFAULT = 0.2        # m, height at the fixed end


@dataclass(frozen=True)
class Geometry:
    """Tapered rectangular bar geometry (symmetric about y=0)."""
    L: float = L_DEFAULT
    h0: float = H0_DEFAULT
    width: float = WIDTH_DEFAULT
    beta: float = 1.0

    @property
    def h1(self) -> float:
        return self.h0 * (1.0 + self.beta)

    @property
    def A0(self) -> float:
        return self.width * self.h0

    @property
    def A1(self) -> float:
        return self.width * self.h1

    @property
    def dhdx(self) -> float:
        return self.beta * self.h0 / self.L

    @property
    def taper_angle_deg(self) -> float:
        return float(np.degrees(np.arctan(abs(self.dhdx))))

    @property
    def eta(self) -> float:
        """Dimensionless 1D applicability index: tan(taper angle) = |beta| h0/L."""
        return abs(self.beta) * self.h0 / self.L

    @property
    def slenderness(self) -> float:
        return self.h0 / self.L

    def h(self, x):
        x = np.asarray(x, dtype=float)
        return self.h0 * (1.0 + self.beta * x / self.L)

    def A(self, x):
        return self.width * self.h(x)

    def h_min(self) -> float:
        return min(self.h0, self.h1)

    def h_max(self) -> float:
        return max(self.h0, self.h1)

    def outline(self, n: int = 101):
        xs = np.linspace(0.0, self.L, n)
        return xs, self.h(xs) / 2.0


@dataclass(frozen=True)
class Material:
    E: float = E_DEFAULT
    nu: float = NU_DEFAULT

    def D_plane_stress(self) -> np.ndarray:
        E, nu = self.E, self.nu
        return E / (1.0 - nu * nu) * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
        )


@dataclass(frozen=True)
class Case:
    key: str
    label: str
    geom: Geometry
    material: Material = Material()
    load: float = P_DEFAULT
    note: str = ""


MAT = Material()
LOAD = P_DEFAULT

CASES = (
    Case("S0", "等截面退化（beta=0）", Geometry(beta=0.0, h0=0.2), MAT, LOAD,
         "退化校核：一维杆理论严格适用"),
    Case("S1", "主构型楔形（beta=+1.0）", Geometry(beta=1.0, h0=0.2), MAT, LOAD,
         "主构型：h 由 0.2 m 线性增至 0.4 m"),
    Case("S2", "收缩楔形（beta=-0.5）", Geometry(beta=-0.5, h0=0.2), MAT, LOAD,
         "反向楔形：h 由 0.2 m 线性减至 0.1 m"),
    Case("S3", "粗短强楔形（beta=+1.0, h0=0.4）", Geometry(beta=1.0, h0=0.4), MAT, LOAD,
         "1D 判据边界：eta=0.40，22 deg 锥角"),
    Case("S4", "极端楔形（beta=+2.0, h0=0.3）", Geometry(beta=2.0, h0=0.3), MAT, LOAD,
         "1D 失效示例：eta=0.60，31 deg 锥角"),
)

CASES_BY_KEY = {c.key: c for c in CASES}

# Mesh levels (nx, ny): nx must be a multiple of 10 so every mesh has node
# columns at x = 0.1, 0.2, ..., 1.0.
MESH_LEVELS = ((10, 2), (20, 4), (40, 8), (80, 16), (160, 32))
# Finest mesh used by the in-house convergence check (CCX reference may differ)
REFINE_FACTOR = 2

# Common reporting stations along the axis (interior stations first)
STATIONS = np.round(np.linspace(0.0, 1.0, 11), 10)
INTERIOR_STATIONS = STATIONS[1:-1]

# Diagonal orientations for the CST sensitivity study
CST_DIAGONALS = ("pos", "neg")

__all__ = [
    "E_DEFAULT", "NU_DEFAULT", "P_DEFAULT", "WIDTH_DEFAULT", "L_DEFAULT",
    "H0_DEFAULT", "Geometry", "Material", "Case", "MAT", "LOAD", "CASES",
    "CASES_BY_KEY", "MESH_LEVELS", "STATIONS", "INTERIOR_STATIONS",
    "CST_DIAGONALS",
]
