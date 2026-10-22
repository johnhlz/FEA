"""1-D variable-section bar element."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .analytic import (
    bar_equiv_stiffness,
    bar_mean_stiffness,
    exact_sigma,
    exact_u,
    exact_u_end,
    exact_u_mid,
)
from .assembly import dirichlet_solve
from .config import Case, Geometry, Material

METHODS = ("equiv", "mean", "gauss")

# 1-D applicability criterion: eta = |beta| h0 / L = |dh/dx| = tan(angle)
ETA_A_MAX = 0.20   # eta <= 0.20          -> A (1-D trusted)
ETA_B_MAX = 0.40   # 0.20 < eta <= 0.40   -> B (2-D check required)
# eta > ETA_B_MAX                         -> C (1-D not trustworthy)

LABEL_DESC = {"A": "A 可信", "B": "B 需 2D 复核", "C": "C 不可信"}

RECOMMENDATIONS = {
    "A": "1D 可信：中截面偏差 <0.5%，端部约 1%",
    "B": "1D 初估：中截面约 1%，端部约 4%，需 2D 复核",
    "C": "1D 不可信：需 2D/3D 分析",
}


def criterion_text() -> str:
    """Human-readable statement of the A/B/C criterion (uses the constants)."""
    return (
        "η=|β|h0/L；"
        f"A≤{ETA_A_MAX:.2f}，B≤{ETA_B_MAX:.2f}，C>{ETA_B_MAX:.2f}"
    )


CRITERION_TEXT = criterion_text()


def classify_eta(eta: float) -> str:
    """Map the geometric index eta to the A/B/C credibility label."""
    eta = abs(float(eta))
    if eta <= ETA_A_MAX:
        return "A"
    if eta <= ETA_B_MAX:
        return "B"
    return "C"


# Bar discretisation and element stiffness
def bar_nodes(geom: Geometry, n_elem: int) -> np.ndarray:
    """Equally spaced 1-D node coordinates on [0, L]."""
    n = int(n_elem)
    if n < 1:
        raise ValueError("n_elem must be >= 1")
    return np.linspace(0.0, geom.L, n + 1)


def bar_connectivity(n_elem: int) -> np.ndarray:
    """(n_elem, 2) connectivity: element e = (e, e+1)."""
    n = int(n_elem)
    if n < 1:
        raise ValueError("n_elem must be >= 1")
    return np.column_stack((np.arange(n), np.arange(1, n + 1)))


def gauss_stiffness(geom: Geometry, mat: Material, x1: float, x2: float,
                    n_gauss: int = 2) -> float:
    """k_e = E/Le**2 * int_{x1}^{x2} A(x) dx by n-point Gauss quadrature."""
    Le = float(x2) - float(x1)
    if Le <= 0.0:
        raise ValueError("require x2 > x1")
    xi, w = np.polynomial.legendre.leggauss(int(n_gauss))
    xm, half = 0.5 * (float(x1) + float(x2)), 0.5 * Le
    area_int = half * float(np.sum(w * geom.A(xm + half * xi)))
    return float(mat.E * area_int / Le ** 2)


def element_stiffness(geom: Geometry, mat: Material, x1: float, x2: float,
                      method: str = "equiv", n_gauss: int = 2) -> float:
    """Axial stiffness of one 2-node variable-section bar element."""
    if method == "equiv":
        return float(bar_equiv_stiffness(geom, mat, x1, x2))
    if method == "mean":
        return float(bar_mean_stiffness(geom, mat, x1, x2))
    if method == "gauss":
        return gauss_stiffness(geom, mat, x1, x2, n_gauss=n_gauss)
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def element_stiffness_vector(geom: Geometry, mat: Material, nodes: np.ndarray,
                             method: str = "equiv",
                             n_gauss: int = 2) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=float)
    return np.array([
        element_stiffness(geom, mat, nodes[i], nodes[i + 1], method, n_gauss)
        for i in range(nodes.size - 1)
    ])


def element_matrix(k: float) -> np.ndarray:
    """2-node bar element stiffness matrix for a given axial stiffness."""
    k = float(k)
    return np.array([[k, -k], [-k, k]])


def assemble_bar(geom: Geometry, mat: Material, n_elem: int,
                 method: str = "equiv", n_gauss: int = 2):
    """Assemble the sparse 1-D bar stiffness matrix."""
    nodes = bar_nodes(geom, n_elem)
    elems = bar_connectivity(n_elem)
    ke = element_stiffness_vector(geom, mat, nodes, method, n_gauss)
    n = int(n_elem)
    idx = np.arange(n)
    rows = np.stack([idx, idx, idx + 1, idx + 1], axis=1).ravel()
    cols = np.stack([idx, idx + 1, idx, idx + 1], axis=1).ravel()
    vals = np.stack([ke, -ke, -ke, ke], axis=1).ravel()
    K = coo_matrix((vals, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    K.sum_duplicates()
    return K, ke, nodes, elems


# Solved model
@dataclass
class BarModel:
    """Solved 1-D variable-section bar (left end fixed, right end loaded)."""

    geom: Geometry
    mat: Material
    P: float
    n_elem: int
    method: str
    nodes: np.ndarray
    elems: np.ndarray
    ke: np.ndarray
    K: csr_matrix
    F: np.ndarray
    U: np.ndarray
    reactions: np.ndarray

    @property
    def Le(self) -> float:
        return self.geom.L / self.n_elem

    @property
    def u_end(self) -> float:
        return float(self.U[-1])

    def u_at(self, x) -> np.ndarray:
        return np.interp(np.asarray(x, dtype=float), self.nodes, self.U)

    @property
    def u_mid(self) -> float:
        return float(self.u_at(0.5 * self.geom.L))

    def u_exact_at(self, x) -> np.ndarray:
        return exact_u(self.geom, self.mat, self.P, np.asarray(x, dtype=float))

    @property
    def element_lengths(self) -> np.ndarray:
        return np.diff(self.nodes)

    @property
    def element_axial_forces(self) -> np.ndarray:
        """N_e = k_e * elongation (tension positive); should equal P."""
        du = self.U[self.elems[:, 1]] - self.U[self.elems[:, 0]]
        return self.ke * du

    @property
    def element_strains(self) -> np.ndarray:
        du = self.U[self.elems[:, 1]] - self.U[self.elems[:, 0]]
        return du / self.element_lengths

    @property
    def element_stresses(self) -> np.ndarray:
        """Constitutive (B^T recovery) axial stress sigma_e = E eps_e."""
        return self.mat.E * self.element_strains

    @property
    def nodal_stresses(self) -> np.ndarray:
        """Nodal stress: mean of adjacent element stresses (equal length)."""
        se = self.element_stresses
        nod = np.empty(self.nodes.size, dtype=float)
        nod[0] = se[0]
        nod[-1] = se[-1]
        if se.size > 1:
            nod[1:-1] = 0.5 * (se[:-1] + se[1:])
        return nod

    def stress_at(self, x) -> np.ndarray:
        return np.interp(np.asarray(x, dtype=float), self.nodes,
                         self.nodal_stresses)

    def sigma_exact_at(self, x) -> np.ndarray:
        return exact_sigma(self.geom, self.mat, self.P,
                           np.asarray(x, dtype=float))

    @property
    def reaction_x(self) -> float:
        """Total axial reaction at the fixed end (should be -P)."""
        return float(np.sum(self.reactions[:1]))

    @property
    def force_error(self) -> float:
        """max_e |N_e - P| / P (equilibrium consistency of the 1-D chain)."""
        return float(np.max(np.abs(self.element_axial_forces - self.P)) / self.P)

    @property
    def strain_energy(self) -> float:
        return 0.5 * float(self.U @ (self.K @ self.U))

    @property
    def external_work(self) -> float:
        return 0.5 * self.P * self.u_end

    def summary(self) -> dict:
        return dict(
            n_elem=int(self.n_elem), method=self.method,
            u_end=float(self.u_end), u_mid=float(self.u_mid),
            reaction_x=float(self.reaction_x),
            force_error=float(self.force_error),
            strain_energy=float(self.strain_energy),
            external_work=float(self.external_work),
            n_elem_force_min=float(np.min(self.element_axial_forces)),
            n_elem_force_max=float(np.max(self.element_axial_forces)),
            sigma_mid=float(self.stress_at(0.5 * self.geom.L)),
        )


def build_bar(geom: Geometry, mat: Material, P: float, n_elem: int,
              method: str = "equiv", n_gauss: int = 2) -> BarModel:
    """Assemble and solve the 1-D bar (u(0)=0, F(L)=P)."""
    K, ke, nodes, elems = assemble_bar(geom, mat, n_elem, method, n_gauss)
    ndof = int(n_elem) + 1
    F = np.zeros(ndof)
    F[-1] = float(P)
    fixed = np.array([0], dtype=int)
    U, reactions = dirichlet_solve(K, F, fixed)
    return BarModel(geom=geom, mat=mat, P=float(P), n_elem=int(n_elem),
                    method=method, nodes=nodes, elems=elems, ke=ke, K=K,
                    F=F, U=U, reactions=np.asarray(reactions, dtype=float))


def solve_case(case: Case, n_elem: int, method: str = "equiv",
               n_gauss: int = 2) -> BarModel:
    return build_bar(case.geom, case.material, case.load, n_elem, method,
                     n_gauss)


# Result records (CSV rows)
def run_record(case: Case, n_elem: int, method: str = "equiv",
               n_gauss: int = 2) -> dict:
    """One row of results/bar_runs.csv."""
    model = solve_case(case, n_elem, method, n_gauss)
    geom, mat, P = case.geom, case.material, case.load
    ue = exact_u_end(geom, mat, P)
    um = exact_u_mid(geom, mat, P)
    sm = float(exact_sigma(geom, mat, P, 0.5 * geom.L))
    s_fem = float(model.stress_at(0.5 * geom.L))
    return dict(
        case=case.key, n_elem=int(n_elem), method=method,
        u_end=model.u_end, u_mid=model.u_mid,
        u_end_exact=ue, u_mid_exact=um,
        e_end_pct=100.0 * (model.u_end - ue) / ue,
        e_mid_pct=100.0 * (model.u_mid - um) / um,
        sigma_mid=s_fem, sigma_mid_exact=sm,
        e_sigma_pct=100.0 * (s_fem - sm) / sm,
    )


def node_records(case: Case, n_elem: int, method: str = "equiv",
                 n_gauss: int = 2) -> list:
    """Rows of results/bar_nodes.csv (all nodes of one run)."""
    model = solve_case(case, n_elem, method, n_gauss)
    geom, mat, P = case.geom, case.material, case.load
    u_ex = exact_u(geom, mat, P, model.nodes)
    s_ex = exact_sigma(geom, mat, P, model.nodes)
    s_fem = model.nodal_stresses
    rows = []
    for i, x in enumerate(model.nodes):
        rows.append(dict(
            case=case.key, n_elem=int(n_elem), method=method,
            x=float(x), u_fem=float(model.U[i]), u_exact=float(u_ex[i]),
            sigma_fem=float(s_fem[i]), sigma_exact=float(s_ex[i]),
        ))
    return rows


def observed_order(ns, errs, n_min: int = 10, zero_tol: float = 1e-9) -> dict:
    """Observed convergence order p from e ~ C n^-p  (log-log least squares)."""
    ns = np.asarray(ns, dtype=float)
    errs = np.asarray(np.abs(errs), dtype=float)
    finite = np.isfinite(errs)
    mask = finite & (errs > 0.0) & (ns >= n_min)
    err_max = float(np.max(errs[mask])) if mask.any() else float("nan")
    if mask.sum() < 2 or not np.isfinite(err_max) or err_max < zero_tol:
        return dict(order=float("nan"), pairs=[], n_used=[],
                    err_max=err_max, zero_tol=float(zero_tol))
    p = -float(np.polyfit(np.log(ns[mask]), np.log(errs[mask]), 1)[0])
    pairs = []
    for n1, e1, n2, e2 in zip(ns[:-1], errs[:-1], ns[1:], errs[1:]):
        if e1 > 0 and e2 > 0 and n1 >= n_min:
            pairs.append(dict(n1=int(n1), n2=int(n2),
                              order=float(np.log(e1 / e2) / np.log(n2 / n1))))
    return dict(order=p, pairs=pairs,
                n_used=[int(v) for v in ns[mask]],
                err_max=err_max, zero_tol=float(zero_tol))


# 1-D applicability criterion
def applicability(geom: Geometry) -> dict:
    """Geometric applicability indices, A/B/C label and recommendation."""
    eta = float(geom.eta)
    label = classify_eta(eta)
    return dict(
        eta=eta,
        angle_deg=float(geom.taper_angle_deg),
        slenderness=float(geom.slenderness),
        area_ratio=float(geom.A1 / geom.A0),
        label=label,
        label_desc=LABEL_DESC[label],
        criterion=CRITERION_TEXT,
        recommendation=RECOMMENDATIONS[label],
        eta_sq_pct=100.0 * eta * eta,
    )


# Optional posterior check against results/ccx_reference.json
CCX_REFERENCE_PATH = "results/ccx_reference.json"

_SIGMA_KEYS = ("sigma_section_avg", "sigma_avg", "sigma_mid", "sigma_x", "sigma")
_U_KEYS = ("u_section_avg", "u_avg", "u_mid", "ux", "u")


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def load_ccx_reference(path=None) -> dict:
    """Load results/ccx_reference.json if present; missing file -> {}."""
    p = pathlib.Path(path) if path is not None else (_repo_root() / CCX_REFERENCE_PATH)
    if not p.is_absolute():
        p = _repo_root() / p
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(data, dict):
        for key in ("cases", "reference", "references", "data"):
            v = data.get(key)
            if isinstance(v, dict) and any(str(k).upper().startswith("S") for k in v):
                data = v
                break
    return data if isinstance(data, dict) else {}


def _case_entry(ref: dict, key: str):
    for k in (key, key.lower(), key.upper()):
        v = ref.get(k)
        if isinstance(v, dict):
            return v
    return None


def _profile_arrays(profile):
    """Return (x_array, {name: array}) from a profile blob (tolerant)."""
    if profile is None:
        return None, {}
    if isinstance(profile, dict):
        x = None
        for k in ("x", "stations", "xs"):
            if k in profile and profile[k] is not None:
                try:
                    x = np.asarray(profile[k], dtype=float)
                except (TypeError, ValueError):
                    x = None
                break
        arrs = {}
        for k, v in profile.items():
            if k in ("x", "stations", "xs"):
                continue
            try:
                arrs[str(k)] = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                pass
        return x, arrs
    if isinstance(profile, (list, tuple)) and profile:
        if isinstance(profile[0], dict):
            x = np.array([float(e.get("x", np.nan)) for e in profile])
            cols = {}
            names = []
            for e in profile:
                for k, v in e.items():
                    if k == "x":
                        continue
                    if isinstance(v, (int, float)) and k not in names:
                        names.append(k)
                        cols[k] = []
                for k in names:
                    v = e.get(k, np.nan)
                    cols[k].append(float(v) if isinstance(v, (int, float)) else np.nan)
            return x, {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    return None, {}


def _profile_lookup(x, arrays: dict, keys, xtarget: float) -> float:
    for key in keys:
        for name, arr in arrays.items():
            if key.lower() in str(name).lower():
                if arr.size == 0:
                    continue
                if x is not None and arr.size == x.size and x.size > 1:
                    order = np.argsort(x)
                    return float(np.interp(xtarget, x[order], arr[order]))
                if arr.size == 1:
                    return float(arr[0])
    return float("nan")


def ccx_case_data(case: Case, ref: dict) -> dict:
    """CCX reference scalars for one case (results/ccx_reference.json schema)."""
    nan = float("nan")
    out = dict(found=False, u_end=nan, u_mid=nan, sigma_mid=nan,
               mesh=None, element_type=None, load_mode=None,
               source="results/ccx_reference.json")
    entry = _case_entry(ref or {}, case.key)
    if entry is None:
        return out

    def pick(keys):
        for k in keys:
            v = entry.get(k)
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                return float(v)
        return nan

    out.update(
        found=True,
        u_end=pick(("u_end", "u_end_avg", "u_end_ccx")),
        u_mid=pick(("u_mid", "u_mid_avg", "u_mid_ccx", "u_mid_mean")),
        sigma_mid=pick(("sigma_mid", "sigma_mid_avg", "sigma_mid_ccx")),
        mesh=entry.get("mesh", entry.get("reference_mesh")),
        element_type=entry.get("element_type", entry.get("reference_element_type")),
        load_mode=entry.get("load_mode"),
    )
    profile = entry.get("profile")
    if profile is None:
        profile = entry.get("profiles")
    x, arrs = _profile_arrays(profile)
    if x is not None:
        if not np.isfinite(out["u_mid"]):
            out["u_mid"] = _profile_lookup(x, arrs, _U_KEYS, 0.5 * case.geom.L)
        if not np.isfinite(out["u_end"]):
            out["u_end"] = _profile_lookup(x, arrs, _U_KEYS, case.geom.L)
        if not np.isfinite(out["sigma_mid"]):
            out["sigma_mid"] = _profile_lookup(x, arrs, _SIGMA_KEYS,
                                               0.5 * case.geom.L)
    return out


def ccx_posterior(case: Case, ref: dict) -> dict:
    """Posterior 1-D vs CCX comparison at x = L/2 (NaN when unavailable)."""
    nan = float("nan")
    out = dict(ccx_found=False, u_mid_ccx=nan, sigma_mid_ccx=nan,
               e_u_mid_pct_ccx=nan, e_sigma_mid_pct_ccx=nan)
    data = ccx_case_data(case, ref)
    if not data["found"]:
        return out
    geom, mat, P = case.geom, case.material, case.load
    u1d = exact_u_mid(geom, mat, P)
    s1d = float(exact_sigma(geom, mat, P, 0.5 * geom.L))
    u_ccx = data["u_mid"]
    s_ccx = data["sigma_mid"]
    out.update(
        ccx_found=True, u_mid_ccx=u_ccx, sigma_mid_ccx=s_ccx,
        e_u_mid_pct_ccx=(100.0 * (u_ccx - u1d) / u1d
                         if np.isfinite(u_ccx) else nan),
        e_sigma_mid_pct_ccx=(100.0 * (s_ccx - s1d) / s1d
                             if np.isfinite(s_ccx) else nan),
    )
    return out


BAR_VS_CCX_FIELDS = ["case", "label", "eta", "u_end_bar", "u_end_ccx",
                     "rel_end_pct", "u_mid_bar", "u_mid_ccx",
                     "rel_mid_pct", "sigma_mid_bar", "sigma_mid_ccx",
                     "rel_sigma_pct"]


def bar_vs_ccx_row(case: Case, ref: dict, n_elem: int = 160,
                   method: str = "equiv") -> dict:
    """One row of results/bar_vs_ccx.csv (1-D bar vs CCX reference)."""
    geom, mat, P = case.geom, case.material, case.load
    model = build_bar(geom, mat, P, n_elem, method)
    app = applicability(geom)
    data = ccx_case_data(case, ref)
    u_end_bar = float(model.u_end)
    u_mid_bar = float(model.u_mid)
    sigma_mid_bar = float(model.stress_at(0.5 * geom.L))

    def rel(ccx, bar):
        if np.isfinite(ccx) and bar != 0.0:
            return 100.0 * (ccx - bar) / bar
        return float("nan")

    return dict(
        case=case.key, label=app["label"], eta=app["eta"],
        u_end_bar=u_end_bar, u_end_ccx=data["u_end"],
        rel_end_pct=rel(data["u_end"], u_end_bar),
        u_mid_bar=u_mid_bar, u_mid_ccx=data["u_mid"],
        rel_mid_pct=rel(data["u_mid"], u_mid_bar),
        sigma_mid_bar=sigma_mid_bar, sigma_mid_ccx=data["sigma_mid"],
        rel_sigma_pct=rel(data["sigma_mid"], sigma_mid_bar),
    )


def criterion_row(case: Case, ref=None) -> dict:
    """One row of results/bar_criterion.csv (incl. optional CCX posterior)."""
    app = applicability(case.geom)
    geom, mat, P = case.geom, case.material, case.load
    post = ccx_posterior(case, ref or {})
    return dict(
        case=case.key,
        eta=app["eta"],
        angle_deg=app["angle_deg"],
        slenderness=app["slenderness"],
        area_ratio=app["area_ratio"],
        label=app["label"],
        criterion=app["criterion"],
        recommendation=app["recommendation"],
        u_mid_exact_um=1.0e6 * exact_u_mid(geom, mat, P),
        u_mid_ccx_um=(1.0e6 * post["u_mid_ccx"]
                      if np.isfinite(post["u_mid_ccx"]) else float("nan")),
        e_u_mid_pct_ccx=post["e_u_mid_pct_ccx"],
        sigma_mid_ccx=post["sigma_mid_ccx"],
        e_sigma_mid_pct_ccx=post["e_sigma_mid_pct_ccx"],
    )


__all__ = [
    "METHODS", "ETA_A_MAX", "ETA_B_MAX", "LABEL_DESC", "RECOMMENDATIONS",
    "CRITERION_TEXT", "criterion_text", "classify_eta", "bar_nodes",
    "bar_connectivity", "gauss_stiffness", "element_stiffness",
    "element_stiffness_vector", "element_matrix", "assemble_bar", "BarModel",
    "build_bar", "solve_case", "run_record", "node_records", "observed_order",
    "applicability", "load_ccx_reference", "ccx_case_data", "ccx_posterior",
    "bar_vs_ccx_row", "BAR_VS_CCX_FIELDS", "criterion_row", "CCX_REFERENCE_PATH",
]
