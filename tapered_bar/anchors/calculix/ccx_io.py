"""CalculiX plane-stress I/O."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import subprocess
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fem.config import Case, Geometry, Material
from fem.mesh import Mesh, make_mesh

CCX_EXE_DEFAULT = "/home/john/miniconda3/envs/fem-ccx/bin/ccx"
FMT = "{:.12e}"          # CalculiX string-field limit (%.16e breaks *NODE)

# station list of the study: x/L = 0.0, 0.1, ..., 1.0
STATIONS = np.round(np.linspace(0.0, 1.0, 11), 10)
PROBE_XI = (0.1, 0.2, 0.5, 0.8, 0.9)      # x/L for the nodal probe table

# regex for a real number in .dat / .frd (always has a decimal point)
_NUM_RE = re.compile(r"[-+]?\d+\.\d*(?:[EeDd][-+]?\d+)?")
_NODE_RE = re.compile(r"^\s*-1\s+(\d+)")
_LEAD_INT_RE = re.compile(r"^\s*(\d+)")


# model construction
@dataclass
class CcxModel:
    """Plane-stress CalculiX model: 2D nodes (z=0), elements, node sets."""
    case_key: str
    geom: Geometry
    mat: Material
    load: float
    nx: int
    ny: int
    element_type: str
    nodes: np.ndarray             # (nn, 3) float, base nodes first, then midside
    elems: np.ndarray             # (ne, nen) int, 0-based
    base_nnodes: int
    mesh: Mesh
    left_nodes: np.ndarray        # 0-based ids, all nodes at x=0
    right_nodes: np.ndarray       # 0-based ids, all nodes at x=L
    edge_mid: Dict[Tuple[int, int], int] = field(default_factory=dict)

    @property
    def nnodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def nelem(self) -> int:
        return int(self.elems.shape[0])

    @property
    def nen(self) -> int:
        return int(self.elems.shape[1])

    @property
    def thickness(self) -> float:
        return float(self.geom.width)

    @property
    def tag(self) -> str:
        return f"{self.case_key}_{self.element_type.lower()}_{self.nx}x{self.ny}"

    def column_nodes(self, i: int, tol: float = 1e-9) -> np.ndarray:
        """0-based ids of all nodes (including midside) in axial column i."""
        x = i * self.geom.L / self.nx
        return np.flatnonzero(np.abs(self.nodes[:, 0] - x) < tol)

    def element_column_mask(self, i: int) -> np.ndarray:
        """Boolean mask of elements in axial element column i (x_i..x_{i+1})."""
        e = np.arange(self.nelem)
        return (e // self.ny) == i


def build_model(case: Optional[Case] = None, *, geom: Optional[Geometry] = None,
                mat: Optional[Material] = None, load: Optional[float] = None,
                nx: int, ny: int, element_type: str = "CPS8") -> CcxModel:
    """Build a CcxModel for a case/geometry and mesh size."""
    if case is None:
        if geom is None:
            raise ValueError("provide case= or geom=")
        mat = mat or Material()
        load = float(load if load is not None else 1000.0)
        key = "CUSTOM"
    else:
        geom = geom or case.geom
        mat = mat or case.material
        load = float(load if load is not None else case.load)
        key = case.key

    element_type = element_type.upper()
    if element_type not in ("CPS4", "CPS8"):
        raise ValueError("element_type must be CPS4 or CPS8")

    mesh = make_mesh(geom, nx, ny)
    base = mesh.nnodes
    nodes = np.zeros((base, 3))
    nodes[:, :2] = mesh.nodes

    if element_type == "CPS4":
        elems = np.array(mesh.quads, dtype=int)
        edge_mid: Dict[Tuple[int, int], int] = {}
    else:
        edge_mid = {}
        elems = np.empty((mesh.nquads, 8), dtype=int)
        next_id = base
        for e, q in enumerate(mesh.quads):
            mids = []
            for a, b in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
                k = (a, b) if a < b else (b, a)
                m = edge_mid.get(k)
                if m is None:
                    m = next_id
                    next_id += 1
                    edge_mid[k] = m
                mids.append(m)
            elems[e, :4] = q
            elems[e, 4:] = mids
        extra = np.zeros((next_id - base, 3))
        for (a, b), m in edge_mid.items():
            extra[m - base, :2] = 0.5 * (nodes[a, :2] + nodes[b, :2])
        nodes = np.vstack([nodes, extra])

    # node sets: *all* nodes on the fixed / loaded ends (midside included)
    tol = 1e-9 * max(geom.L, 1.0)
    left = np.flatnonzero(np.abs(nodes[:, 0] - 0.0) < tol)
    right = np.flatnonzero(np.abs(nodes[:, 0] - geom.L) < tol)
    if left.size < 2 or right.size < 2:
        raise RuntimeError("empty left/right node set")

    return CcxModel(case_key=key, geom=geom, mat=mat, load=load, nx=nx, ny=ny,
                    element_type=element_type, nodes=nodes, elems=elems,
                    base_nnodes=base, mesh=mesh, left_nodes=left,
                    right_nodes=right, edge_mid=edge_mid)


# load definition
def nodal_loads(model: CcxModel, mode: str = "equal") -> Dict[int, float]:
    """Return {0-based node id: axial force} with total force = P."""
    P = model.load
    if mode == "equal":
        n = model.right_nodes.size
        return {int(k): P / n for k in model.right_nodes}
    if mode == "consistent":
        w: Dict[int, float] = {}
        ny, nx = model.ny, model.nx
        base_id = lambda i, j: i * (ny + 1) + j          # noqa: E731
        edge_force = P / ny
        for j in range(ny):
            a, b = base_id(nx, j), base_id(nx, j + 1)
            if model.element_type == "CPS4":
                w[a] = w.get(a, 0.0) + 0.5 * edge_force
                w[b] = w.get(b, 0.0) + 0.5 * edge_force
            else:
                m = model.edge_mid[(min(a, b), max(a, b))]
                w[a] = w.get(a, 0.0) + edge_force / 6.0
                w[m] = w.get(m, 0.0) + 2.0 * edge_force / 3.0
                w[b] = w.get(b, 0.0) + edge_force / 6.0
        return w
    raise ValueError("mode must be 'equal' or 'consistent'")


# input deck
def _write_id_lines(ap, ids, per_line: int = 16) -> None:
    """Write 1-based node ids, max per_line entries per line (CalculiX limit)."""
    ids = [int(v) + 1 for v in ids]
    for k in range(0, len(ids), per_line):
        ap(", ".join(str(v) for v in ids[k:k + per_line]))


def write_inp(model: CcxModel, path: str, *, load_mode: str = "consistent",
              write_frd: bool = True, solver: Optional[str] = None) -> str:
    """Write a complete CalculiX plane-stress input deck and return its path."""
    loads = nodal_loads(model, mode=load_mode)
    total = sum(loads.values())
    if abs(total - model.load) > 1e-8 * abs(model.load):
        raise RuntimeError(f"load resultant {total} != P={model.load}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    L = []
    ap = L.append
    ap("*HEADING")
    ap(f"Tapered bar {model.case_key} beta={model.geom.beta:g} "
       f"h0={model.geom.h0:g} E={model.mat.E:g} nu={model.mat.nu:g} "
       f"P={model.load:g} {model.element_type} {model.nx}x{model.ny} "
       f"load={load_mode}")
    ap("*NODE, NSET=NALL")
    for k, (x, y, z) in enumerate(model.nodes, start=1):
        ap(f"{k}, " + ", ".join(FMT.format(v) for v in (x, y, z)))
    ap(f"*ELEMENT, TYPE={model.element_type}, ELSET=EALL")
    for k, conn in enumerate(model.elems, start=1):
        ap(str(k) + ", " + ", ".join(str(int(v) + 1) for v in conn))
    ap("*NSET, NSET=LEFT")
    _write_id_lines(ap, model.left_nodes)
    ap("*NSET, NSET=RIGHT")
    _write_id_lines(ap, model.right_nodes)
    ap("*BOUNDARY")
    ap("LEFT, 1, 2")
    ap("*MATERIAL, NAME=MAT1")
    ap("*ELASTIC")
    ap(f"{FMT.format(model.mat.E)}, {FMT.format(model.mat.nu)}")
    ap("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1")
    ap(FMT.format(model.thickness))
    ap("*STEP")
    if solver:
        ap(f"*STATIC, SOLVER={solver}")
    else:
        ap("*STATIC")
    ap("*CLOAD")
    for nid in sorted(loads):
        ap(f"{nid + 1}, 1, {FMT.format(loads[nid])}")
    ap("*NODE PRINT, NSET=NALL")
    ap("U, RF")
    ap("*EL PRINT, ELSET=EALL")
    ap("S")
    if write_frd:
        ap("*NODE FILE")
        ap("U, RF")
        ap("*EL FILE")
        ap("S")
    ap("*END STEP")
    text = "\n".join(L) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ccx execution
def ccx_available(exe: str = CCX_EXE_DEFAULT) -> bool:
    return os.path.isfile(exe) and os.access(exe, os.X_OK)


def run_ccx(inp_path: str, *, exe: str = CCX_EXE_DEFAULT, timeout: float = 3600.0,
            keep_dat: bool = True, keep_frd: bool = True) -> dict:
    """Run ccx -i <job> in the directory of inp_path and return output paths."""
    inp_path = os.path.abspath(inp_path)
    workdir = os.path.dirname(inp_path)
    stem = os.path.splitext(os.path.basename(inp_path))[0]
    if not ccx_available(exe):
        raise FileNotFoundError(f"CalculiX executable not found: {exe}")
    log_path = os.path.join(workdir, stem + ".ccx.log")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run([exe, "-i", stem], cwd=workdir,
                              stdout=log, stderr=subprocess.STDOUT,
                              timeout=timeout)
    wall = time.time() - t0
    dat_path = os.path.join(workdir, stem + ".dat")
    frd_path = os.path.join(workdir, stem + ".frd")
    if proc.returncode != 0 or not os.path.isfile(dat_path):
        tail = ""
        if os.path.isfile(log_path):
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-30:])
        raise RuntimeError(f"ccx failed (rc={proc.returncode}) for {inp_path}\n{tail}")
    out = dict(inp_path=inp_path, workdir=workdir, stem=stem,
               dat_path=dat_path, frd_path=frd_path if os.path.isfile(frd_path) else None,
               log_path=log_path, returncode=proc.returncode, wall_time_s=wall)
    for suffix in (".12d", ".cvg", ".sta"):
        p = os.path.join(workdir, stem + suffix)
        if os.path.isfile(p):
            os.remove(p)
    spool = os.path.join(workdir, "spooles.out")
    if os.path.isfile(spool):
        os.remove(spool)
    if not keep_frd and out["frd_path"]:
        os.remove(out["frd_path"])
        out["frd_path"] = None
    if not keep_dat:
        os.remove(dat_path)
        out["dat_path"] = None
    return out


# .dat parsing (integration-point stresses, nodal displacements/forces)
def parse_dat(path: str) -> dict:
    """Parse CalculiX .dat output."""
    displ: Dict[int, Tuple[float, float, float]] = {}
    forces: Dict[int, Tuple[float, float, float]] = {}
    stress_rows: Dict[int, List[Tuple[float, ...]]] = {}
    block = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue          # blank line inside a .dat block: keep block
            low = s.lower()
            if low.startswith("displacements ("):
                block = "displ"
                continue
            if low.startswith("forces ("):
                block = "forces"
                continue
            if low.startswith("stresses ("):
                block = "stress"
                continue
            if block is None:
                continue
            if block in ("displ", "forces"):
                vals = _NUM_RE.findall(s)
                if len(vals) != 3:
                    continue
                m = _LEAD_INT_RE.match(s)
                if not m:
                    continue
                nid = int(m.group(1))
                tup = tuple(float(v) for v in vals)
                if block == "displ":
                    displ[nid] = tup
                else:
                    forces[nid] = tup
            else:  # stress
                vals = _NUM_RE.findall(s)
                if len(vals) != 6:
                    continue
                toks = s.split()
                eid, ip = int(toks[0]), int(toks[1])
                stress_rows.setdefault(eid, []).append(tuple(float(v) for v in vals))
    stresses = {e: np.asarray(rows, dtype=float) for e, rows in stress_rows.items()}
    return dict(displacements=displ, forces=forces, stresses=stresses)


# .frd parsing (nodal DISP / STRESS / FORC)
def parse_frd(path: str) -> dict:
    """Parse the result part of a CalculiX .frd file."""
    nodes: Dict[int, Tuple[float, float, float]] = {}
    results: Dict[str, Dict[int, np.ndarray]] = {"disp": {}, "stress": {}, "force": {}}
    block = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("    2C"):
                block = "mesh"
                continue
            stripped = line.rstrip("\n")
            if stripped.startswith(" -4"):
                name = stripped[3:].split()[0].upper() if stripped[3:].strip() else ""
                if name == "DISP":
                    block = "disp"
                elif name == "STRESS":
                    block = "stress"
                elif name in ("FORC", "FORCE"):
                    block = "force"
                else:
                    block = "other"
                continue
            if stripped.startswith(" -3"):
                block = None
                continue
            if block is None or not stripped.startswith(" -1"):
                continue
            m = _NODE_RE.match(stripped)
            if not m:
                continue
            nid = int(m.group(1))
            vals = _NUM_RE.findall(stripped[m.end():])
            arr = np.asarray([float(v) for v in vals], dtype=float)
            if block == "mesh":
                if arr.size == 3:
                    nodes[nid] = (float(arr[0]), float(arr[1]), float(arr[2]))
            elif block in results and arr.size:
                results[block][nid] = arr
    return dict(nodes=nodes, **results)


# post-processing
def element_mean_stress(model: CcxModel, stresses: Dict[int, np.ndarray]) -> np.ndarray:
    """(ne, 6) element-average stress in input element order."""
    out = np.full((model.nelem, 6), np.nan)
    for e in range(1, model.nelem + 1):
        s = stresses.get(e)
        if s is None:
            continue
        out[e - 1] = s.mean(axis=0)
    if np.isnan(out).any():
        raise RuntimeError("missing stress output for some elements")
    return out


def section_quantities(model: CcxModel, displ: Dict[int, Tuple[float, ...]],
                       elem_sxx: np.ndarray,
                       stations: Sequence[float] = STATIONS) -> List[dict]:
    """Section averages at x/L = stations."""
    L = model.geom.L
    rows = []
    elem_sxx = np.asarray(elem_sxx, dtype=float)
    for xi in stations:
        k = int(round(float(xi) * model.nx))
        xv = k * L / model.nx
        nodes = model.column_nodes(k)
        if nodes.size == 0:
            raise RuntimeError(f"no nodes at station x/L={xi}")
        u = float(np.mean([displ[int(n) + 1][0] for n in nodes]))
        cols = [0] if k == 0 else ([model.nx - 1] if k == model.nx else [k - 1, k])
        eids: List[int] = []
        for c in cols:
            eids.extend(int(e) for e in np.flatnonzero(model.element_column_mask(c)))
        sig = float(np.mean(elem_sxx[eids, 0]))
        rows.append(dict(x=float(xv), u_section_avg=u, sigma_section_avg=sig,
                         sigma_nom=float(model.load / model.geom.A(xv))))
    return rows


def section_average_u(model: CcxModel, displ: Dict[int, Tuple[float, ...]], k: int) -> float:
    """Trapezoidal section average of ux over node column k."""
    nodes = model.column_nodes(k)
    y = np.asarray(model.nodes[nodes, 1], dtype=float)
    order = np.argsort(y)
    nodes = nodes[order]
    y = y[order]
    u = np.asarray([displ[int(n) + 1][0] for n in nodes], dtype=float)
    dy = np.diff(y)
    w = np.zeros_like(y)
    w[:-1] += 0.5 * dy
    w[1:] += 0.5 * dy
    area = w.sum()
    if area <= 0:
        raise RuntimeError("degenerate node column")
    return float(np.sum(w * u) / area)


def reaction_axial(model: CcxModel, forces: Dict[int, Tuple[float, ...]]) -> float:
    """Total x reaction at the fixed end (CalculiX reports it in forces)."""
    return float(sum(forces[int(n) + 1][0] for n in model.left_nodes))


def probe_nodes(model: CcxModel, xi_values: Sequence[float] = PROBE_XI) -> List[dict]:
    """Corner nodes at x/L=xi and y/h in {0, +-1/4, +-1/2} when they exist."""
    L, ny = model.geom.L, model.ny
    if ny % 4 == 0:
        j_offsets = (-ny // 2, -ny // 4, 0, ny // 4, ny // 2)
    else:
        j_offsets = (-ny // 2, 0, ny // 2)
    out = []
    for xi in xi_values:
        i = int(round(float(xi) * model.nx))
        x = i * L / model.nx
        h = float(model.geom.h(x))
        for dj in j_offsets:
            j = ny // 2 + dj
            nid = model.mesh.node_index(i, j)          # corner-node numbering
            y = (j / ny - 0.5) * h
            out.append(dict(xi=float(xi), x=float(x), y=float(y), j=int(j),
                            node=nid))
    return out


def probe_table(model: CcxModel, frd: dict,
                xi_values: Sequence[float] = PROBE_XI) -> List[dict]:
    """Rows (case,x,y,ux,uy,sxx,syy,sxy,mesh,element_type) from .frd nodes."""
    rows = []
    mesh_tag = f"{model.nx}x{model.ny}"
    for p in probe_nodes(model, xi_values):
        nid = p["node"] + 1
        if nid not in frd["disp"] or nid not in frd["stress"]:
            raise RuntimeError(f"node {nid} missing in .frd results")
        u = frd["disp"][nid]
        s = frd["stress"][nid]
        rows.append(dict(case=model.case_key, x=p["x"], y=p["y"],
                         ux=float(u[0]), uy=float(u[1]),
                         sxx=float(s[0]), syy=float(s[1]), sxy=float(s[3]),
                         mesh=mesh_tag, element_type=model.element_type))
    return rows


# single-run convenience wrapper
def run_case(case: Case, nx: int, ny: int, element_type: str,
             runs_root: str, *, load_mode: str = "consistent", timeout: float = 3600.0,
             keep_dat: bool = False, keep_frd: bool = False,
             solver: Optional[str] = None) -> dict:
    """Build, run and post-process one CalculiX job; returns a result dict."""
    model = build_model(case, nx=nx, ny=ny, element_type=element_type)
    jobdir = os.path.join(runs_root, model.tag)
    os.makedirs(jobdir, exist_ok=True)
    inp = write_inp(model, os.path.join(jobdir, "job.inp"),
                    load_mode=load_mode, write_frd=True, solver=solver)
    run = run_ccx(inp, timeout=timeout, keep_dat=True, keep_frd=True)
    dat = parse_dat(run["dat_path"])
    frd = parse_frd(run["frd_path"]) if run["frd_path"] else None
    elem_s = element_mean_stress(model, dat["stresses"])
    profiles = section_quantities(model, dat["displacements"], elem_s)
    k_mid = model.nx // 2
    u_end = section_average_u(model, dat["displacements"], model.nx)
    u_mid = section_average_u(model, dat["displacements"], k_mid)
    sig_mid = float(profiles[5]["sigma_section_avg"])       # station x/L = 0.5
    reaction = reaction_axial(model, dat["forces"])
    probes = probe_table(model, frd) if frd and frd["stress"] else []
    if not keep_frd and run["frd_path"]:
        os.remove(run["frd_path"])
        run["frd_path"] = None
    if not keep_dat:
        os.remove(run["dat_path"])
        run["dat_path"] = None
    return dict(model=model, run=run, dat=dat, frd=frd, elem_stress=elem_s,
                profiles=profiles, u_end=u_end, u_mid=u_mid,
                sigma_mid=sig_mid, reaction_x=reaction, probes=probes,
                nnodes=model.nnodes, nelem=model.nelem,
                n_ip=int(next(iter(dat["stresses"].values())).shape[0]),
                wall_time_s=run["wall_time_s"], jobdir=jobdir, inp=inp)


__all__ = [
    "CCX_EXE_DEFAULT", "STATIONS", "PROBE_XI", "CcxModel", "build_model",
    "nodal_loads", "write_inp", "ccx_available", "run_ccx", "parse_dat",
    "parse_frd", "element_mean_stress", "section_quantities",
    "section_average_u", "reaction_axial", "probe_nodes", "probe_table",
    "run_case",
]
