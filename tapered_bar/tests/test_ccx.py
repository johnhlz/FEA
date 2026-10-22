"""Tests for the CalculiX anchor."""
from __future__ import annotations

import atexit
import pathlib
import shutil
import sys
import tempfile
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anchors.calculix import ccx_io as C
from fem.config import CASES_BY_KEY, STATIONS
from fem.mesh import make_mesh

_RUNS_DIR = pathlib.Path(tempfile.mkdtemp(prefix="ccx_test_"))
atexit.register(shutil.rmtree, str(_RUNS_DIR), True)
_CACHE = {}


def _run(case_key, nx, ny, et, mode="consistent"):
    key = (case_key, nx, ny, et, mode)
    if key not in _CACHE:
        _CACHE[key] = C.run_case(CASES_BY_KEY[case_key], nx, ny, et,
                                 str(_RUNS_DIR), load_mode=mode,
                                 keep_dat=False, keep_frd=False)
    return _CACHE[key]


# model / input generation
def test_ccx_available():
    assert C.ccx_available(), f"CalculiX executable not found: {C.CCX_EXE_DEFAULT}"


def test_build_model_corner_nodes_match_make_mesh():
    case = CASES_BY_KEY["S1"]
    mesh = make_mesh(case.geom, 20, 4)
    for et in ("CPS4", "CPS8"):
        model = C.build_model(case, nx=20, ny=4, element_type=et)
        assert model.base_nnodes == mesh.nnodes
        assert model.nnodes == mesh.nnodes or et == "CPS8"
        assert np.allclose(model.nodes[:mesh.nnodes, :2], mesh.nodes, atol=0, rtol=0)
        assert np.array_equal(model.elems[:, :4], mesh.quads)
        assert model.nelem == mesh.nquads


def test_cps8_midside_connectivity_and_sharing():
    case = CASES_BY_KEY["S1"]
    mesh = make_mesh(case.geom, 4, 2)
    model = C.build_model(case, nx=4, ny=2, element_type="CPS8")
    n_mid = 4 * (2 + 1) + (4 + 1) * 2       # horizontal + vertical edges
    assert model.nnodes == mesh.nnodes + n_mid
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    for e in range(model.nelem):
        q = mesh.quads[e]
        for (a, b), mid in zip(edges, model.elems[e, 4:]):
            expect = 0.5 * (model.nodes[q[a], :2] + model.nodes[q[b], :2])
            assert np.allclose(model.nodes[mid, :2], expect, atol=1e-12)
    # element 0 and element 1 (same x column, adjacent j) share one midside node:
    # element 0 edge n3-n4 (midside slot 6) == element 1 edge n1-n2 (slot 4)
    assert model.elems[0, 6] == model.elems[1, 4]


def test_node_sets_include_midside_nodes():
    case = CASES_BY_KEY["S0"]
    case_g = case.geom
    m8 = C.build_model(case, nx=20, ny=4, element_type="CPS8")
    assert m8.left_nodes.size == 2 * 4 + 1          # corners + midside on x=0
    assert m8.right_nodes.size == 2 * 4 + 1
    assert np.allclose(m8.nodes[m8.left_nodes, 0], 0.0)
    assert np.allclose(m8.nodes[m8.right_nodes, 0], case_g.L)
    m4 = C.build_model(case, nx=20, ny=4, element_type="CPS4")
    assert m4.left_nodes.size == 5
    assert m4.right_nodes.size == 5


def test_inp_generation_cps4_and_cps8():
    case = CASES_BY_KEY["S0"]
    tmp = pathlib.Path(tempfile.mkdtemp(dir=str(_RUNS_DIR)))
    m4 = C.build_model(case, nx=8, ny=2, element_type="CPS4")
    p4 = C.write_inp(m4, str(tmp / "cps4.inp"), load_mode="consistent")
    txt = pathlib.Path(p4).read_text()
    assert "*ELEMENT, TYPE=CPS4, ELSET=EALL" in txt
    assert "*ELASTIC" in txt and "2.000000000000e+11" in txt
    assert "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1" in txt
    assert "\n1.000000000000e-01\n" in txt          # thickness b = 0.1
    assert "LEFT, 1, 2" in txt
    assert "*NODE FILE" in txt and "*EL FILE" in txt
    # node count
    node_lines = txt.split("*NODE, NSET=NALL\n")[1].split("*ELEMENT")[0].strip().splitlines()
    assert len(node_lines) == m4.nnodes
    # CLOAD block: one line per loaded node, total P
    cload = txt.split("*CLOAD\n")[1].split("*NODE PRINT")[0].strip().splitlines()
    assert len(cload) == m4.right_nodes.size
    forces = [float(line.split(",")[2]) for line in cload]
    assert abs(sum(forces) - case.load) < 1e-9 * case.load
    # CalculiX field-length constraint: no real field longer than 20 chars
    for line in node_lines:
        for field in line.split(",")[1:]:
            assert len(field.strip()) <= 20
    m8 = C.build_model(case, nx=4, ny=2, element_type="CPS8")
    p8 = C.write_inp(m8, str(tmp / "cps8.inp"))
    txt8 = pathlib.Path(p8).read_text()
    assert "*ELEMENT, TYPE=CPS8, ELSET=EALL" in txt8
    elem_lines = txt8.split("*ELEMENT, TYPE=CPS8, ELSET=EALL\n")[1].split("*NSET")[0].strip().splitlines()
    assert len(elem_lines) == m8.nelem
    assert all(len(line.split(",")) == 9 for line in elem_lines)   # id + 8 nodes
    nset = txt8.split("*NSET, NSET=LEFT\n")[1].split("*NSET")[0]
    assert all(len(line.split(",")) <= 16 for line in nset.strip().splitlines())


def test_consistent_and_lumped_load_vectors():
    case = CASES_BY_KEY["S0"]
    # CPS4: edge load P/ny, corners half, interior nodes full
    m4 = C.build_model(case, nx=20, ny=4, element_type="CPS4")
    cons = C.nodal_loads(m4, "consistent")
    lump = C.nodal_loads(m4, "equal")
    assert abs(sum(cons.values()) - case.load) < 1e-9 * case.load
    assert abs(sum(lump.values()) - case.load) < 1e-9 * case.load
    ny = m4.ny
    w = [cons[int(k)] for k in m4.right_nodes]      # right_nodes sorted by j
    assert np.isclose(w[0], 0.5 * case.load / ny)
    assert np.isclose(w[-1], 0.5 * case.load / ny)
    assert np.allclose(w[1:-1], case.load / ny)
    # CPS8: corner ends P/(6ny), interior corners P/(3ny), midside 2P/(3ny)
    m8 = C.build_model(case, nx=20, ny=4, element_type="CPS8")
    cons8 = C.nodal_loads(m8, "consistent")
    assert abs(sum(cons8.values()) - case.load) < 1e-9 * case.load
    # right_nodes lists the (ny+1) corner nodes first, then the ny midside nodes
    weights = [cons8[int(k)] for k in m8.right_nodes]
    corners, mids = weights[:ny + 1], weights[ny + 1:]
    assert len(mids) == ny
    assert np.isclose(corners[0], case.load / (6.0 * ny))
    assert np.isclose(corners[-1], case.load / (6.0 * ny))
    assert np.allclose(corners[1:-1], 2.0 * case.load / (6.0 * ny))
    assert np.allclose(mids, 4.0 * case.load / (6.0 * ny))


# complete runs
def test_cps4_run_reaction_and_theory_magnitude():
    case = CASES_BY_KEY["S0"]
    r = _run("S0", 20, 4, "CPS4")
    assert abs(r["reaction_x"] + case.load) < 1e-6 * case.load
    assert r["n_ip"] == 8                          # C3D8 expansion
    assert len(r["dat"]["displacements"]) == r["nnodes"]
    assert len(r["dat"]["stresses"]) == r["nelem"]
    u_exact = case.load * case.geom.L / (case.material.E * case.geom.A0)
    assert abs(r["u_end"] / u_exact - 1.0) < 0.03   # Q4 on 20x4: < 3 %
    assert abs(r["sigma_mid"] / 50000.0 - 1.0) < 0.01
    assert len(r["profiles"]) == 11
    assert np.allclose([p["x"] for p in r["profiles"]], STATIONS)


def test_cps8_run_frd_parse_and_reaction():
    case = CASES_BY_KEY["S0"]
    r = _run("S0", 20, 4, "CPS8")
    assert abs(r["reaction_x"] + case.load) < 1e-6 * case.load
    assert r["n_ip"] == 27                         # C3D20 expansion
    u_exact = case.load * case.geom.L / (case.material.E * case.geom.A0)
    assert abs(r["u_end"] / u_exact - 1.0) < 0.02
    # u_end is the trapezoidal average of the .dat right-column ux values
    m = r["model"]
    nodes = m.column_nodes(m.nx)
    ys = m.nodes[nodes, 1]
    order = np.argsort(ys)
    us = np.array([r["dat"]["displacements"][int(nodes[i]) + 1][0] for i in order])
    yy = ys[order]
    dy = np.diff(yy)
    w = np.zeros_like(yy)
    w[:-1] += 0.5 * dy
    w[1:] += 0.5 * dy
    assert np.isclose(r["u_end"], np.sum(w * us) / np.sum(w), rtol=1e-12)
    frd = r["frd"]
    assert frd is not None
    assert len(frd["disp"]) == r["nnodes"]
    assert len(frd["stress"]) == r["nnodes"]
    assert len(frd["force"]) == r["nnodes"]
    # .frd nodal displacement agrees with .dat (frd prints ~6 significant digits)
    for nid in (1, 2, r["model"].base_nnodes // 2, r["nnodes"]):
        u_dat = np.array(r["dat"]["displacements"][nid])
        u_frd = frd["disp"][nid]
        assert np.allclose(u_frd, u_dat, rtol=1e-4, atol=1e-13)
    # .frd stress is nodal: 6 components
    s = next(iter(frd["stress"].values()))
    assert s.size == 6
    # probe table columns and point count (ny=4 -> 5 y levels * 5 x)
    probes = r["probes"]
    assert len(probes) == 25
    assert list(probes[0].keys()) == ["case", "x", "y", "ux", "uy", "sxx",
                                      "syy", "sxy", "mesh", "element_type"]
    h = case.geom.h(0.1)
    ys = sorted({round(p["y"], 12) for p in probes if abs(p["x"] - 0.1) < 1e-12})
    assert np.allclose(ys, [-h / 2, -h / 4, 0.0, h / 4, h / 2], atol=1e-12)


def test_consistent_load_gives_uniform_bar_stress():
    """S0 with consistent traction: mid-section stress must be ~P/A = 5e4."""
    r = _run("S0", 20, 4, "CPS8")
    mid = [p for p in r["probes"] if abs(p["x"] - 0.5) < 1e-12 and abs(p["y"]) < 1e-12]
    assert len(mid) == 1
    assert abs(mid[0]["sxx"] / 50000.0 - 1.0) < 0.005


def test_section_quantities_definition():
    case = CASES_BY_KEY["S0"]
    r = _run("S0", 20, 4, "CPS4")
    for p in r["profiles"]:
        assert np.isclose(p["sigma_nom"], case.load / case.geom.A(p["x"]))
        assert np.isfinite(p["u_section_avg"]) and np.isfinite(p["sigma_section_avg"])
    # symmetry of the bar: u at x=0 must be zero (fixed end)
    assert abs(r["profiles"][0]["u_section_avg"]) < 1e-12
    # u_section_avg is monotonically increasing for the uniform bar
    us = [p["u_section_avg"] for p in r["profiles"]]
    assert all(b > a for a, b in zip(us, us[1:]))


def test_section_average_u_trapezoid_weights():
    """u average = trapezoid over actual y (end weights 0.5, inner 1)."""
    case = CASES_BY_KEY["S0"]
    m = C.build_model(case, nx=20, ny=4, element_type="CPS8")
    k = m.nx // 2
    nodes = m.column_nodes(k)
    y = m.nodes[nodes, 1]
    order = np.argsort(y)
    # ux = 1 exactly at the two end nodes, 0 elsewhere: trapezoid average is
    # (0.5 + 0.5) / (2*ny) = 0.5/ny, whereas the arithmetic node mean would be
    # 2/(2*ny+1) (the legacy O(1/ny) bias).
    displ = {n + 1: (0.0, 0.0, 0.0) for n in range(m.nnodes)}
    displ[int(nodes[order[0]]) + 1] = (1.0, 0.0, 0.0)
    displ[int(nodes[order[-1]]) + 1] = (1.0, 0.0, 0.0)
    assert np.isclose(C.section_average_u(m, displ, k), 0.5 / m.ny, rtol=1e-14)
    # a constant ux field must average to itself
    displ2 = {n + 1: (2.5, 0.0, 0.0) for n in range(m.nnodes)}
    assert np.isclose(C.section_average_u(m, displ2, k), 2.5, rtol=1e-14)


def test_probe_nodes_y_levels():
    case = CASES_BY_KEY["S1"]
    m2 = C.build_model(case, nx=20, ny=2, element_type="CPS8")
    p2 = C.probe_nodes(m2)
    assert len(p2) == 15                            # ny=2: y/h in {0, +-1/2}
    m4 = C.build_model(case, nx=20, ny=4, element_type="CPS8")
    p4 = C.probe_nodes(m4)
    assert len(p4) == 25
    h = case.geom.h(0.2)
    ys = sorted({round(p["y"], 12) for p in p4 if abs(p["x"] - 0.2) < 1e-12})
    assert np.allclose(ys, [-h / 2, -h / 4, 0.0, h / 4, h / 2], atol=1e-12)


def test_parse_dat_blocks_of_cached_run():
    r = _run("S0", 20, 4, "CPS4")
    d = r["dat"]
    assert len(d["displacements"]) == r["nnodes"]
    assert len(d["forces"]) == r["nnodes"]
    assert sum(v.shape[0] for v in d["stresses"].values()) == r["nelem"] * 8


def _run_all():
    names = [n for n in globals() if n.startswith("test_") and callable(globals()[n])]
    failures = []
    for name in names:
        try:
            globals()[name]()
        except Exception as exc:
            failures.append(name)
            print("FAIL", name, ":", repr(exc))
            traceback.print_exc()
        else:
            print("PASS", name)
    print("ok %d/%d" % (len(names) - len(failures), len(names)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
