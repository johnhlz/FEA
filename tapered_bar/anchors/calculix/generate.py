"""Write a CalculiX .inp."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from anchors.calculix import ccx_io as C
from fem.config import CASES_BY_KEY


def main(argv=None):
    ap = argparse.ArgumentParser(description="generate one CalculiX .inp deck")
    ap.add_argument("--case", default="S1", choices=sorted(CASES_BY_KEY))
    ap.add_argument("--mesh", default="160x32", help="nx x ny (e.g. 160x32)")
    ap.add_argument("--type", dest="element_type", default="CPS8",
                    choices=("CPS4", "CPS8"))
    ap.add_argument("--load-mode", default="equal",
                    choices=("equal", "consistent"))
    ap.add_argument("--solver", default=None)
    ap.add_argument("--out", default=None,
                    help="output .inp path (default anchors/calculix/runs/<tag>/job.inp)")
    args = ap.parse_args(argv)
    nx, ny = (int(v) for v in args.mesh.lower().replace("x", " ").split())
    case = CASES_BY_KEY[args.case]
    model = C.build_model(case, nx=nx, ny=ny, element_type=args.element_type)
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "runs", model.tag, "job.inp")
    C.write_inp(model, out, load_mode=args.load_mode, solver=args.solver)
    print(f"wrote {out}")
    print(f"  {model.element_type} {model.nx}x{model.ny}: nnodes={model.nnodes} "
          f"nelem={model.nelem} left={model.left_nodes.size} "
          f"right={model.right_nodes.size} loads={len(C.nodal_loads(model, args.load_mode))}")


if __name__ == "__main__":
    main()
