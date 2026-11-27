"""Prepare a reusable tetrahedral mesh or solve a prepared 3D FEM design."""

import json
import sys
from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path

import meshio
import numpy as np
from skfem import MeshTet

from examples.postprocessing.fem import write_solution


def mesh_statistics(mesh):
    corners = mesh.p[:, mesh.t].transpose(2, 1, 0)
    volumes = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6
    if not volumes.size or not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("mesh must contain finite, nondegenerate tetrahedra")
    return {
        "points": int(mesh.nvertices),
        "tetrahedra": int(mesh.nelements),
        "min_volume_mm3": float(volumes.min()),
        "max_volume_mm3": float(volumes.max()),
    }


def prepare(module, design, scale, output):
    module.PROBLEM.check_design(design, scale)
    if output.exists():
        raise FileExistsError(output)
    mesh = module.prepare_mesh(design, scale)
    metadata = {
        "case": module.PROBLEM.name,
        "design": design,
        "mesh_scale": scale,
        "mesh": mesh_statistics(mesh),
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "mesh.npz",
        points=mesh.p,
        cells=mesh.t,
        metadata=json.dumps(metadata),
    )
    meshio.write_points_cells(output / "mesh.vtu", mesh.p.T, [("tetra", mesh.oriented().t.T)])
    (output / "mesh.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def solve(mesh_file, output, cases):
    with np.load(mesh_file, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["case"] not in cases:
            raise ValueError(f"unknown FEM case {metadata['case']!r} in {mesh_file}")
        mesh = MeshTet(data["points"], data["cells"])
    module = import_module(metadata["case"])
    problem = module.PROBLEM
    design = metadata["design"]
    problem.check_design(design, metadata["mesh_scale"])
    metadata["mesh"] = mesh_statistics(mesh)
    if output.exists():
        raise FileExistsError(output)
    fields = {}
    metrics = module.solve_mesh(design, mesh, fields)
    record = {
        **metadata,
        "metrics": metrics,
        "feasible": problem.is_feasible(metrics),
    }
    output.mkdir(parents=True, exist_ok=False)
    record["fields"] = write_solution(output / "solution.vtu", fields)
    (output / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def cantilever(args):
    from cantilever import (
        DEFLECTION_LIMIT_MM,
        LENGTH_M,
        ROOT_HEIGHT_BOUNDS_MM,
        TIP_RATIO_BOUNDS,
        solve_beam,
    )

    root_min, root_max = ROOT_HEIGHT_BOUNDS_MM
    tip_min, tip_max = TIP_RATIO_BOUNDS
    if not root_min <= args.root_height_mm <= root_max:
        raise ValueError(f"root height must be {root_min}–{root_max} mm")
    if not tip_min <= args.tip_ratio <= tip_max:
        raise ValueError(f"tip ratio must be {tip_min:g}–{tip_max:g}")
    if args.elements < 1:
        raise ValueError("elements must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)

    mass, basis, displacement = solve_beam(args.root_height_mm, args.tip_ratio, args.elements)
    # Sample the Hermite solution at eight intervals per element for a VTK line grid.
    x = np.linspace(0, LENGTH_M, 8 * args.elements + 1)
    down = -1_000 * basis.interpolator(displacement)(x[None, :])
    points = np.column_stack((1_000 * x, np.zeros_like(x), np.zeros_like(x)))
    cells = np.column_stack((np.arange(x.size - 1), np.arange(1, x.size)))
    record = {
        "case": "cantilever",
        "design": {"root_height_mm": args.root_height_mm, "tip_ratio": args.tip_ratio},
        "mesh": {"hermite_elements": args.elements},
        "metrics": {"mass_kg": mass, "tip_deflection_mm": float(down[-1])},
        "feasible": bool(down[-1] <= DEFLECTION_LIMIT_MM),
        "fields": {"point": ["downward_deflection_mm"], "cell": []},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    meshio.write_points_cells(
        args.output / "solution.vtu",
        points,
        [("line", cells)],
        point_data={"downward_deflection_mm": down},
    )
    (args.output / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main():
    # The existing case scripts import their shared modules from examples/fem.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fem"))
    from common.cases import CASES

    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    mesh_parser = commands.add_parser("prepare", help="mesh a reference or specified design")
    mesh_parser.add_argument("case", choices=CASES)
    mesh_parser.add_argument("output", type=Path)
    mesh_parser.add_argument("--mesh-scale", type=float, default=1.0)
    mesh_parser.add_argument("--design", type=json.loads, help="complete design as a JSON object")
    solve_parser = commands.add_parser("solve", help="solve and export fields on a prepared mesh")
    solve_parser.add_argument("mesh", type=Path, help="mesh.npz from prepare")
    solve_parser.add_argument("output", type=Path)
    beam_parser = commands.add_parser("cantilever", help="solve the 1D beam and export its profile")
    beam_parser.add_argument("output", type=Path)
    beam_parser.add_argument("--root-height-mm", type=float, default=60)
    beam_parser.add_argument("--tip-ratio", type=float, default=0.75)
    beam_parser.add_argument("--elements", type=int, default=24)
    args = parser.parse_args()
    if args.command == "prepare":
        module = import_module(args.case)
        design = module.PROBLEM.reference_design() if args.design is None else args.design
        record = prepare(module, design, args.mesh_scale, args.output)
    elif args.command == "solve":
        record = solve(args.mesh, args.output, CASES)
    else:
        record = cantilever(args)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
