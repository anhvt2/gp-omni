# Bundled centrifugal pump case

This directory contains all mesh and field inputs needed by
`../centrifugal_pump.py`. OpenFOAM reads the gzip-compressed mesh files directly.
The solver run requires this repository and OpenCFD OpenFOAM v2406.

## Provenance and license

The source is 20jeka08's FreeCAD_OpenFOAM_Volute_Pump, revision
`0f01c42f339f952f89d01cab5795f3666366737d`.
The original GPL-3.0 license is included in [LICENSE](LICENSE). Original file
headers and attribution are preserved. These case assets retain that license.

The fields and dictionaries come from `volute/{0,constant,system}`.
The mesh comes from `transient_case/constant/polyMesh`. Mesh payloads are
unchanged and compressed with gzip with a fixed timestamp. The driver replaces
controlDict in each generated case to configure measurements and iteration count,
sets rotor speed, and scales a fresh copy of the mesh for each design.

The original repository location is recorded for attribution.
[Original source](https://github.com/20jeka08/FreeCAD_OpenFOAM_Volute_Pump/tree/0f01c42f339f952f89d01cab5795f3666366737d)

## Integrity and reproduction

`SHA256SUMS.json` records SHA-256 hashes for all 25 bundled input and license
files. The driver checks file names and hashes before a run. Verify the bundled
inputs offline from the repository root with the following command.

```bash
uv run --locked python examples/cfd/centrifugal_pump.py --verify-only
```

Validation used OpenCFD OpenFOAM package `2406.241212-3`, double precision,
32-bit labels, on Linux ARM64. Baseline and variant solves reached 800 SIMPLE
iterations, including a restart at iteration 500. The final ten measurements
gave the following reference values.

| Mesh scale | Speed magnitude rad/s | Static pressure rise Pa | Shaft power W | Outlet flow m³/s |
| --- | --- | --- | --- | --- |
| 1.0 | 314.0 | 1040771.96 | 53327.37 | 0.04263863 |
| 1.0274 | 292.3026 | 949478.35 | 49028.11 | 0.04263828 |
| 1.0213 | 335.6032 | 1286269.80 | 67616.05 | 0.04263808 |

These values document the 800-iteration validation. The default campaign uses
3,000 iterations per design. Mesh resolution, platform arithmetic, and solver
patch releases can affect results. The driver checks the final ten measurement
samples for stability and requires both patch flows within 3% of the prescribed
0.0416666667 m³/s. Each campaign writes the input hashes, solver version, seed,
iteration count, and duty settings to `run.json`.
