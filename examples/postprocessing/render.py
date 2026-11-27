"""Render a VTU mesh or scalar/vector field with pvpython and sample optional line data."""

import csv
import json
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np
from paraview import servermanager, simple
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkVersion
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm
from vtkmodules.vtkFiltersCore import vtkProbeFilter
from vtkmodules.vtkFiltersSources import vtkLineSource

# Sampled from viridis, interpolated in RGB in both ParaView and Matplotlib.
COLORS = (
    [0.267, 0.005, 0.329],
    [0.283, 0.141, 0.458],
    [0.254, 0.265, 0.530],
    [0.207, 0.372, 0.553],
    [0.164, 0.471, 0.558],
    [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518],
    [0.267, 0.749, 0.441],
    [0.478, 0.821, 0.318],
    [0.741, 0.873, 0.150],
    [0.993, 0.906, 0.144],
)
VIEW_SIZE = (1200, 700)


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length == 0:
        raise ValueError("directions must be finite and nonzero")
    return vector / length


def frame(view, bounds, direction, up):
    corners = np.array(list(product(*np.asarray(bounds).reshape(3, 2))))
    center = (corners.max(axis=0) + corners.min(axis=0)) / 2
    eye = unit(direction)
    up = unit(up if up is not None else ((0, 0, 1) if abs(eye[2]) < 0.9 else (0, 1, 0)))
    right = unit(np.cross(up, eye))
    vertical = np.cross(eye, right)
    width, height = np.ptp(corners @ right), np.ptp(corners @ vertical)
    scale = 0.56 * max(height, width * VIEW_SIZE[1] / VIEW_SIZE[0])
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = center.tolist()
    view.CameraPosition = (center + 3 * np.linalg.norm(np.ptp(corners, axis=0)) * eye).tolist()
    view.CameraViewUp = vertical.tolist()
    view.CameraParallelScale = scale
    return 2 * scale * VIEW_SIZE[0] / VIEW_SIZE[1]


def magnitude(values):
    if values.ndim == 1:
        return values
    if values.shape[1] != 3:
        raise ValueError("choose a scalar array or a three-component vector")
    return np.linalg.norm(values, axis=1)


def arrays(dataset, association):
    return dataset.GetPointData() if association == "POINTS" else dataset.GetCellData()


def field_values(dataset, association, field):
    data = arrays(dataset, association).GetArray(field)
    if data is None:
        raise ValueError(f"{field!r} is absent from {association.lower()} data")
    return magnitude(vtk_to_numpy(data))


def sample_profile(dataset, args):
    """Return the CSV rows and the summary of the field sampled along ``args.line``."""
    start, end = np.asarray(args.line).reshape(2, 3)
    length = float(np.linalg.norm(end - start))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("profile line must have finite, distinct endpoints")

    # Remove arrays from the opposite association on a shallow copy. Some VTU files
    # carry the same field name on both points and cells.
    volume = vtkUnstructuredGrid()
    volume.ShallowCopy(dataset)
    other = "CELLS" if args.association == "POINTS" else "POINTS"
    arrays(volume, other).RemoveArray(args.field)
    line = vtkLineSource()
    line.SetPoint1(*start)
    line.SetPoint2(*end)
    line.SetResolution(args.samples - 1)
    line.SetOutputPointsPrecision(vtkAlgorithm.DOUBLE_PRECISION)
    probe = vtkProbeFilter()
    probe.SetSourceData(volume)
    probe.SetInputConnection(line.GetOutputPort())
    probe.Update()
    data = probe.GetOutput()
    values = field_values(data, "POINTS", args.field)
    valid = vtk_to_numpy(data.GetPointData().GetArray("vtkValidPointMask")).astype(bool)
    valid &= np.isfinite(values)
    if not valid.any():
        raise ValueError("the profile line has no valid samples inside the volume")

    points = start + np.linspace(0, 1, args.samples)[:, None] * (end - start)
    rows = [
        [distance, *point, value if inside else "", int(inside)]
        for distance, point, value, inside in zip(
            np.linspace(0, length, args.samples), points, values, valid
        )
    ]
    return rows, {
        "endpoints": [start.tolist(), end.tolist()],
        "samples": args.samples,
        "valid_samples": int(valid.sum()),
    }


def render(args):
    simple._DisableFirstRenderCameraReset()
    source = simple.XMLUnstructuredGridReader(FileName=[str(args.input.resolve())])
    source.UpdatePipeline()
    volume = servermanager.Fetch(source)
    if volume.GetNumberOfCells() == 0:
        raise ValueError(f"{args.input} contains no cells")

    visible = source
    if args.slice:
        origin, normal = np.asarray(args.slice).reshape(2, 3)
        visible = simple.Slice(Input=source)
        visible.SliceType.Origin = origin.tolist()
        visible.SliceType.Normal = unit(normal).tolist()
        visible.UpdatePipeline()
    info = visible.GetDataInformation()
    if info.GetNumberOfCells() == 0:
        raise ValueError("the slice does not intersect the mesh")

    view = simple.CreateView("RenderView")
    layout = simple.CreateLayout(name="Field")
    simple.AssignViewToLayout(view=view, layout=layout, hint=0)
    layout.SetSize(*VIEW_SIZE)
    view.UseColorPaletteForBackground = 0
    view.Background = [1, 1, 1]
    view.OrientationAxesVisibility = 0
    display = simple.Show(visible, view)
    display.Ambient = 1
    display.Diffuse = 0
    direction = args.view or (args.slice[3:] if args.slice else (1, -1.5, 1.15))
    view_width = frame(view, info.GetBounds(), direction, args.up)

    stats = None
    if args.field:
        data = volume if visible is source else servermanager.Fetch(visible)
        values = field_values(data, args.association, args.field)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("visible field values must be finite and nonempty")
        limits = [float(values.min()), float(values.max())]
        low, high = args.range or limits
        if args.range is None and low == high:
            padding = max(abs(low), 1.0) * 0.01
            low, high = low - padding, high + padding
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError("--range must be two finite, increasing values")

        simple.ColorBy(display, (args.association, args.field, "Magnitude"))
        lut = simple.GetColorTransferFunction(args.field)
        lut.AutomaticRescaleRangeMode = "Never"
        lut.ColorSpace = "RGB"
        lut.RGBPoints = [
            value
            for x, rgb in zip(np.linspace(low, high, len(COLORS)), COLORS)
            for value in (float(x), *rgb)
        ]
        display.SetScalarBarVisibility(view, False)
        stats = {
            "name": args.field,
            "label": args.label or args.field,
            "association": args.association,
            "data_range": limits,
            "color_range": [low, high],
            "palette": COLORS,
        }
    else:
        display.SetScalarColoring(None, servermanager.GetAssociationFromString("POINTS"))
        display.Representation = "Surface With Edges"
        display.DiffuseColor = [0.84, 0.89, 0.93]
        display.AmbientColor = display.DiffuseColor
        display.EdgeColor = [0.19, 0.26, 0.31]

    # A beam axis is a line grid, which is invisible in print at the default one-pixel width.
    if volume.GetCell(0).GetCellDimension() == 1:
        display.RenderLinesAsTubes = 1
        display.LineWidth = 24

    # Sample before creating the output, so a rejected line leaves no directory behind.
    rows, profile_info = sample_profile(volume, args) if args.line else (None, None)
    args.output.mkdir(parents=True, exist_ok=False)
    if rows is not None:
        with (args.output / "profile.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["distance", "x", "y", "z", "value", "valid"])
            writer.writerows(rows)
    simple.Render(view)
    simple.SaveScreenshot(
        str(args.output / "scene.png"),
        view,
        ImageResolution=[dimension * 4 for dimension in VIEW_SIZE],
    )
    simple.SaveState(str(args.output / "scene.pvsm"))
    return {
        "input": str(args.input.resolve()),
        "length_unit": args.length_unit,
        "field": stats,
        "versions": {
            "paraview": str(simple.GetParaViewVersion()),
            "vtk": vtkVersion.GetVTKVersion(),
        },
        "geometry": {
            "points": volume.GetNumberOfPoints(),
            "cells": volume.GetNumberOfCells(),
            "bounds": list(volume.GetBounds()),
        },
        "view": {
            "width": view_width,
            "direction": list(direction),
            "up": list(view.CameraViewUp),
            "slice": args.slice,
        },
        "profile": profile_info,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="VTU file")
    parser.add_argument("output", type=Path, help="new figure directory")
    parser.add_argument("--field", help="VTU array name; omit for a mesh plot")
    parser.add_argument("--association", choices=("POINTS", "CELLS"), default="POINTS")
    parser.add_argument("--label", help="field label including units")
    parser.add_argument("--length-unit", required=True, help="native coordinate unit, e.g. mm or m")
    parser.add_argument("--slice", type=float, nargs=6, metavar="N", help="origin then normal")
    parser.add_argument("--view", type=float, nargs=3, metavar="N", help="direction toward camera")
    parser.add_argument("--up", type=float, nargs=3, metavar="N", help="camera up direction")
    parser.add_argument("--range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--line", type=float, nargs=6, metavar="N", help="profile start and end")
    parser.add_argument("--samples", type=int, default=401)
    args = parser.parse_args()
    if args.input.suffix.lower() != ".vtu" or not args.input.is_file():
        parser.error("input must be an existing VTU file")
    if (args.line or args.range or args.label) and not args.field:
        parser.error("--line, --range and --label require --field")
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    metadata = render(args)
    (args.output / "figure.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
