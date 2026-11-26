"""Render reference geometries for the engineering example documentation.

Run from the repository root with
uv run --locked --with matplotlib --with gmsh --with scikit-fem python examples/render_geometry.py
"""

import gzip
import re
import sys
from importlib import import_module
from pathlib import Path

import gmsh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"


def surface(ax, points, triangles, color, elevation=25, azimuth=-55):
    polygons = points[triangles]
    normal = np.cross(polygons[:, 1] - polygons[:, 0], polygons[:, 2] - polygons[:, 0])
    lengths = np.linalg.norm(normal, axis=1)
    light = np.array([0.4, -0.5, 0.75])
    intensity = 0.50 + 0.50 * np.abs(normal @ light) / np.maximum(lengths, 1e-12)
    rgb = np.array(matplotlib.colors.to_rgb(color))
    colors = np.clip(intensity[:, None] * rgb, 0, 1)
    ax.add_collection3d(Poly3DCollection(polygons, facecolors=colors, linewidths=0))
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    radius = (high - low).max() * 0.55
    ax.set(
        xlim=(center[0] - radius, center[0] + radius),
        ylim=(center[1] - radius, center[1] + radius),
        zlim=(center[2] - radius, center[2] + radius),
    )
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_axis_off()


def fem_geometry(module):
    design = module.PROBLEM.reference_design()
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        module.geometry(gmsh.model.occ, design)
        gmsh.model.occ.synchronize()
        bounds = np.array(gmsh.model.getBoundingBox(-1, -1))
        size = max(bounds[3:] - bounds[:3]) / 24
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / 3)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        gmsh.model.mesh.generate(2)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        rows = {int(tag): i for i, tag in enumerate(tags)}
        _, corners = gmsh.model.mesh.getElementsByType(2)
        triangles = np.array([rows[int(tag)] for tag in corners]).reshape(-1, 3)
        return coordinates.reshape(-1, 3), triangles
    finally:
        gmsh.finalize()


def render_fem():
    sys.path.insert(0, str(ROOT / "fem"))
    from common.cases import CASES

    modules = [import_module(name) for name in CASES]

    gallery = plt.figure(figsize=(16, 9), facecolor="white")
    for i, module in enumerate(modules):
        points, triangles = fem_geometry(module)
        title = module.PROBLEM.title
        for fig, ax in (
            (gallery, gallery.add_subplot(2, 5, i + 1, projection="3d")),
            (plt.figure(figsize=(7, 5)), None),
        ):
            if ax is None:
                ax = fig.add_subplot(projection="3d")
            surface(ax, points, triangles, "#3e88a5")
            ax.set_title(title, fontsize=13, pad=0)
            if fig is not gallery:
                fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92)
                fig.savefig(ASSETS / f"{module.PROBLEM.name}.png", dpi=160, facecolor="white")
                plt.close(fig)
        print(f"Rendered {module.PROBLEM.name}", flush=True)
    gallery.suptitle("FEM reference geometries", fontsize=22, y=0.98)
    gallery.text(
        0.5,
        0.025,
        "Reference dimensions · undeformed geometry · individual views use independent scales",
        ha="center",
        fontsize=12,
        color="#53616d",
    )
    gallery.subplots_adjust(left=0, right=1, bottom=0.06, top=0.90, wspace=0, hspace=0.08)
    gallery.savefig(ASSETS / "fem_reference_geometries.png", dpi=160, facecolor="white")
    plt.close(gallery)


def render_pump():
    mesh = ROOT / "cfd/pump_case/constant/polyMesh"

    def read(name):
        return gzip.decompress((mesh / f"{name}.gz").read_bytes()).decode()

    points = np.array(
        [
            [float(x) for x in row.split()]
            for row in re.findall(r"^\(([^()\n]+)\)$", read("points"), re.MULTILINE)
        ]
    )
    faces = [
        [int(x) for x in row.split()]
        for row in re.findall(r"^\d+\(([^()\n]+)\)$", read("faces"), re.MULTILINE)
    ]
    boundary = read("boundary")

    def patch(name):
        body = re.search(r"\b" + name + r"\s*\{([^}]+)\}", boundary)[1]
        start = int(re.search(r"startFace\s+(\d+)", body)[1])
        count = int(re.search(r"nFaces\s+(\d+)", body)[1])
        return np.array(
            [
                [face[0], face[i], face[i + 1]]
                for face in faces[start : start + count]
                for i in range(1, len(face) - 1)
            ]
        )

    fig = plt.figure(figsize=(12, 6))
    for i, (patch_name, title, color) in enumerate(
        (
            ("Wall_3", "Volute wall", "#3e88a5"),
            ("Wall_2", "Rotating impeller wall", "#d49b45"),
        )
    ):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        triangles = patch(patch_name)
        used, indices = np.unique(triangles, return_inverse=True)
        surface(ax, points[used], indices.reshape(-1, 3), color, elevation=15, azimuth=15)
        ax.set_title(title, fontsize=17)
    fig.suptitle("Centrifugal pump · bundled CFD mesh", fontsize=21)
    fig.text(
        0.5,
        0.035,
        "Surface geometry from the reference mesh · independent view scales",
        ha="center",
        fontsize=12,
        color="#53616d",
    )
    fig.subplots_adjust(left=0, right=1, bottom=0.08, top=0.85, wspace=0)
    fig.savefig(ASSETS / "centrifugal_pump.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    render_pump()
    render_fem()
