"""Typeset a ParaView scene and its optional VTK profile for publication."""

import json
import math
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter

DPI = 600
INK = "#232d37"


def scale_length(width):
    target = width * 0.22
    power = 10 ** math.floor(math.log10(target))
    return max(step * power for step in (1, 2, 5) if step * power <= target)


def scene(directory, metadata, width_mm, preview=False):
    image = plt.imread(directory / "scene.png")
    aspect = image.shape[1] / image.shape[0]
    width = width_mm / 25.4
    available = 0.92
    scene_height = width * available / aspect
    field = metadata["field"]
    footer = 0.55 if field else 0.29
    height = scene_height + footer
    fig = plt.figure(figsize=(width, height), facecolor="white")

    ax = fig.add_axes([(1 - available) / 2, footer / height, available, scene_height / height])
    # The render is larger than the print size. Filter before resampling to keep thin edges.
    ax.imshow(image, interpolation="lanczos", interpolation_stage="rgba")
    ax.axis("off")

    bar_width = metadata["view"]["width"]
    distance = scale_length(bar_width)
    scale_y = (footer - 0.12) / height
    start = 0.06
    end = start + available * distance / bar_width
    fig.add_artist(
        plt.Line2D(
            [start, end],
            [scale_y, scale_y],
            transform=fig.transFigure,
            lw=1.25,
            color=INK,
            solid_capstyle="butt",
        )
    )
    fig.text(
        (start + end) / 2,
        scale_y - 0.045 / height,
        f"{distance:g} {metadata['length_unit']}",
        ha="center",
        va="top",
    )
    if field:
        cmap = LinearSegmentedColormap.from_list("viridis", field["palette"])
        color_axis = fig.add_axes([0.36, (footer - 0.31) / height, 0.56, 0.065 / height])
        low, high = field["color_range"]
        data_low, data_high = field["data_range"]
        extend = (
            "both"
            if data_low < low and data_high > high
            else ("min" if data_low < low else "max" if data_high > high else "neither")
        )
        bar = fig.colorbar(
            plt.cm.ScalarMappable(norm=Normalize(low, high), cmap=cmap),
            cax=color_axis,
            orientation="horizontal",
            extend=extend,
        )
        bar.ax.set_title(field["label"], fontsize=8, pad=5)
        bar.locator = MaxNLocator(nbins=5, min_n_ticks=3)
        bar.formatter = ScalarFormatter(useMathText=True, useOffset=False)
        bar.update_ticks()
        bar.outline.set_linewidth(0.45)
        bar.ax.tick_params(length=2.5, width=0.55, pad=2.5)

    if preview:
        fig.savefig(directory / "preview.png", dpi=155)
    else:
        fig.savefig(directory / "field.png", dpi=DPI)
        fig.savefig(directory / "field.tiff", dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def profile(directory, metadata, width_mm, preview=False):
    rows = np.genfromtxt(directory / "profile.csv", delimiter=",", names=True)
    width = min(width_mm, 120) / 25.4
    fig, ax = plt.subplots(figsize=(width, width * 0.66), constrained_layout=True)
    values = np.where(rows["valid"], rows["value"], np.nan)
    ax.plot(rows["distance"], values, color="#156b82", lw=1.25)
    ax.set(
        xlabel=f"Distance along line ({metadata['length_unit']})", ylabel=metadata["field"]["label"]
    )
    ax.set_xlim(rows["distance"][0], rows["distance"][-1])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e4e9ed", linewidth=0.5)
    for extension in ("pdf", "svg"):
        fig.savefig(directory / f"profile.{extension}")
    if preview:
        fig.savefig(directory / "profile.png", dpi=205)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("figures", type=Path, help="directory from render.py")
    parser.add_argument("--width-mm", type=float, default=180, help="figure width in print")
    parser.add_argument("--preview", action="store_true", help="write a compact PNG scene")
    args = parser.parse_args()
    if not math.isfinite(args.width_mm) or args.width_mm < 85:
        parser.error("--width-mm must be at least 85 mm")
    metadata = json.loads((args.figures / "figure.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.6,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    scene(args.figures, metadata, args.width_mm, preview=args.preview)
    if metadata["profile"] is not None:
        profile(args.figures, metadata, args.width_mm, preview=args.preview)


if __name__ == "__main__":
    main()
