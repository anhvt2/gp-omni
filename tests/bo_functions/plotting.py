"""Reusable file-based plots for campaign objective and average-regret histories.

Requires Matplotlib, which is optional for running the numerical tests.
"""

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def plot_campaign_curves(
    runs,
    output_stem: str | Path,
    *,
    labels: dict[str, str] | None = None,
    symlog_cases=(),
    show_initialization: bool = False,
) -> tuple[Path, Path]:
    """Save paired best-objective / average-regret panels as PNG and SVG.

    Group records by ``case``. Each record supplies ``dimension``, ``global_min``,
    ``n_init``, ``n_evals`` and four histories: ``objective_history``,
    ``random_objective_history``, ``average_regret_history`` and
    ``random_average_regret_history``. Runs of a case must share a budget.
    Plot medians and observed ranges across seeds without smoothing the data.
    ``output_stem`` is a path without a file extension; parent directories are created.
    """
    groups = {}
    for record in runs:
        groups.setdefault(record["case"], []).append(record)
    if not groups:
        raise ValueError("at least one campaign is required")
    labels = labels or {}
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    max_evals = max(r["n_evals"] for records in groups.values() for r in records)
    with matplotlib.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "gp-omni-bo-campaigns",
        }
    ):
        fig = Figure(figsize=(12, max(3, len(groups) * 25 / 9)), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(len(groups), 2, squeeze=False)
        for (case, records), (objective_ax, regret_ax) in zip(groups.items(), axes, strict=True):
            n = records[0]["n_evals"]
            dimension = records[0]["dimension"]
            minimum = records[0]["global_min"]
            if any(r["n_evals"] != n for r in records):
                raise ValueError(f"{case}: all seeds must share an evaluation budget")
            x = np.arange(1, n + 1)
            for ax, keys in (
                (objective_ax, ("objective_history", "random_objective_history")),
                (regret_ax, ("average_regret_history", "random_average_regret_history")),
            ):
                for key, color, style in zip(
                    keys, ("#225ea8", "#777777"), ("-", "--"), strict=True
                ):
                    values = np.asarray([r[key] for r in records])
                    if values.shape != (len(records), n) or not np.isfinite(values).all():
                        raise ValueError(f"{case}: invalid {key} histories")
                    if ax is objective_ax:
                        ax.step(
                            x,
                            np.median(values, axis=0),
                            where="post",
                            color=color,
                            linestyle=style,
                            linewidth=1.8,
                        )
                        ax.fill_between(
                            x, values.min(0), values.max(0), step="post", color=color, alpha=0.13
                        )
                    else:
                        ax.plot(
                            x,
                            np.median(values, axis=0),
                            color=color,
                            linestyle=style,
                            linewidth=1.8,
                        )
                        ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=0.13)
                if show_initialization:
                    ax.axvline(records[0]["n_init"], color="#8856a7", linestyle=":", linewidth=1)
                ax.set_xlim(0, max_evals)
                ax.set_xlabel("Objective evaluations")
                ax.grid(alpha=0.18)
            objective_ax.axhline(minimum, color="#202020", linestyle=":", linewidth=1.2)
            scale = ""
            if case in symlog_cases:
                objective_ax.set_yscale("symlog", linthresh=1)
                upper = max(
                    max(r[key])
                    for r in records
                    for key in ("objective_history", "random_objective_history")
                )
                objective_ax.set_ylim(min(-0.1, 1.1 * minimum), 1.1 * upper)
                scale = "; symlog y"
            name = labels.get(case, case)
            objective_ax.set_title(f"{name} ({dimension}D; n={n}{scale})", fontsize=10)
            objective_ax.set_ylabel("Best objective value")
            regret_ax.set_title(f"{name} ({dimension}D): average regret", fontsize=10)
            regret_ax.set_ylabel("Average regret (R(T)/T)")
            regret_ax.set_ylim(bottom=0)
            regret_ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
        handles = [
            Line2D([0], [0], color="#225ea8", lw=2, label="EI Bayesian optimization"),
            Line2D([0], [0], color="#777777", lw=2, ls="--", label="Uniform random search"),
            Line2D([0], [0], color="#202020", lw=1.5, ls=":", label="Known global minimum"),
        ]
        if show_initialization:
            handles.append(
                Line2D([0], [0], color="#8856a7", lw=1, ls=":", label="Initial design complete")
            )
        fig.legend(
            handles=handles,
            loc="outside upper center",
            ncol=2 if show_initialization else 3,
            frameon=False,
        )
        outputs = []
        for extension in ("png", "svg"):
            path = output_stem.with_suffix(f".{extension}")
            fig.savefig(path, dpi=160, metadata={"Date": None} if extension == "svg" else None)
            if extension == "svg":
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
                )
            outputs.append(path)
        fig.clear()
    return tuple(outputs)
