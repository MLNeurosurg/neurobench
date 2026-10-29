"""Plot report win, tie, and loss percentages with question coverage."""

import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_comparison(summary_rows, path: Path, *, unit='studies'):
    """Save report preference percentages and question coverage as a PNG."""
    n_complete = summary_rows[0]["n_complete"]
    n_excluded = summary_rows[0]["n_excluded"]
    n_total = summary_rows[0]['n_questions']
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.55 * len(summary_rows) + 1.8)))
    if not n_complete:
        ax.text(
            0.5,
            0.5,
            f"No retained {unit}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        for index, row in enumerate(summary_rows):
            counts = [row[field] for field in ("candidate_1_wins", "ties", "candidate_2_wins")]
            start = 0
            for count, color, label in zip(
                counts,
                ("#3679ad", "#c8c8c8", "#da9142"),
                ("Left model wins", "Tie", "Right model wins"),
            ):
                width = 100 * count / n_complete
                ax.barh(
                    index,
                    width,
                    left=start,
                    color=color,
                    label=label if index == 0 else None,
                )
                if width >= 8:
                    ax.text(
                        start + width / 2,
                        index,
                        f"{width:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )
                start += width
        ax.set_yticks(
            range(len(summary_rows)),
            [f"{row['candidate_1_model']} vs {row['candidate_2_model']}" for row in summary_rows],
        )
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"Percentage of retained {unit}")
    ax.set_title(
        f"Report generation comparison\n{n_complete:,}/{n_total:,} {unit} retained; {n_excluded:,} excluded"
    )
    # Reserve a footer for the legend so it cannot overlap the x-axis label.
    fig.tight_layout(rect=(0, 0.22 if n_complete else 0, 1, 1))
    if n_complete:
        position = ax.get_position()
        fig.legend(
            *ax.get_legend_handles_labels(),
            loc="lower center",
            bbox_to_anchor=((position.x0 + position.x1) / 2, 0.01),
            ncol=3,
            frameon=False,
        )
    fd, temporary = tempfile.mkstemp(
        prefix=".results.", suffix=".png", dir=path.parent
    )
    os.close(fd)
    try:
        fig.savefig(temporary, dpi=180, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        plt.close(fig)
        Path(temporary).unlink(missing_ok=True)
