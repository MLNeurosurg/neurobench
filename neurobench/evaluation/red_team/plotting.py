"""Plot red-team summary scores, study coverage, and illustrative question templates."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import tempfile
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patches import Patch
from matplotlib.text import Annotation


@dataclass(frozen=True)
class TemplateTable:
    """Describe prompt-template rows and column widths for a red-team figure."""

    headers: tuple[str, ...]
    widths: tuple[float, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class PlotRow:
    """Describe the conditions, endpoint panels, and template table for one red-team plot row."""

    name: str
    conditions: tuple[str, ...]
    positions: tuple[float, ...]
    hatches: tuple[str, ...]
    endpoints: tuple[tuple[str, str], ...]
    templates: TemplateTable
    heading: str = ''
    subgroups: tuple[tuple[float, str], ...] = ()
    table_gap: float = 1.05
    table_height: float = 1.6


@dataclass(frozen=True)
class FigureSpec:
    """Describe the titles, plot rows, legend, and styling of a red-team summary figure."""

    title: str
    subtitle: str
    rows: tuple[PlotRow, ...]
    legend: tuple[tuple[str, str], ...]
    ylabel: str
    min_width: float = 12
    value_labels_max_models: int = 2
    footnote: str = ''
    score_note: str = ''


# Inches keep visible bar thickness independent of condition count and panel size.
BAR_WIDTH_IN = .35
MODEL_GAP_IN = .65
CONDITION_GAP_IN = .08
EDGE_PADDING_IN = .12


def template_text(ax, x, y, text, bold=False):
    """Draw prompt-template text with emphasis on the rubric-defining phrases."""
    lines = []
    for line in text.split('\n'):
        pieces = []
        for part in re.split(r'(\*\*.*?\*\*)', line):
            if not part:
                continue
            highlighted = part.startswith('**')
            props = dict(fontsize=9, weight='bold' if bold or highlighted else 'normal')
            if highlighted:
                props['bbox'] = dict(facecolor='#fff1b8', edgecolor='none', pad=1)
            pieces.append(TextArea(part[2:-2] if highlighted else part, textprops=props))
        lines.append(HPacker(children=pieces, align='baseline', pad=0, sep=0))
    box = VPacker(children=lines, align='left', pad=0, sep=4)
    ax.add_artist(AnnotationBbox(box, (x, y), xycoords='data', box_alignment=(0, .5),
                                frameon=False, pad=0, annotation_clip=False))


def draw_templates(ax, table):
    """Draw a question-template table alongside the red-team summary plots."""
    headers, widths, rows = table.headers, table.widths, table.rows
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(rows) + .65)
    positions = [.012]
    for width in widths[:-1]:
        positions.append(positions[-1] + width)
    for x, header in zip(positions, headers):
        template_text(ax, x, len(rows) + .32, header, bold=True)
    ax.hlines(len(rows), 0, 1, color='#aaaaaa', linewidth=.8)
    for i, row in enumerate(rows):
        top = len(rows) - i
        ax.axhspan(top - 1, top, color='#f4f4f6' if i % 2 == 0 else 'white')
        ax.hlines(top - 1, 0, 1, color='#dddddd', linewidth=.6)
        for column, (x, value) in enumerate(zip(positions, row)):
            template_text(ax, x, top - .5, value, bold=column == 0)


def make_figure(summary, models, spec):
    """Build red-team plots and template tables, checking that panels report the same study coverage."""
    rows = spec.rows
    if not rows or not models:
        raise ValueError('Figures require at least one plot row and model')
    for row in rows:
        if not row.conditions or not row.endpoints or not (len(row.conditions) == len(row.positions) == len(row.hatches)):
            raise ValueError('Plot conditions, positions, and hatches must have matching nonzero lengths')
        if any(b <= a for a, b in zip(row.positions, row.positions[1:])):
            raise ValueError('Condition positions must be strictly increasing')
    lookup = {(row['model'], row['condition'], row['endpoint']): row for row in summary}
    if len(lookup) != len(summary):
        raise ValueError('Figure summary contains duplicate model/condition/endpoint rows')
    coverage = {(int(row['n_complete']), int(row['n_excluded'])) for row in summary}
    if len(coverage) != 1:
        raise ValueError('Figure panels must use the same scored-study cohort')
    complete, excluded = coverage.pop()
    top = 1.45
    chart_height, row_gap, footer = 2.7, .7, .65
    row_heights = [(.80 if row.heading else 0) + chart_height + row.table_gap + row.table_height
                   for row in rows]
    height = top + sum(row_heights) + row_gap * (len(rows) - 1) + footer
    left, right, panel_gap = .065, .985, .27
    condition_step = BAR_WIDTH_IN + CONDITION_GAP_IN
    figure_width = spec.min_width
    for row in rows:
        group_width = (row.positions[-1] - row.positions[0]) * condition_step + BAR_WIDTH_IN
        model_slot = max(group_width + MODEL_GAP_IN, 1.8)
        required_panel_width = len(models) * model_slot + 2 * EDGE_PADDING_IN
        panel_fraction = (right - left) / (len(row.endpoints) + panel_gap * (len(row.endpoints) - 1))
        figure_width = max(figure_width, required_panel_width / panel_fraction)
    fig = plt.figure(figsize=(figure_width, height), facecolor='white')
    # Colors follow --candidate_answer order, separated by hue before shade.
    color_order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    colors = {model: plt.get_cmap('tab20')(color_order[i % 20]) for i, model in enumerate(models)}
    row_top = top
    for row, row_height in zip(rows, row_heights):
        conditions, positions, hatches = row.conditions, row.positions, row.hatches
        endpoints = row.endpoints
        span = positions[-1] - positions[0]
        group_width = span * condition_step + BAR_WIDTH_IN
        heading_height = .80 if row.heading else 0
        if row.heading:
            fig.text(.5, 1 - row_top / height, row.heading,
                     fontsize=12, weight='bold', ha='center', va='top')
        chart_top = row_top + heading_height
        grid = fig.add_gridspec(1, len(endpoints), left=left, right=right,
                                top=1 - chart_top / height,
                                bottom=1 - (chart_top + chart_height) / height, wspace=panel_gap)
        for column, (endpoint, title) in enumerate(endpoints):
            ax = fig.add_subplot(grid[0, column])
            ax.set_label(f'{row.name}:{endpoint}')
            panel_width = ax.get_position().width * figure_width
            # Use inches as x units so bars keep the same visible width as models are added.
            ax.set_xlim(0, panel_width)
            slot_width = (panel_width - 2 * EDGE_PADDING_IN) / len(models)
            model_centers = [EDGE_PADDING_IN + (i + .5) * slot_width for i in range(len(models))]
            for i, model in enumerate(models):
                values = []
                for condition in conditions:
                    value = lookup[model, condition, endpoint]['score']
                    value = None if value == '' else float(value)
                    if complete and (value is None or not math.isfinite(value) or not 0 <= value <= 1):
                        raise ValueError('Figure scores must be finite proportions on a complete cohort')
                    values.append(value)
                if not complete:
                    continue
                centers = [model_centers[i] + (x - positions[0] - span / 2) * condition_step
                           for x in positions]
                bars = ax.bar(centers, [100 * v for v in values], width=BAR_WIDTH_IN,
                              color=colors[model], edgecolor='white', linewidth=.7, zorder=3)
                for bar, hatch in zip(bars, hatches):
                    bar.set_hatch(hatch)
                if len(models) <= spec.value_labels_max_models:
                    ax.bar_label(bars, labels=[f'{100 * v:.1f}' for v in values],
                                 padding=3, fontsize=8 if len(endpoints) > 1 and len(models) > 1 else 9)
            if not complete:
                ax.text(.5, .5, 'No scored studies', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=11, weight='bold', pad=12)
            model_labels = [textwrap.fill(model, 22) for model in models]
            ax.set_xticks(model_centers, model_labels, fontsize=11, weight='bold')
            ax.tick_params(axis='x', length=0, pad=10)
            if row.subgroups:
                # Optional subgroup labels sit above a bracket identifying each model's bars.
                ax.set_xticklabels([])
                for center, model_label in zip(model_centers, model_labels):
                    for position, label in row.subgroups:
                        x = center + (position - positions[0] - span / 2) * condition_step
                        ax.text(x, -.055, label, transform=ax.get_xaxis_transform(),
                                ha='center', va='top', fontsize=9)
                    ax.plot([center - group_width / 2, center + group_width / 2], [-.21, -.21],
                            transform=ax.get_xaxis_transform(), color='#888888',
                            linewidth=.7, clip_on=False)
                    ax.text(center, -.25, model_label, transform=ax.get_xaxis_transform(),
                            ha='center', va='top', fontsize=11, weight='bold')
            ax.set_ylim(0, 112)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis='y', color='#e4e4e4', linewidth=.7, zorder=0)
            ax.spines[['top', 'right']].set_visible(False)
            if column == 0:
                ax.set_ylabel(spec.ylabel, fontsize=10)
            else:
                ax.tick_params(axis='y', labelleft=False)
        template_axis = fig.add_axes([.065, 1 - (row_top + row_height) / height,
                                      .92, row.table_height / height], label=f'templates:{row.name}')
        draw_templates(template_axis, row.templates)
        row_top += row_height + row_gap
    fig.suptitle(spec.title, fontsize=16, weight='bold', y=1 - .17 / height)
    fig.text(.5, 1 - .55 / height, spec.subtitle,
             ha='center', fontsize=11, color='#555555')
    legend_conditions = spec.legend
    fig.legend(handles=[Patch(facecolor='#777777', edgecolor='white', hatch=hatch,
                              label=label.replace('\n', ' ')) for label, hatch in legend_conditions],
               loc='upper center', bbox_to_anchor=(.5, 1 - .73 / height),
               ncol=len(legend_conditions), frameon=False, fontsize=10,
               handleheight=1.5, handlelength=2.2)
    note = f'{complete:,}/{complete + excluded:,} studies retained · {excluded:,} excluded'
    if spec.score_note:
        note += ' · ' + spec.score_note
    if spec.footnote:
        note += '\n' + spec.footnote
    fig.text(.5, .15 / height, note, ha='center', va='bottom', fontsize=9, color='#555555')
    # Similar condition scores can put value labels side by side at the same height.
    # Move only colliding labels upward, preserving every displayed estimate.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        occupied = []
        for label in sorted((t for t in ax.texts if isinstance(t, Annotation)), key=lambda t: t.xy[0]):
            box = label.get_window_extent(renderer).expanded(1.08, 1.1)
            while any(box.overlaps(other) for other in occupied):
                x, y = label.get_position()
                label.set_position((x, y + 3))
                box = label.get_window_extent(renderer).expanded(1.08, 1.1)
            occupied.append(box)
    return fig


def plot_summary(summary, models, spec, output_dir):
    """Save red-team results.png atomically and close the completed figure."""
    fig = make_figure(summary, models, spec)
    destination = Path(output_dir) / 'results.png'
    fd, name = tempfile.mkstemp(prefix='.results.', suffix='.png', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        fig.savefig(temporary, dpi=180, bbox_inches='tight', facecolor='white')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)
