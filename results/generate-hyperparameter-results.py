"""
Most of this script was generated using ChatGPT 5.6 Sol Model
to enhance the quality of the output.

Reproduce the figures and uncertainty summaries used in the sweep report.

Run from the repository root with:

    uv sync --extra notebooks
    .venv/bin/python results/generate-hyperparameter-results.py

Each figure is written as SVG for the Markdown report and as vector PDF for
LaTeX.
"""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "docs" / "assets"

SFT = "#334155"
DMPO = "#2563eb"
DEPO = "#059669"
MUTED = "#94a3b8"
GRID = "#dbe3ec"
TEXT = "#172033"


def _svg_number(value: str | None, extent: float | None = None) -> float:
    """Parse the simple SVG lengths emitted by this script."""
    if value is None:
        return 0.0
    if value.endswith("%"):
        if extent is None:
            raise ValueError(f"Cannot resolve SVG percentage without an extent: {value}")
        return float(value[:-1]) * extent / 100
    return float(value.removesuffix("px"))


def svg_to_pdf(svg: str, destination: Path) -> None:
    """Render this script's flat SVG primitives to a vector PDF with Matplotlib."""
    try:
        import matplotlib

        matplotlib.use("pdf")
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
    except ImportError as error:
        raise RuntimeError(
            "PDF export requires Matplotlib. Run `uv sync --extra notebooks` first."
        ) from error

    root = ElementTree.fromstring(svg)
    min_x, min_y, width, height = map(float, root.attrib["viewBox"].split())
    title = next(
        (element.text for element in root if element.tag.rsplit("}", 1)[-1] == "title"),
        destination.stem,
    )

    points_per_svg_unit = 72 / 96
    with matplotlib.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=(width / 96, height / 96), frameon=False)
        axes = figure.add_axes((0, 0, 1, 1))
        axes.set_xlim(min_x, min_x + width)
        axes.set_ylim(min_y + height, min_y)
        axes.set_axis_off()

        for element in root:
            tag = element.tag.rsplit("}", 1)[-1]
            attributes = element.attrib
            if tag in {"title", "desc"}:
                continue

            fill = attributes.get("fill", "none")
            face_colour = "none" if fill == "none" else fill
            stroke = attributes.get("stroke", "none")
            edge_colour = "none" if stroke == "none" else stroke
            opacity = float(attributes.get("fill-opacity", "1"))
            line_width = _svg_number(attributes.get("stroke-width", "1"))

            if tag == "rect":
                x = _svg_number(attributes.get("x"), width)
                y = _svg_number(attributes.get("y"), height)
                rect_width = _svg_number(attributes.get("width"), width)
                rect_height = _svg_number(attributes.get("height"), height)
                radius = _svg_number(attributes.get("rx"))
                common = {
                    "facecolor": face_colour,
                    "edgecolor": edge_colour,
                    "linewidth": line_width,
                    "alpha": opacity,
                }
                if radius:
                    patch = FancyBboxPatch(
                        (x, y),
                        rect_width,
                        rect_height,
                        boxstyle=f"round,pad=0,rounding_size={radius}",
                        **common,
                    )
                else:
                    patch = Rectangle((x, y), rect_width, rect_height, **common)
                axes.add_patch(patch)
            elif tag == "circle":
                axes.add_patch(
                    Circle(
                        (_svg_number(attributes["cx"]), _svg_number(attributes["cy"])),
                        _svg_number(attributes["r"]),
                        facecolor=face_colour,
                        edgecolor=edge_colour,
                        linewidth=line_width,
                        alpha=opacity,
                    )
                )
            elif tag == "line":
                dash = attributes.get("stroke-dasharray")
                linestyle: str | tuple[int, tuple[float, ...]] = "-"
                if dash:
                    linestyle = (0, tuple(float(item) for item in dash.replace(",", " ").split()))
                axes.add_line(
                    Line2D(
                        [_svg_number(attributes["x1"]), _svg_number(attributes["x2"])],
                        [_svg_number(attributes["y1"]), _svg_number(attributes["y2"])],
                        color=edge_colour,
                        linewidth=line_width,
                        linestyle=linestyle,
                    )
                )
            elif tag == "text":
                anchor = attributes.get("text-anchor", "start")
                horizontal_alignment = {"start": "left", "middle": "center", "end": "right"}[anchor]
                rotation = 0.0
                transform = attributes.get("transform", "")
                if transform.startswith("rotate("):
                    rotation = -float(transform[7:-1].split()[0])
                font_family = attributes.get("font-family", "Arial").split(",", 1)[0]
                axes.text(
                    _svg_number(attributes["x"]),
                    _svg_number(attributes["y"]),
                    element.text or "",
                    color=face_colour,
                    fontsize=_svg_number(attributes.get("font-size", "12")) * points_per_svg_unit,
                    fontfamily=font_family,
                    fontweight=attributes.get("font-weight", "normal"),
                    horizontalalignment=horizontal_alignment,
                    verticalalignment="baseline",
                    rotation=rotation,
                    rotation_mode="anchor",
                )
            else:
                raise ValueError(f"Unsupported SVG element in generated figure: {tag}")

        figure.savefig(
            destination,
            format="pdf",
            bbox_inches=None,
            pad_inches=0,
            metadata={"Title": title or destination.stem},
        )
        plt.close(figure)


def write_figure(filename: str, svg: str) -> None:
    """Write matching SVG and vector-PDF versions of a generated figure."""
    svg_path = ASSETS / f"{filename}.svg"
    svg_path.write_text(svg)
    svg_to_pdf(svg, ASSETS / f"{filename}.pdf")


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    half_width = (
        z * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
    )
    return centre - half_width, centre + half_width


def exact_mcnemar(gained: int, lost: int) -> float:
    discordant = gained + lost
    tail = min(gained, lost)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant
    return min(1.0, 2 * probability)


def read_attempts(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: row["instance_id"])
    token_field = "total_tokens" if "total_tokens" in rows[0] else "total_tokens_total"
    ids = [row["instance_id"] for row in rows]
    tokens = np.asarray([float(row[token_field]) for row in rows])
    resolved = np.asarray([row["resolved"].lower() == "true" for row in rows], dtype=float)
    return ids, tokens, resolved


def read_attempt_rows(path: Path) -> dict[str, dict[str, object]]:
    """Read either SWE-smith or SWE-bench analysis rows into a common schema."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    token_field = "total_tokens" if "total_tokens" in rows[0] else "total_tokens_total"
    completion_field = (
        "completion_tokens" if "completion_tokens" in rows[0] else "completion_tokens_total"
    )
    step_field = "model_api_calls" if "model_api_calls" in rows[0] else "agent_action_steps"
    normalised = {}
    for row in rows:
        failure_category = row.get("failure_category") or row.get("evaluation_status") or "unknown"
        normalised[row["instance_id"]] = {
            "instance_id": row["instance_id"],
            "repo": row["repo"].removeprefix("swesmith/"),
            "resolved": row["resolved"].lower() == "true",
            "patch_present": row.get("patch_present", "").lower() == "true",
            "total_tokens": float(row[token_field]),
            "completion_tokens": float(row[completion_field]),
            "steps": float(row[step_field]),
            "evaluation_status": row.get("evaluation_status") or "unknown",
            "failure_category": failure_category,
        }
    return normalised


def paired_breakdown(baseline_path: Path, candidate_path: Path) -> dict[str, object]:
    baseline = read_attempt_rows(baseline_path)
    candidate = read_attempt_rows(candidate_path)
    if baseline.keys() != candidate.keys():
        raise ValueError("Breakdown inputs do not use the same task matrix")

    records = []
    for instance_id in sorted(baseline):
        base = baseline[instance_id]
        cand = candidate[instance_id]
        if base["resolved"] and cand["resolved"]:
            transition = "both_resolved"
        elif cand["resolved"]:
            transition = "gained"
        elif base["resolved"]:
            transition = "lost"
        else:
            transition = "both_unresolved"
        records.append(
            {
                "instance_id": instance_id,
                "repo": base["repo"],
                "transition": transition,
                "baseline_resolved": base["resolved"],
                "candidate_resolved": cand["resolved"],
                "baseline_tokens": base["total_tokens"],
                "candidate_tokens": cand["total_tokens"],
                "token_delta": cand["total_tokens"] - base["total_tokens"],
                "step_delta": cand["steps"] - base["steps"],
                "baseline_status": base["evaluation_status"],
                "baseline_failure": base["failure_category"],
                "candidate_status": cand["evaluation_status"],
                "candidate_failure": cand["failure_category"],
                "candidate_patch_present": cand["patch_present"],
            }
        )

    def summarise(group: list[dict[str, object]]) -> dict[str, float | int]:
        deltas = [float(row["token_delta"]) for row in group]
        return {
            "tasks": len(group),
            "baseline_resolved": sum(bool(row["baseline_resolved"]) for row in group),
            "candidate_resolved": sum(bool(row["candidate_resolved"]) for row in group),
            "baseline_tokens": sum(float(row["baseline_tokens"]) for row in group),
            "candidate_tokens": sum(float(row["candidate_tokens"]) for row in group),
            "token_delta": sum(deltas),
            "mean_delta": statistics.fmean(deltas) if deltas else 0.0,
            "median_delta": statistics.median(deltas) if deltas else 0.0,
            "candidate_cheaper": sum(delta < 0 for delta in deltas),
            "candidate_costlier": sum(delta > 0 for delta in deltas),
        }

    transitions = {
        name: summarise([row for row in records if row["transition"] == name])
        for name in ["both_resolved", "gained", "lost", "both_unresolved"]
    }
    repos = {
        repo: summarise([row for row in records if row["repo"] == repo])
        for repo in sorted({str(row["repo"]) for row in records})
    }
    failures = {
        failure: summarise([row for row in records if row["candidate_failure"] == failure])
        for failure in sorted({str(row["candidate_failure"]) for row in records})
    }
    attempt_groups = {
        "no_patch": summarise([row for row in records if not bool(row["candidate_patch_present"])])
    }
    deltas = np.asarray([float(row["token_delta"]) for row in records])
    positive = sorted((float(value) for value in deltas if value > 0), reverse=True)
    negative = sorted((-float(value) for value in deltas if value < 0), reverse=True)
    return {
        "records": records,
        "overall": summarise(records),
        "transitions": transitions,
        "repos": repos,
        "failures": failures,
        "attempt_groups": attempt_groups,
        "delta_quantiles": {
            str(quantile): float(np.quantile(deltas, quantile))
            for quantile in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        },
        "positive_spend": sum(positive),
        "negative_savings": sum(negative),
        "top_5_burn_share": sum(positive[:5]) / sum(positive),
        "top_5_saving_share": sum(negative[:5]) / sum(negative),
        "largest_burns": sorted(records, key=lambda row: float(row["token_delta"]), reverse=True)[
            :10
        ],
        "largest_savings": sorted(records, key=lambda row: float(row["token_delta"]))[:10],
    }


def bootstrap_cost_delta(
    baseline_path: Path,
    candidate_path: Path,
    *,
    seed: int = 42,
    replicates: int = 20_000,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    baseline_ids, baseline_tokens, baseline_resolved = read_attempts(baseline_path)
    candidate_ids, candidate_tokens, candidate_resolved = read_attempts(candidate_path)
    if baseline_ids != candidate_ids:
        raise ValueError("Bootstrap inputs do not use the same task matrix")

    rng = np.random.default_rng(seed)
    cost_deltas: list[np.ndarray] = []
    token_deltas: list[np.ndarray] = []
    resolution_deltas: list[np.ndarray] = []
    n = len(baseline_ids)
    for offset in range(0, replicates, 1_000):
        batch = min(1_000, replicates - offset)
        indices = rng.integers(0, n, size=(batch, n), dtype=np.int32)
        base_token_sum = baseline_tokens[indices].sum(axis=1)
        candidate_token_sum = candidate_tokens[indices].sum(axis=1)
        base_successes = baseline_resolved[indices].sum(axis=1)
        candidate_successes = candidate_resolved[indices].sum(axis=1)
        cost_deltas.append(
            100
            * ((candidate_token_sum / candidate_successes) / (base_token_sum / base_successes) - 1)
        )
        token_deltas.append(100 * (candidate_token_sum / base_token_sum - 1))
        resolution_deltas.append(100 * (candidate_successes - base_successes) / n)

    cost = np.concatenate(cost_deltas)
    tokens = np.concatenate(token_deltas)
    resolution = np.concatenate(resolution_deltas)
    return (
        tuple(np.percentile(cost, [2.5, 97.5])),
        tuple(np.percentile(tokens, [2.5, 97.5])),
        tuple(np.percentile(resolution, [2.5, 97.5])),
    )


def svg_text(x: float, y: float, value: str, **attributes: object) -> str:
    attrs = {"x": x, "y": y, "fill": TEXT, "font-family": "Arial, sans-serif", **attributes}
    rendered = " ".join(
        f'{key.replace("_", "-")}="{html.escape(str(item))}"' for key, item in attrs.items()
    )
    return f"<text {rendered}>{html.escape(value)}</text>"


def screening_figure(screen: dict) -> str:
    labels = {
        "sft": "SFT",
        "dmpo-g07": "DMPO γ=0.7",
        "dmpo-g09": "DMPO γ=0.9",
        "dmpo-g05": "DMPO γ=0.5",
        "depo-paper": "DEPO paper",
        "depo-completion": "DEPO completion",
        "depo-total": "DEPO total",
    }
    colours = {
        name: (SFT if name == "sft" else DMPO if name.startswith("dmpo") else DEPO)
        for name in labels
    }
    selected = {"sft", "dmpo-g07", "depo-total"}
    offsets = {
        "sft": (10, -10),
        "dmpo-g07": (10, -12),
        "dmpo-g09": (10, 20),
        "dmpo-g05": (10, -10),
        "depo-paper": (-118, -10),
        "depo-completion": (-148, -10),
        "depo-total": (10, 20),
    }

    width, height = 980, 560
    left, right, top, bottom = 95, 35, 75, 80
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max = 4.5, 8.0
    y_min, y_max = 6.5, 12.0

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">SWE-smith screening trade-off</title>',
        '<desc id="desc">Resolution rate against total tokens per resolved task for the seven screened arms.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(left, 34, "200-task SWE-smith screening", font_size=22, font_weight=700),
        svg_text(
            left,
            57,
            "Preferred direction is upper-left; filled points mark SFT and the promoted candidates.",
            font_size=13,
            fill="#536176",
        ),
    ]

    for tick in [4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]:
        x = x_position(tick)
        elements.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" stroke="{GRID}"/>'
        )
        elements.append(
            svg_text(x, height - bottom + 25, f"{tick:g}", font_size=12, text_anchor="middle")
        )
    for tick in [7, 8, 9, 10, 11, 12]:
        y = y_position(tick)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        elements.append(svg_text(left - 12, y + 4, f"{tick}%", font_size=12, text_anchor="end"))

    threshold_y = y_position(8.5)
    elements.append(
        f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{width - right}" y2="{threshold_y:.1f}" '
        f'stroke="#b45309" stroke-width="2" stroke-dasharray="7 6"/>'
    )
    elements.append(
        svg_text(
            width - right - 5,
            threshold_y - 8,
            "non-inferiority screen: 8.5%",
            font_size=12,
            fill="#92400e",
            text_anchor="end",
        )
    )

    for arm in screen["arms"]:
        name = arm["name"]
        cost = arm["efficiency"]["total_tokens_per_resolved_task"] / 1_000_000
        rate = 100 * arm["efficiency"]["resolution_rate"]
        x, y = x_position(cost), y_position(rate)
        colour = colours[name]
        opacity = 1 if name in selected else 0.55
        radius = 8 if name in selected else 6
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{colour}" fill-opacity="{opacity}" '
            f'stroke="white" stroke-width="2"/>'
        )
        dx, dy = offsets[name]
        elements.append(
            svg_text(
                x + dx,
                y + dy,
                labels[name],
                font_size=12,
                font_weight=700 if name in selected else 400,
                fill=colour,
            )
        )

    elements.extend(
        [
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="{TEXT}"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="{TEXT}"/>',
            svg_text(
                left + plot_width / 2,
                height - 22,
                "Total tokens per resolved task (millions; lower is better)",
                font_size=14,
                text_anchor="middle",
            ),
            f'<text x="24" y="{top + plot_height / 2}" fill="{TEXT}" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" transform="rotate(-90 24 {top + plot_height / 2})">Resolution rate (higher is better)</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


def final_figure(confirmation: dict, verified: dict) -> str:
    width, height = 1080, 700
    colours = {"sft": SFT, "dmpo": DMPO, "depo": DEPO}
    labels = {"sft": "SFT", "dmpo": "DMPO", "depo": "DMPO→DEPO"}
    positions = [130, 230, 330, 650, 750, 850]
    combined = [("SWE-smith", arm) for arm in confirmation["arms"]] + [
        ("SWE-bench Verified", arm) for arm in verified["arms"]
    ]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Confirmatory and external-test performance</title>',
        '<desc id="desc">Resolution rates with Wilson intervals and total tokens per resolved task for SFT, DMPO and sequential DEPO.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(75, 34, "500-task evaluation", font_size=22, font_weight=700),
        svg_text(
            75,
            58,
            "Error bars are 95% Wilson intervals; cost includes all evaluated attempts.",
            font_size=13,
            fill="#536176",
        ),
        svg_text(75, 92, "A. Resolution rate", font_size=17, font_weight=700),
        svg_text(
            970,
            92,
            "orange dashed: -3 pp tolerance",
            font_size=12,
            fill="#92400e",
            text_anchor="end",
        ),
        svg_text(75, 405, "B. Total tokens per resolved task", font_size=17, font_weight=700),
    ]

    chart_left, chart_right = 75, 970
    top_a, bottom_a = 110, 350
    top_b, bottom_b = 425, 625
    for tick in [0, 10, 20, 30, 40]:
        y = bottom_a - tick / 45 * (bottom_a - top_a)
        elements.append(
            f'<line x1="{chart_left}" y1="{y:.1f}" x2="{chart_right}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        elements.append(
            svg_text(chart_left - 10, y + 4, f"{tick}%", font_size=12, text_anchor="end")
        )
    for tick in range(0, 8, 1):
        y = bottom_b - tick / 7 * (bottom_b - top_b)
        elements.append(
            f'<line x1="{chart_left}" y1="{y:.1f}" x2="{chart_right}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        elements.append(
            svg_text(chart_left - 10, y + 4, f"{tick}M", font_size=12, text_anchor="end")
        )

    # The pre-specified three-percentage-point point-estimate thresholds.
    for x1, x2, rate in [(85, 385, 11.0), (605, 905, 36.2)]:
        y = bottom_a - rate / 45 * (bottom_a - top_a)
        elements.append(
            f'<line x1="{x1}" y1="{y:.1f}" x2="{x2}" y2="{y:.1f}" stroke="#b45309" stroke-width="1.5" stroke-dasharray="6 5"/>'
        )

    for x, (benchmark, arm) in zip(positions, combined):
        name = arm["name"]
        rate = 100 * arm["efficiency"]["resolution_rate"]
        successes = arm["efficiency"]["resolved_trajectories"]
        trials = arm["efficiency"]["trajectories"]
        low, high = (100 * value for value in wilson(successes, trials))
        y = bottom_a - rate / 45 * (bottom_a - top_a)
        y_low = bottom_a - low / 45 * (bottom_a - top_a)
        y_high = bottom_a - high / 45 * (bottom_a - top_a)
        colour = colours[name]
        elements.extend(
            [
                f'<line x1="{x}" y1="{y_high:.1f}" x2="{x}" y2="{y_low:.1f}" stroke="{colour}" stroke-width="2"/>',
                f'<line x1="{x - 6}" y1="{y_high:.1f}" x2="{x + 6}" y2="{y_high:.1f}" stroke="{colour}" stroke-width="2"/>',
                f'<line x1="{x - 6}" y1="{y_low:.1f}" x2="{x + 6}" y2="{y_low:.1f}" stroke="{colour}" stroke-width="2"/>',
                f'<circle cx="{x}" cy="{y:.1f}" r="7" fill="{colour}"/>',
                svg_text(
                    x,
                    y_high - 8,
                    f"{rate:.1f}%",
                    font_size=12,
                    text_anchor="middle",
                    font_weight=700,
                    fill=colour,
                ),
            ]
        )

        cost = arm["efficiency"]["total_tokens_per_resolved_task"] / 1_000_000
        bar_top = bottom_b - cost / 7 * (bottom_b - top_b)
        elements.append(
            f'<rect x="{x - 28}" y="{bar_top:.1f}" width="56" height="{bottom_b - bar_top:.1f}" rx="3" fill="{colour}"/>'
        )
        elements.append(
            svg_text(
                x,
                bar_top - 9,
                f"{cost:.2f}M",
                font_size=12,
                text_anchor="middle",
                font_weight=700,
                fill=colour,
            )
        )
        elements.append(
            svg_text(x, 650, labels[name], font_size=12, text_anchor="middle", fill=colour)
        )

    elements.extend(
        [
            svg_text(
                230,
                380,
                "SWE-smith confirmation",
                font_size=14,
                text_anchor="middle",
                font_weight=700,
            ),
            svg_text(
                750, 380, "SWE-bench Verified", font_size=14, text_anchor="middle", font_weight=700
            ),
            svg_text(
                230,
                680,
                "SWE-smith confirmation",
                font_size=14,
                text_anchor="middle",
                font_weight=700,
            ),
            svg_text(
                750, 680, "SWE-bench Verified", font_size=14, text_anchor="middle", font_weight=700
            ),
            "</svg>",
        ]
    )
    return "\n".join(elements)


def token_delta_decomposition_figure(smith: dict, verified: dict) -> str:
    """Show which paired outcomes and failure modes drive aggregate token changes."""
    width, height = 1080, 740
    order = ["both_resolved", "gained", "lost", "both_unresolved"]
    labels = {
        "both_resolved": "Both solved",
        "gained": "DEPO-only",
        "lost": "SFT-only",
        "both_unresolved": "Neither",
    }
    group_positions = [[150, 250, 350, 450], [650, 750, 850, 950]]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Drivers of the DEPO token difference</title>',
        '<desc id="desc">Aggregate paired token changes by resolution transition and the token share of no-patch attempts in both evaluations.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(80, 34, "What drives DEPO's token difference?", font_size=22, font_weight=700),
        svg_text(
            80,
            59,
            "All changes are DEPO minus SFT; green is a saving and orange is additional spend.",
            font_size=13,
            fill="#536176",
        ),
        svg_text(
            80,
            92,
            "A. Aggregate total-token change by paired outcome",
            font_size=17,
            font_weight=700,
        ),
    ]

    left, right, top, bottom = 80, 1020, 110, 330
    y_min, y_max = -25.0, 25.0

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (bottom - top)

    for tick in [-20, -10, 0, 10, 20]:
        y = y_position(tick)
        colour = TEXT if tick == 0 else GRID
        width_px = 1.5 if tick == 0 else 1
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            f'stroke="{colour}" stroke-width="{width_px}"/>'
        )
        elements.append(svg_text(left - 10, y + 4, f"{tick:+d}M", font_size=11, text_anchor="end"))

    zero = y_position(0)
    for positions, breakdown in zip(group_positions, [smith, verified]):
        for x, name in zip(positions, order):
            value = breakdown["transitions"][name]["token_delta"] / 1_000_000
            y = y_position(value)
            colour = DEPO if value < 0 else "#c25b22"
            rect_y = min(y, zero)
            rect_height = abs(y - zero)
            elements.append(
                f'<rect x="{x - 29}" y="{rect_y:.1f}" width="58" height="{rect_height:.1f}" '
                f'rx="3" fill="{colour}"/>'
            )
            label_y = y - 8 if value >= 0 else y + 18
            elements.append(
                svg_text(
                    x,
                    label_y,
                    f"{value:+.2f}M",
                    font_size=11,
                    text_anchor="middle",
                    font_weight=700,
                    fill=colour,
                )
            )
            elements.append(svg_text(x, 352, labels[name], font_size=11, text_anchor="middle"))

    elements.extend(
        [
            svg_text(
                300,
                382,
                "SWE-smith confirmation",
                font_size=14,
                text_anchor="middle",
                font_weight=700,
            ),
            svg_text(
                800, 382, "SWE-bench Verified", font_size=14, text_anchor="middle", font_weight=700
            ),
            svg_text(
                80,
                425,
                "B. No-patch attempts consume a disproportionate token share",
                font_size=17,
                font_weight=700,
            ),
            svg_text(
                80,
                448,
                "Bars compare their share of tasks with their share of all DEPO tokens.",
                font_size=13,
                fill="#536176",
            ),
        ]
    )

    no_patch_groups = []
    for benchmark, breakdown, positions in [
        ("SWE-smith confirmation", smith, [245, 345]),
        ("SWE-bench Verified", verified, [705, 805]),
    ]:
        overall = breakdown["overall"]
        values = breakdown["attempt_groups"]["no_patch"]
        task_share = 100 * values["tasks"] / overall["tasks"]
        token_share = 100 * values["candidate_tokens"] / overall["candidate_tokens"]
        no_patch_groups.append(
            {
                "benchmark": benchmark,
                "positions": positions,
                "task_share": task_share,
                "token_share": token_share,
                "delta": values["token_delta"] / 1_000_000,
            }
        )

    chart_top, chart_bottom = 475, 640

    def share_y(value: float) -> float:
        return chart_bottom - value / 25 * (chart_bottom - chart_top)

    for tick in [0, 5, 10, 15, 20, 25]:
        y = share_y(tick)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        elements.append(svg_text(left - 10, y + 4, f"{tick}%", font_size=11, text_anchor="end"))

    for group in no_patch_groups:
        for x, value, label, colour in [
            (group["positions"][0], group["task_share"], "Task share", MUTED),
            (group["positions"][1], group["token_share"], "Token share", "#c25b22"),
        ]:
            y = share_y(float(value))
            elements.append(
                f'<rect x="{x - 31}" y="{y:.1f}" width="62" height="{chart_bottom - y:.1f}" '
                f'rx="3" fill="{colour}"/>'
            )
            elements.append(
                svg_text(
                    x,
                    y - 8,
                    f"{value:.1f}%",
                    font_size=11,
                    text_anchor="middle",
                    font_weight=700,
                    fill=colour,
                )
            )
            elements.append(
                svg_text(x, 662, label, font_size=11, text_anchor="middle", fill=colour)
            )
        centre = sum(group["positions"]) / 2
        elements.append(
            svg_text(
                centre,
                696,
                str(group["benchmark"]),
                font_size=14,
                text_anchor="middle",
                font_weight=700,
            )
        )
        elements.append(
            svg_text(
                centre,
                719,
                f"{group['delta']:+.2f}M vs SFT",
                font_size=12,
                text_anchor="middle",
                fill="#92400e",
            )
        )

    elements.append("</svg>")
    return "\n".join(elements)


def paired_token_distribution_figure(smith: dict, verified: dict) -> str:
    """Plot rank-ordered paired token differences on a common scale."""
    width, height = 1080, 570
    chart_top, chart_bottom = 110, 445
    y_min, y_max = -5.5, 5.5
    panels = [
        ("A. SWE-smith confirmation", smith, 80, 510),
        ("B. SWE-bench Verified", verified, 570, 1000),
    ]

    def y_position(value: float) -> float:
        return chart_top + (y_max - value) / (y_max - y_min) * (chart_bottom - chart_top)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Distribution of paired task-level token changes</title>',
        '<desc id="desc">Rank-ordered DEPO minus SFT total-token differences for SWE-smith and SWE-bench Verified on a common vertical scale.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(
            70,
            34,
            "Distribution of paired task-level token changes",
            font_size=22,
            font_weight=700,
        ),
        svg_text(
            70,
            58,
            "Green indicates a DEPO saving; orange indicates additional spend. Panels share the same scale.",
            font_size=13,
            fill="#536176",
        ),
    ]

    for panel_index, (title, breakdown, left, right) in enumerate(panels):
        panel_width = right - left
        elements.append(svg_text(left, 91, title, font_size=17, font_weight=700))

        for tick in [-5, -2.5, 0, 2.5, 5]:
            y = y_position(tick)
            colour = TEXT if tick == 0 else GRID
            width_px = 1.5 if tick == 0 else 1
            elements.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                f'stroke="{colour}" stroke-width="{width_px}"/>'
            )
            if panel_index == 0:
                tick_label = f"{tick:+g}M" if tick else "0M"
                elements.append(
                    svg_text(left - 10, y + 4, tick_label, font_size=11, text_anchor="end")
                )

        for percentile in [0, 25, 50, 75, 100]:
            x = left + percentile / 100 * panel_width
            elements.append(
                f'<line x1="{x:.1f}" y1="{chart_top}" x2="{x:.1f}" y2="{chart_bottom}" '
                f'stroke="{GRID}"/>'
            )
            elements.append(
                svg_text(
                    x,
                    chart_bottom + 22,
                    f"{percentile}%",
                    font_size=10,
                    text_anchor="middle",
                )
            )

        deltas = sorted(float(row["token_delta"]) / 1_000_000 for row in breakdown["records"])
        for index, value in enumerate(deltas):
            x = left + index / (len(deltas) - 1) * panel_width
            y = y_position(value)
            colour = DEPO if value < 0 else "#c25b22" if value > 0 else MUTED
            elements.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="{colour}" fill-opacity="0.72"/>'
            )

        overall = breakdown["overall"]
        centre = (left + right) / 2
        elements.extend(
            [
                svg_text(
                    centre,
                    493,
                    f"{overall['candidate_cheaper']} cheaper | {overall['candidate_costlier']} costlier",
                    font_size=12,
                    text_anchor="middle",
                    font_weight=700,
                ),
                svg_text(
                    centre,
                    514,
                    f"median {overall['median_delta'] / 1_000:+.1f}K | net {overall['token_delta'] / 1_000_000:+.2f}M",
                    font_size=12,
                    text_anchor="middle",
                    fill="#536176",
                ),
                svg_text(
                    centre,
                    548,
                    "Ordered task percentile",
                    font_size=13,
                    text_anchor="middle",
                ),
            ]
        )

    elements.extend(
        [
            f'<text x="24" y="{(chart_top + chart_bottom) / 2}" fill="{TEXT}" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" transform="rotate(-90 24 {(chart_top + chart_bottom) / 2})">DEPO - SFT total tokens (millions)</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


def repository_effects_figure(verified: dict) -> str:
    """Plot repository contributions to aggregate SWE-bench Verified changes."""
    width, height = 1080, 650
    left, right, top, bottom = 115, 1005, 105, 555
    x_min, x_max = -16.0, 12.5
    y_min, y_max = -5.5, 4.5

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (bottom - top)

    zero_x, zero_y = x_position(0), y_position(0)
    short_names = {
        "astropy/astropy": "astropy",
        "django/django": "django",
        "matplotlib/matplotlib": "matplotlib",
        "mwaskom/seaborn": "seaborn",
        "pallets/flask": "flask",
        "psf/requests": "requests",
        "pydata/xarray": "xarray",
        "pylint-dev/pylint": "pylint",
        "pytest-dev/pytest": "pytest",
        "scikit-learn/scikit-learn": "scikit-learn",
        "sphinx-doc/sphinx": "sphinx",
        "sympy/sympy": "sympy",
    }
    offsets = {
        "astropy": (10, 18, "start"),
        "django": (10, 18, "start"),
        "flask": (10, 18, "start"),
        "matplotlib": (10, -12, "start"),
        "pylint": (10, -12, "start"),
        "pytest": (-10, 18, "end"),
        "requests": (10, -12, "start"),
        "scikit-learn": (10, 20, "start"),
        "seaborn": (-10, 20, "end"),
        "sphinx": (10, 18, "start"),
        "sympy": (10, -12, "start"),
        "xarray": (-10, -12, "end"),
    }
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">SWE-bench Verified repository contributions</title>',
        '<desc id="desc">Repository-level resolution and aggregate token changes for DEPO relative to SFT.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(
            80,
            35,
            "Repository contributions to the aggregate SWE-bench Verified result",
            font_size=22,
            font_weight=700,
        ),
        svg_text(
            80,
            61,
            "Upper-left improves both outcomes; bubble area is proportional to the number of tasks.",
            font_size=13,
            fill="#536176",
        ),
        f'<rect x="{left}" y="{top}" width="{zero_x - left:.1f}" height="{zero_y - top:.1f}" fill="#eaf7f1"/>',
        f'<rect x="{zero_x:.1f}" y="{zero_y:.1f}" width="{right - zero_x:.1f}" height="{bottom - zero_y:.1f}" fill="#fff1e8"/>',
    ]

    for tick in [-15, -10, -5, 0, 5, 10]:
        x = x_position(tick)
        colour = TEXT if tick == 0 else GRID
        width_px = 1.5 if tick == 0 else 1
        elements.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{colour}" stroke-width="{width_px}"/>'
        )
        elements.append(svg_text(x, bottom + 24, f"{tick:+d}M", font_size=11, text_anchor="middle"))
    for tick in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]:
        y = y_position(tick)
        colour = TEXT if tick == 0 else GRID
        width_px = 1.5 if tick == 0 else 1
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{colour}" stroke-width="{width_px}"/>'
        )
        elements.append(svg_text(left - 10, y + 4, f"{tick:+d}", font_size=11, text_anchor="end"))

    elements.extend(
        [
            svg_text(
                left + 10,
                top + 22,
                "higher resolution + lower token use",
                font_size=12,
                fill="#047857",
                font_weight=700,
            ),
            svg_text(
                right - 10,
                bottom - 12,
                "lower resolution + higher token use",
                font_size=12,
                fill="#b45309",
                font_weight=700,
                text_anchor="end",
            ),
        ]
    )

    for repo, values in verified["repos"].items():
        token_delta = values["token_delta"] / 1_000_000
        resolution_delta = values["candidate_resolved"] - values["baseline_resolved"]
        x, y = x_position(token_delta), y_position(resolution_delta)
        if token_delta < 0 and resolution_delta > 0:
            colour = DEPO
        elif token_delta > 0:
            colour = "#c25b22"
        else:
            colour = DMPO
        radius = 4.5 + math.sqrt(values["tasks"]) * 0.45
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{colour}" '
            f'fill-opacity="0.88" stroke="white" stroke-width="2"/>'
        )
        short = short_names[repo]
        dx, dy, anchor = offsets[short]
        elements.append(
            svg_text(
                x + dx,
                y + dy,
                f"{short} ({values['tasks']})",
                font_size=11,
                text_anchor=anchor,
                font_weight=700,
                fill=colour,
            )
        )

    elements.extend(
        [
            svg_text(
                (left + right) / 2,
                624,
                "Aggregate token change (millions; negative = saving)",
                font_size=14,
                text_anchor="middle",
            ),
            f'<text x="27" y="{(top + bottom) / 2}" fill="{TEXT}" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" transform="rotate(-90 27 {(top + bottom) / 2})">Resolution change (tasks; higher is better)</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


def main() -> None:
    screen_path = RESULTS / "swesmith_screening_200" / "comparison.json"
    confirmation_path = RESULTS / "swesmith_confirmation_500" / "comparison.json"
    verified_path = RESULTS / "swebench_verified_500" / "comparison.json"
    screen = load_json(screen_path)
    confirmation = load_json(confirmation_path)
    verified = load_json(verified_path)

    ASSETS.mkdir(exist_ok=True)
    write_figure("hyperparameter-screening", screening_figure(screen))
    write_figure("hyperparameter-final-results", final_figure(confirmation, verified))

    run_paths = {
        "SWE-smith": {
            "sft": RESULTS / "swesmith_confirmation_500" / "sft.csv",
            "dmpo": RESULTS / "swesmith_confirmation_500" / "dmpo.csv",
            "depo": RESULTS / "swesmith_confirmation_500" / "depo.csv",
        },
        "SWE-bench Verified": {
            "sft": RESULTS / "swebench_verified_500" / "sft.csv",
            "dmpo": RESULTS / "swebench_verified_500" / "dmpo.csv",
            "depo": RESULTS / "swebench_verified_500" / "depo.csv",
        },
    }
    detailed = {
        benchmark: paired_breakdown(paths["sft"], paths["depo"])
        for benchmark, paths in run_paths.items()
    }
    write_figure(
        "hyperparameter-token-delta-decomposition",
        token_delta_decomposition_figure(detailed["SWE-smith"], detailed["SWE-bench Verified"]),
    )
    write_figure(
        "hyperparameter-paired-token-distribution",
        paired_token_distribution_figure(detailed["SWE-smith"], detailed["SWE-bench Verified"]),
    )
    write_figure(
        "hyperparameter-repository-effects",
        repository_effects_figure(detailed["SWE-bench Verified"]),
    )

    comparisons = {"SWE-smith": confirmation, "SWE-bench Verified": verified}
    for benchmark, comparison in comparisons.items():
        print(benchmark)
        for arm in comparison["arms"]:
            successes = arm["efficiency"]["resolved_trajectories"]
            trials = arm["efficiency"]["trajectories"]
            low, high = wilson(successes, trials)
            print(f"  {arm['name']}: Wilson 95% CI = {100 * low:.2f}% to {100 * high:.2f}%")
            if arm["name"] != "sft":
                transitions = arm["resolution_transitions_vs_baseline"]
                p_value = exact_mcnemar(transitions["gained"], transitions["lost"])
                cost_ci, tokens_ci, resolution_ci = bootstrap_cost_delta(
                    run_paths[benchmark]["sft"], run_paths[benchmark][arm["name"]]
                )
                print(
                    f"    McNemar p={p_value:.4f}; bootstrap cost delta 95% CI "
                    f"[{cost_ci[0]:.1f}%, {cost_ci[1]:.1f}%]; total-token delta "
                    f"[{tokens_ci[0]:.1f}%, {tokens_ci[1]:.1f}%]; resolution delta "
                    f"[{resolution_ci[0]:.1f}, {resolution_ci[1]:.1f}] pp"
                )
        breakdown = detailed[benchmark]
        overall = breakdown["overall"]
        print(
            f"  DEPO detail: net token delta={overall['token_delta'] / 1_000_000:.2f}M; "
            f"median task delta={overall['median_delta'] / 1_000:.1f}K; "
            f"cheaper/costlier={overall['candidate_cheaper']}/{overall['candidate_costlier']}; "
            f"gross spend/saving={breakdown['positive_spend'] / 1_000_000:.2f}M/"
            f"{breakdown['negative_savings'] / 1_000_000:.2f}M; "
            f"top-five burn share={100 * breakdown['top_5_burn_share']:.1f}%; "
            f"top-five saving share={100 * breakdown['top_5_saving_share']:.1f}%"
        )
        print(
            "  Token-delta quantiles (candidate-SFT): "
            + ", ".join(
                f"q{100 * float(quantile):g}={value / 1_000:.1f}K"
                for quantile, value in breakdown["delta_quantiles"].items()
            )
        )
        print("  By transition:")
        for name, values in breakdown["transitions"].items():
            print(
                f"    {name}: n={values['tasks']}, delta={values['token_delta'] / 1_000_000:.2f}M, "
                f"mean={values['mean_delta'] / 1_000:.1f}K, median={values['median_delta'] / 1_000:.1f}K, "
                f"cheaper/costlier={values['candidate_cheaper']}/{values['candidate_costlier']}"
            )
        print("  By repository (largest burns first):")
        for name, values in sorted(
            breakdown["repos"].items(), key=lambda item: item[1]["token_delta"], reverse=True
        ):
            resolution_delta = values["candidate_resolved"] - values["baseline_resolved"]
            print(
                f"    {name}: n={values['tasks']}, resolved "
                f"DEPO/SFT={values['candidate_resolved']}/{values['baseline_resolved']} "
                f"({resolution_delta:+d}), "
                f"delta={values['token_delta'] / 1_000_000:+.2f}M"
            )
        print("  By DEPO failure category:")
        for name, values in sorted(
            breakdown["failures"].items(), key=lambda item: item[1]["token_delta"], reverse=True
        ):
            print(
                f"    {name}: n={values['tasks']}, "
                f"tokens DEPO/SFT={values['candidate_tokens'] / 1_000_000:.2f}M/"
                f"{values['baseline_tokens'] / 1_000_000:.2f}M, "
                f"delta={values['token_delta'] / 1_000_000:+.2f}M, "
                f"mean={values['mean_delta'] / 1_000:+.1f}K"
            )
        for failure in ["context_limit", "empty_patch"]:
            affected = [row for row in breakdown["records"] if row["candidate_failure"] == failure]
            if affected:
                baseline_failures = Counter(str(row["baseline_failure"]) for row in affected)
                transitions = Counter(str(row["transition"]) for row in affected)
                print(
                    f"    {failure} cross-check: SFT outcomes={dict(baseline_failures)}, "
                    f"transitions={dict(transitions)}"
                )
        print("  Largest burns:")
        for row in breakdown["largest_burns"][:5]:
            print(
                f"    {row['instance_id']}: {row['token_delta'] / 1_000_000:+.2f}M, "
                f"{row['transition']}, {row['candidate_failure']}"
            )
        print("  Largest savings:")
        for row in breakdown["largest_savings"][:5]:
            print(
                f"    {row['instance_id']}: {row['token_delta'] / 1_000_000:+.2f}M, "
                f"{row['transition']}, {row['candidate_failure']}"
            )


if __name__ == "__main__":
    main()
