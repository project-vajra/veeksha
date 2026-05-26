"""Unified sweep plotter for concurrency and input-size benchmark runs.

Examples
--------
    python scripts/plot_sweep.py --sweep-type conc --plot-type sweep \\
        --engine1-dir benchmark_output/vajra_qwen3tts_new_server_async \\
        --engine2-dir benchmark_output/vajra_qwen3tts_async_ttfa_fix \\
        --engine1-name Vajra --engine2-name vLLM

    python scripts/plot_sweep.py --sweep-type input --plot-type report \\
        --engine1-dir benchmark_output/vajra_qwen3tts_new_server_input_async \\
        --engine2-dir benchmark_output/vllm_omni_sweep_input
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib")
)

import pandas as pd
import rekha as rk
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import (  # type: ignore[import-not-found]  # noqa: E402
    DEFAULT_NARADA_DIR,
    InputRun,
    PERCENTILES,
    REPORT_METRICS,
    Run,
    archive_selected_sweep_runs,
    build_concurrency_df,
    build_input_df,
    collect_input_runs,
    collect_runs,
    filter_complete_axis,
    pick_best_input_rtf,
    pick_best_rtf,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark_output" / "sweep_plots"
SINGLE_CONC_METRICS: Tuple[Tuple[str, str, str, bool], ...] = (
    ("ttfa", "TTFA", "ms", True),
    ("rtf", "RTF", "ratio", True),
)
SINGLE_CONC_PERCENTILES = ("P50", "P90", "P99")

LIGHT_COLORS = {
    "value_text": "#5C6166",
    "delta_box_face": "white",
    "delta_win": "#86B300",
    "delta_lose": "#FA8D3E",
    "missing": "#F07171",
}
DARK_COLORS = {
    "value_text": "#BFBDB6",
    "delta_box_face": "#1F2430",
    "delta_win": "#BAE67E",
    "delta_lose": "#F28779",
    "missing": "#F28779",
}
DEFAULT_ENGINE1_COLOR = "#73D0FF"
DEFAULT_ENGINE2_COLOR = "#FFA759"


def _theme_colors(dark_mode: bool) -> Dict[str, str]:
    return DARK_COLORS if dark_mode else LIGHT_COLORS


def _normalize_sweep_type(raw: str) -> str:
    value = raw.lower()
    if value in {"conc", "concurrency"}:
        return "conc"
    if value == "input":
        return "input"
    raise argparse.ArgumentTypeError("sweep type must be conc/concurrency or input")


def _axis_col(sweep_type: str) -> str:
    return "concurrency" if sweep_type == "conc" else "input_chars"


MetricSpec = Tuple[str, str, str, bool]
SelectedRun = Run | InputRun


def _axis_label(sweep_type: str) -> str:
    return "Concurrent Sessions" if sweep_type == "conc" else "Input Length (chars)"


def _sort_selected_runs(
    selected: Dict[Tuple[str, int], SelectedRun],
) -> List[SelectedRun]:
    return [run for _, run in sorted(selected.items(), key=lambda item: item[0])]


def _bar_is_missing(rect: Rectangle) -> bool:
    height = rect.get_height()
    try:
        return not (height is not None and height > 0 and not pd.isna(height))
    except TypeError:
        return True


def _format_delta(winner: float, loser: float, delta_style: str) -> str:
    if winner <= 0:
        return ""
    if delta_style == "multiplier":
        return f"{loser / winner:.2f}x"
    return f"Delta {abs(loser - winner) / winner * 100.0:.1f}%"


def _annotate_bar_axes(
    ax,
    value_fmt: str,
    *,
    lower_is_better: bool,
    dark_mode: bool,
    compare_pairs: bool,
    delta_style: str = "pct",
    show_value_labels: bool = True,
) -> None:
    theme = _theme_colors(dark_mode)
    bars = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    if not bars:
        return
    bars_sorted = sorted(bars, key=lambda rect: rect.get_x() + rect.get_width() / 2)
    valid_heights = [rect.get_height() for rect in bars_sorted if not _bar_is_missing(rect)]
    if not valid_heights:
        return
    y_max = max(valid_heights)
    y_pad = y_max * 0.015

    for rect in bars_sorted:
        if _bar_is_missing(rect):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                y_max * 0.04,
                "X",
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="bold",
                color=theme["missing"],
            )
            continue
        if show_value_labels:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + y_pad,
                value_fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=8,
                color=theme["value_text"],
            )

    if compare_pairs:
        for i in range(0, len(bars_sorted) - 1, 2):
            left, right = bars_sorted[i], bars_sorted[i + 1]
            if _bar_is_missing(left) or _bar_is_missing(right):
                continue
            left_h, right_h = left.get_height(), right.get_height()
            winner = min(left_h, right_h) if lower_is_better else max(left_h, right_h)
            loser = max(left_h, right_h) if lower_is_better else min(left_h, right_h)
            label = _format_delta(winner, loser, delta_style)
            if not label:
                continue
            winner_is_left = left_h < right_h if lower_is_better else left_h > right_h
            color = theme["delta_win"] if winner_is_left else theme["delta_lose"]
            mid_x = (
                left.get_x() + left.get_width() / 2
                + right.get_x() + right.get_width() / 2
            ) / 2
            top_y = max(left_h, right_h) + y_pad * 6
            ax.text(
                mid_x,
                top_y,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=color,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor=theme["delta_box_face"],
                    edgecolor=color,
                    linewidth=0.8,
                    alpha=0.85,
                ),
            )
    ax.set_ylim(top=y_max * 1.28)


def _annotate_line_axes(
    ax,
    sub: pd.DataFrame,
    axis_col: str,
    value_col: str,
    value_fmt: str,
    *,
    lower_is_better: bool,
    dark_mode: bool,
    compare_pairs: bool,
    delta_style: str = "pct",
) -> None:
    if sub.empty or not isinstance(sub[axis_col].dtype, pd.CategoricalDtype):
        return
    theme = _theme_colors(dark_mode)
    axis_values = list(sub[axis_col].cat.categories)
    systems = (
        list(sub["system"].cat.categories)
        if isinstance(sub["system"].dtype, pd.CategoricalDtype)
        else list(sub["system"].unique())
    )
    lookup: Dict[Tuple[str, str], float] = {}
    for _, row in sub.iterrows():
        lookup[(str(row["system"]), str(row[axis_col]))] = float(row[value_col])

    valid = [value for value in lookup.values() if not pd.isna(value) and value > 0]
    if not valid:
        return
    y_max = max(valid)
    y_pad = y_max * 0.018

    for idx, axis_value in enumerate(axis_values):
        for system in systems:
            value = lookup.get((str(system), str(axis_value)), float("nan"))
            if pd.isna(value) or value <= 0:
                continue
            ax.text(
                idx,
                value + y_pad,
                value_fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=7,
                color=theme["value_text"],
            )

    if compare_pairs and len(systems) >= 2:
        first, second = str(systems[0]), str(systems[1])
        for idx, axis_value in enumerate(axis_values):
            first_v = lookup.get((first, str(axis_value)), float("nan"))
            second_v = lookup.get((second, str(axis_value)), float("nan"))
            if pd.isna(first_v) or pd.isna(second_v) or first_v <= 0 or second_v <= 0:
                continue
            winner = min(first_v, second_v) if lower_is_better else max(first_v, second_v)
            loser = max(first_v, second_v) if lower_is_better else min(first_v, second_v)
            label = _format_delta(winner, loser, delta_style)
            if not label:
                continue
            winner_is_first = first_v < second_v if lower_is_better else first_v > second_v
            color = theme["delta_win"] if winner_is_first else theme["delta_lose"]
            ax.text(
                idx,
                max(first_v, second_v) + y_pad * 5,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color=color,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor=theme["delta_box_face"],
                    edgecolor=color,
                    linewidth=0.8,
                    alpha=0.85,
                ),
            )
    ax.set_ylim(top=y_max * 1.35)


def _value_fmt(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "{:.2f}"
    return "{:.3f}" if float(values.max()) < 10 else "{:.1f}"


def _color_mapping(args: argparse.Namespace, systems: Sequence[str]) -> Dict[str, str]:
    colors = [args.engine1_color, args.engine2_color]
    return {system: colors[i] for i, system in enumerate(systems) if i < len(colors)}


def plot_metric_sweep(
    df: pd.DataFrame,
    *,
    axis_col: str,
    sweep_type: str,
    metric_key: str,
    metric_display: str,
    units: str,
    lower_is_better: bool,
    output_dir: Path,
    systems: Sequence[str],
    dark_mode: bool,
    transparent: bool,
) -> None:
    sub = df[df["metric"] == metric_display].copy()
    if sub.empty:
        return
    compare_pairs = len(systems) >= 2
    direction = "lower is better" if lower_is_better else "higher is better"
    sub_bar = sub.copy()
    sub_bar.loc[pd.isna(sub_bar["value"]), "value"] = 0.0
    value_fmt = _value_fmt(sub["value"])

    if compare_pairs:
        fig = rk.bar(
            sub_bar,
            x=axis_col,
            y="value",
            color="system",
            facet_col="percentile",
            palette="cool",
            barmode="group",
            bar_edge=True,
            share_y=True,
            share_x=True,
            figsize=(15, 5.5),
            dark_mode=dark_mode,
            title=f"{metric_display} ({units}) vs {_axis_label(sweep_type)} ({direction})",
            xlabel=_axis_label(sweep_type),
            ylabel=f"{metric_display} ({units})",
            color_label="System",
            facet_col_label="Percentile",
        )
        axes = fig.axes.flatten()  # type: ignore[union-attr]
    else:
        fig = rk.bar(
            sub_bar,
            x=axis_col,
            y="value",
            color="percentile",
            palette="cool",
            barmode="group",
            bar_edge=True,
            figsize=(12, 5.5),
            dark_mode=dark_mode,
            title=f"{metric_display} ({units}) vs {_axis_label(sweep_type)} ({direction})",
            xlabel=_axis_label(sweep_type),
            ylabel=f"{metric_display} ({units})",
            color_label="Percentile",
        )
        axes = fig.get_axes()
    for ax in axes:
        _annotate_bar_axes(
            ax,
            value_fmt,
            lower_is_better=lower_is_better,
            dark_mode=dark_mode,
            compare_pairs=compare_pairs,
        )
    path = output_dir / f"{metric_key}_bar_{sweep_type}_sweep.png"
    fig.save(str(path), transparent=transparent)
    print(f"Wrote {path}")

    fig = rk.line(
        sub,
        x=axis_col,
        y="value",
        color="system" if compare_pairs else "percentile",
        facet_col="percentile" if compare_pairs else None,
        palette="cool",
        markers=True,
        line_width=2.5,
        marker_size=8,
        share_y=True,
        share_x=True,
        figsize=(15, 4.5) if compare_pairs else (12, 4.5),
        dark_mode=dark_mode,
        title=f"{metric_display} ({units}) vs {_axis_label(sweep_type)} ({direction})",
        xlabel=_axis_label(sweep_type),
        ylabel=f"{metric_display} ({units})",
        color_label="System" if compare_pairs else "Percentile",
        facet_col_label="Percentile" if compare_pairs else None,
    )
    path = output_dir / f"{metric_key}_line_{sweep_type}_sweep.png"
    fig.save(str(path), transparent=transparent)
    print(f"Wrote {path}")


def plot_metric_report(
    df: pd.DataFrame,
    *,
    axis_col: str,
    sweep_type: str,
    metric_key: str,
    metric_display: str,
    units: str,
    lower_is_better: bool,
    output_dir: Path,
    systems: Sequence[str],
    dark_mode: bool,
    transparent: bool,
    color_mapping: Dict[str, str],
) -> None:
    compare_pairs = len(systems) >= 2
    for percentile in PERCENTILES:
        sub = df[(df["metric"] == metric_display) & (df["percentile"] == percentile)].copy()
        if sub.empty:
            continue
        sub_bar = sub.copy()
        sub_bar.loc[pd.isna(sub_bar["value"]), "value"] = 0.0
        value_fmt = _value_fmt(sub["value"])
        tag = percentile.lower()

        fig = rk.bar(
            sub_bar,
            x=axis_col,
            y="value",
            color="system",
            palette="cool",
            color_mapping=color_mapping,
            barmode="group",
            bar_edge=True,
            figsize=(11, 5.0),
            dark_mode=dark_mode,
            xlabel=_axis_label(sweep_type),
            ylabel=f"{metric_display} {percentile} ({units})",
            color_label="System",
        )
        for ax in fig.get_axes():
            _annotate_bar_axes(
                ax,
                value_fmt,
                lower_is_better=lower_is_better,
                dark_mode=dark_mode,
                compare_pairs=compare_pairs,
                delta_style="multiplier",
                show_value_labels=metric_display != "RTF",
            )
            ax.set_title("")
        path = output_dir / f"{metric_key}_{tag}_bar_{sweep_type}_report.png"
        fig.save(str(path), transparent=transparent)
        print(f"Wrote {path}")

        fig = rk.line(
            sub,
            x=axis_col,
            y="value",
            color="system",
            palette="cool",
            color_mapping=color_mapping,
            markers=True,
            line_width=2.5,
            marker_size=8,
            figsize=(11, 4.5),
            dark_mode=dark_mode,
            xlabel=_axis_label(sweep_type),
            ylabel=f"{metric_display} {percentile} ({units})",
            color_label="System",
        )
        for ax in fig.get_axes():
            _annotate_line_axes(
                ax,
                sub,
                axis_col,
                "value",
                value_fmt,
                lower_is_better=lower_is_better,
                dark_mode=dark_mode,
                compare_pairs=compare_pairs,
                delta_style="multiplier",
            )
            ax.set_title("")
        path = output_dir / f"{metric_key}_{tag}_line_{sweep_type}_report.png"
        fig.save(str(path), transparent=transparent)
        print(f"Wrote {path}")


def plot_auxiliary_sweep(
    df: pd.DataFrame,
    *,
    axis_col: str,
    sweep_type: str,
    output_dir: Path,
    systems: Sequence[str],
    dark_mode: bool,
    transparent: bool,
) -> None:
    for value_col, label, filename, value_fmt in (
        ("completed_requests", "Completed Requests", "completed_requests", "{:.0f}"),
        (
            "chars_per_sec_aggregate",
            "Chars/sec (aggregate)",
            "chars_per_sec_aggregate",
            "{:.1f}",
        ),
    ):
        if value_col not in df.columns:
            continue
        sub = df[["system", axis_col, value_col]].drop_duplicates().copy()
        values = pd.to_numeric(sub[value_col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        sub[value_col] = values.fillna(0.0)
        fig = rk.bar(
            sub,
            x=axis_col,
            y=value_col,
            color="system",
            palette="cool",
            barmode="group",
            bar_edge=True,
            figsize=(12, 5.0),
            dark_mode=dark_mode,
            title=f"{label} vs {_axis_label(sweep_type)}",
            xlabel=_axis_label(sweep_type),
            ylabel=label,
            color_label="System",
        )
        for ax in fig.get_axes():
            _annotate_bar_axes(
                ax,
                value_fmt,
                lower_is_better=False,
                dark_mode=dark_mode,
                compare_pairs=len(systems) >= 2,
            )
        path = output_dir / f"{filename}_bar_{sweep_type}_sweep.png"
        fig.save(str(path), transparent=transparent)
        print(f"Wrote {path}")


def _build_dataframe(
    args: argparse.Namespace, systems: Sequence[str]
) -> Tuple[pd.DataFrame, Sequence[MetricSpec], List[SelectedRun]]:
    if args.sweep_type == "conc":
        metrics = (
            SINGLE_CONC_METRICS
            if len(systems) == 1 and args.plot_type == "sweep"
            else REPORT_METRICS
        )
        percentiles = (
            SINGLE_CONC_PERCENTILES
            if len(systems) == 1 and args.plot_type == "sweep"
            else PERCENTILES
        )
        runs = collect_runs(
            systems[0],
            args.engine1_dir,
            args.min_completed_requests,
            metrics=metrics,
            percentiles=percentiles,
        )
        if args.engine2_dir:
            runs += collect_runs(
                systems[1],
                args.engine2_dir,
                args.min_completed_requests,
                metrics=metrics,
                percentiles=percentiles,
            )
        selected = pick_best_rtf(runs)
        return (
            build_concurrency_df(
                selected, systems, metrics=metrics, percentiles=percentiles
            ),
            metrics,
            _sort_selected_runs(selected),
        )

    runs = collect_input_runs(
        systems[0],
        args.engine1_dir,
        args.min_completed_requests,
        metrics=REPORT_METRICS,
        percentiles=PERCENTILES,
    )
    if args.engine2_dir:
        runs += collect_input_runs(
            systems[1],
            args.engine2_dir,
            args.min_completed_requests,
            metrics=REPORT_METRICS,
            percentiles=PERCENTILES,
        )
    selected = pick_best_input_rtf(runs)
    return (
        build_input_df(
            selected, systems, metrics=REPORT_METRICS, percentiles=PERCENTILES
        ),
        REPORT_METRICS,
        _sort_selected_runs(selected),
    )


def print_summary(df: pd.DataFrame, axis_col: str) -> None:
    if df.empty:
        print("[warn] No data to display.", file=sys.stderr)
        return
    pivot = df.pivot_table(
        index=["metric", "percentile", axis_col],
        columns="system",
        values="value",
        observed=True,
    )
    print("\nSelected-run comparison:")
    with pd.option_context("display.float_format", lambda value: f"{value:.4f}"):
        print(pivot.to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sweep-type", type=_normalize_sweep_type, required=True)
    parser.add_argument("--plot-type", choices=("sweep", "report"), default="sweep")
    parser.add_argument("--engine1-dir", "--vajra-dir", required=True)
    parser.add_argument("--engine2-dir", "--vllm-dir")
    parser.add_argument("--engine1-name", "--vajra-name", default="Engine 1")
    parser.add_argument("--engine2-name", "--vllm-name", default="Engine 2")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-completed-requests", type=int, default=0)
    parser.add_argument("--dark-mode", action="store_true")
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--engine1-color", default=DEFAULT_ENGINE1_COLOR)
    parser.add_argument("--engine2-color", default=DEFAULT_ENGINE2_COLOR)
    parser.add_argument(
        "--plot-both",
        action="store_true",
        help="Only plot x-axis values present for every provided engine.",
    )
    parser.add_argument(
        "--exp-name",
        help="Copy selected run data under --benchmarks-dir using sweep/{conc|input}/{engine}/{axis=value}.",
    )
    parser.add_argument(
        "--model-name",
        help="Model folder name for --exp-name archives; inferred from run configs if omitted.",
    )
    parser.add_argument(
        "--benchmarks-dir",
        default=str(DEFAULT_NARADA_DIR / "benchmarks"),
        help="Archive root for --exp-name data copies.",
    )
    parser.add_argument(
        "--include-plots",
        action="store_true",
        help="Also copy generated PNGs from each selected run's metrics directory.",
    )
    parser.add_argument(
        "--overwrite",
        "--overwrite-archive",
        dest="overwrite_archive",
        action="store_true",
        help="Replace existing archived data directories.",
    )
    parser.add_argument(
        "--dry-run-archive",
        action="store_true",
        help="Print archive targets without copying selected run data.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    systems = [args.engine1_name]
    if args.engine2_dir:
        systems.append(args.engine2_name)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    axis_col = _axis_col(args.sweep_type)

    df, metrics, selected_runs = _build_dataframe(args, systems)
    if df.empty:
        print("[error] No valid benchmark runs found.", file=sys.stderr)
        return 1

    if args.plot_both and len(systems) >= 2:
        df = filter_complete_axis(df, axis_col, systems)
        if df.empty:
            print("[error] --plot-both removed all x-axis values.", file=sys.stderr)
            return 1

    csv_path = output_dir / "sweep_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    color_mapping = _color_mapping(args, systems)
    for key, display, units, lower_is_better in metrics:
        if args.plot_type == "report":
            plot_metric_report(
                df,
                axis_col=axis_col,
                sweep_type=args.sweep_type,
                metric_key=key,
                metric_display=display,
                units=units,
                lower_is_better=lower_is_better,
                output_dir=output_dir,
                systems=systems,
                dark_mode=args.dark_mode,
                transparent=args.transparent,
                color_mapping=color_mapping,
            )
        else:
            plot_metric_sweep(
                df,
                axis_col=axis_col,
                sweep_type=args.sweep_type,
                metric_key=key,
                metric_display=display,
                units=units,
                lower_is_better=lower_is_better,
                output_dir=output_dir,
                systems=systems,
                dark_mode=args.dark_mode,
                transparent=args.transparent,
            )

    if args.plot_type == "sweep":
        plot_auxiliary_sweep(
            df,
            axis_col=axis_col,
            sweep_type=args.sweep_type,
            output_dir=output_dir,
            systems=systems,
            dark_mode=args.dark_mode,
            transparent=args.transparent,
        )

    print_summary(df, axis_col)
    archive_selected_sweep_runs(
        selected_runs,
        axis_values=df[axis_col].dropna().astype(str).unique(),
        axis_col=axis_col,
        sweep_type=args.sweep_type,
        systems=systems,
        exp_name=args.exp_name,
        model_name=args.model_name,
        benchmarks_dir=args.benchmarks_dir,
        include_plots=args.include_plots,
        overwrite=args.overwrite_archive,
        dry_run=args.dry_run_archive,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
