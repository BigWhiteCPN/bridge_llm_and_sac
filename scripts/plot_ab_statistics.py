#!/usr/bin/env python3
"""Generate tables and an SVG figure from paired Bridge A/B results."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any


METHOD_LABELS = {"off": "Baseline", "replan": "Bridge/Replan"}


def result_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        direct = path / "results.json"
        if direct.exists():
            files.append(direct)
        files.extend(sorted(path.glob("ab_*/results.json")))
        files.extend(sorted(path.glob("ab_runtime_*/results.json")))
        files.extend(sorted(path.glob("showcase_*/results.json")))
    return sorted(set(files))


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    files = result_files(paths)
    if not files:
        searched = ", ".join(str(path) for path in paths)
        raise FileNotFoundError(f"No results.json files found under: {searched}")
    rows: list[dict[str, Any]] = []
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{file} must contain a list of result rows")
        for row in data:
            if isinstance(row, dict):
                row = dict(row)
                row.setdefault("experiment", file.parent.name)
                row.setdefault("results_file", str(file))
                rows.append(row)
    return rows


def as_float(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def as_bool(row: dict[str, Any], *keys: str, default: bool = False) -> bool:
    for key in keys:
        if key in row:
            return bool(row[key])
    return default


def metric_value(row: dict[str, Any], metric: str) -> float:
    if metric == "success":
        return 1.0 if as_bool(row, "episode_success", "target_complete") else 0.0
    if metric == "env_steps":
        return as_float(row, "episode_env_steps", "steps")
    if metric == "path_length":
        return as_float(row, "path_length", default=float("nan"))
    if metric == "final_distance_to_goal":
        return as_float(row, "final_distance_to_goal", default=float("nan"))
    if metric == "num_replans":
        return as_float(row, "num_replans", "advisor_control_stops")
    raise KeyError(metric)


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean(values: list[float]) -> float:
    values = finite(values)
    return statistics.fmean(values) if values else float("nan")


def ci95(values: list[float]) -> float:
    values = finite(values)
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def pct(delta: float, base: float) -> float:
    if not math.isfinite(delta) or not math.isfinite(base) or abs(base) < 1e-9:
        return float("nan")
    return 100.0 * delta / base


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        condition = str(row.get("condition", ""))
        if pair_id and condition in {"off", "replan"}:
            grouped.setdefault(pair_id, {})[condition] = row

    pairs = []
    for pair_id, items in sorted(grouped.items()):
        off = items.get("off")
        replan = items.get("replan")
        if not off or not replan:
            continue
        task = str(off.get("task", replan.get("task", "")))
        seed = int(as_float(off, "seed", default=0.0))
        row = {
            "pair_id": pair_id,
            "experiment": str(off.get("experiment", replan.get("experiment", ""))),
            "task": task,
            "seed": seed,
            "success_off": int(metric_value(off, "success")),
            "success_replan": int(metric_value(replan, "success")),
            "env_steps_off": metric_value(off, "env_steps"),
            "env_steps_replan": metric_value(replan, "env_steps"),
            "path_length_off": metric_value(off, "path_length"),
            "path_length_replan": metric_value(replan, "path_length"),
            "final_distance_to_goal_off": metric_value(off, "final_distance_to_goal"),
            "final_distance_to_goal_replan": metric_value(replan, "final_distance_to_goal"),
            "num_replans_replan": metric_value(replan, "num_replans"),
            "done_reason_off": str(off.get("done_reason", "")),
            "done_reason_replan": str(replan.get("done_reason", "")),
        }
        for key in ("env_steps", "path_length", "final_distance_to_goal"):
            off_value = float(row[f"{key}_off"])
            replan_value = float(row[f"{key}_replan"])
            row[f"{key}_delta_replan_minus_off"] = replan_value - off_value
        pairs.append(row)
    return pairs


def metric_summaries(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("success", "Success rate", True, "%"),
        ("env_steps", "Episode steps", False, "steps"),
        ("path_length", "Path length", False, "m"),
        ("final_distance_to_goal", "Final distance", False, "m"),
    ]
    summaries = []
    for key, label, higher_is_better, unit in specs:
        off_values = [float(row[f"{key}_off"]) if key != "success" else float(row["success_off"]) for row in pairs]
        replan_values = [
            float(row[f"{key}_replan"]) if key != "success" else float(row["success_replan"]) for row in pairs
        ]
        off_values = finite(off_values)
        replan_values = finite(replan_values)
        deltas = [b - a for a, b in zip(off_values, replan_values) if math.isfinite(a) and math.isfinite(b)]
        baseline_mean = mean(off_values)
        bridge_mean = mean(replan_values)
        delta = bridge_mean - baseline_mean
        if key == "success":
            delta_pct = 100.0 * delta
        else:
            delta_pct = pct(delta, baseline_mean)
        summaries.append(
            {
                "metric": key,
                "label": label,
                "unit": unit,
                "higher_is_better": higher_is_better,
                "n": min(len(off_values), len(replan_values)),
                "baseline_mean": baseline_mean,
                "bridge_mean": bridge_mean,
                "delta_replan_minus_off": delta,
                "delta_percent": delta_pct,
                "baseline_ci95": ci95(off_values),
                "bridge_ci95": ci95(replan_values),
                "paired_delta_ci95": ci95(deltas),
            }
        )
    return summaries


def fmt(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_report(path: Path, summaries: list[dict[str, Any]], pairs: list[dict[str, Any]], figure_name: str) -> None:
    success = next(row for row in summaries if row["metric"] == "success")
    env_steps = next(row for row in summaries if row["metric"] == "env_steps")
    improved_steps = sum(1 for row in pairs if float(row["env_steps_delta_replan_minus_off"]) < 0)
    bridge_success_gain = sum(1 for row in pairs if row["success_off"] == 0 and row["success_replan"] == 1)
    bridge_success_loss = sum(1 for row in pairs if row["success_off"] == 1 and row["success_replan"] == 0)
    lines = [
        "# Bridge A/B Statistics",
        "",
        f"Paired episodes: **{len(pairs)}**",
        "",
        f"![A/B statistics]({figure_name})",
        "",
        "## Key Results",
        "",
        f"- Success rate: baseline {fmt(100 * success['baseline_mean'], 1)}%, "
        f"bridge {fmt(100 * success['bridge_mean'], 1)}% "
        f"({fmt(success['delta_percent'], 1)} percentage points).",
        f"- Mean episode steps: baseline {fmt(env_steps['baseline_mean'], 1)}, "
        f"bridge {fmt(env_steps['bridge_mean'], 1)} "
        f"({fmt(env_steps['delta_percent'], 1)}%).",
        f"- Pairs with fewer bridge steps: {improved_steps}/{len(pairs)}.",
        f"- Success gains/losses from bridge: +{bridge_success_gain}/-{bridge_success_loss} pairs.",
        "",
        "## Metrics",
        "",
        "| metric | baseline | bridge | delta | delta % / pp | n |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        delta_unit = "pp" if row["metric"] == "success" else "%"
        baseline = 100 * row["baseline_mean"] if row["metric"] == "success" else row["baseline_mean"]
        bridge = 100 * row["bridge_mean"] if row["metric"] == "success" else row["bridge_mean"]
        delta = 100 * row["delta_replan_minus_off"] if row["metric"] == "success" else row["delta_replan_minus_off"]
        lines.append(
            f"| {row['label']} | {fmt(baseline, 2)} | {fmt(bridge, 2)} | "
            f"{fmt(delta, 2)} | {fmt(row['delta_percent'], 2)} {delta_unit} | {row['n']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Lower is better for episode steps, path length, and final distance.",
            "- The figure and tables are generated only from the supplied `results.json` files.",
            "- Use at least 50-100 paired episodes before making aggregate claims.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def svg_bar(
    x: float,
    y: float,
    width: float,
    height: float,
    value: float,
    scale: float,
    color: str,
    label: str,
    value_label: str,
) -> str:
    bar_width = 0.0 if not math.isfinite(value) or scale <= 0 else max(1.0, width * value / scale)
    return "\n".join(
        [
            f'<text x="{x:.1f}" y="{y - 6:.1f}" font-size="13" fill="#263238">{html.escape(label)}</text>',
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="3" fill="#eef2f5"/>',
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}" rx="3" fill="{color}"/>',
            f'<text x="{x + width + 10:.1f}" y="{y + height - 5:.1f}" font-size="13" fill="#263238">{html.escape(value_label)}</text>',
        ]
    )


def write_svg(path: Path, summaries: list[dict[str, Any]], pairs: list[dict[str, Any]], title: str) -> None:
    width = 980
    height = 680
    left = 260
    bar_width = 430
    bar_height = 22
    colors = {"Baseline": "#4C78A8", "Bridge/Replan": "#59A14F"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="40" y="48" font-size="26" font-weight="700" fill="#17202a">{html.escape(title)}</text>',
        f'<text x="40" y="76" font-size="14" fill="#52616b">N={len(pairs)} paired episodes. Error bars are reported in CSV/Markdown; bars show means.</text>',
    ]
    y = 125
    for row in summaries:
        baseline = float(row["baseline_mean"])
        bridge = float(row["bridge_mean"])
        if row["metric"] == "success":
            baseline *= 100.0
            bridge *= 100.0
            scale = 100.0
            suffix = "%"
            delta_text = f"{fmt(row['delta_percent'], 1)} pp"
        else:
            scale = max([value for value in [baseline, bridge] if math.isfinite(value)] + [1.0]) * 1.15
            suffix = f" {row['unit']}" if row["unit"] else ""
            delta_text = f"{fmt(row['delta_percent'], 1)}%"
        parts.append(f'<text x="40" y="{y:.1f}" font-size="18" font-weight="700" fill="#263238">{html.escape(row["label"])}</text>')
        direction = "higher is better" if row["higher_is_better"] else "lower is better"
        parts.append(f'<text x="40" y="{y + 20:.1f}" font-size="12" fill="#6b7780">{direction}; bridge delta {html.escape(delta_text)}</text>')
        parts.append(svg_bar(left, y - 8, bar_width, bar_height, baseline, scale, colors["Baseline"], "Baseline", f"{fmt(baseline, 1)}{suffix}"))
        parts.append(
            svg_bar(
                left,
                y + 24,
                bar_width,
                bar_height,
                bridge,
                scale,
                colors["Bridge/Replan"],
                "Bridge/Replan",
                f"{fmt(bridge, 1)}{suffix}",
            )
        )
        y += 118

    improved = sum(1 for row in pairs if float(row["env_steps_delta_replan_minus_off"]) < 0)
    same = sum(1 for row in pairs if float(row["env_steps_delta_replan_minus_off"]) == 0)
    worse = len(pairs) - improved - same
    y += 12
    parts.extend(
        [
            f'<text x="40" y="{y:.1f}" font-size="18" font-weight="700" fill="#263238">Paired step comparison</text>',
            f'<text x="40" y="{y + 24:.1f}" font-size="14" fill="#263238">Bridge fewer steps: {improved}</text>',
            f'<text x="290" y="{y + 24:.1f}" font-size="14" fill="#263238">Equal: {same}</text>',
            f'<text x="420" y="{y + 24:.1f}" font-size="14" fill="#263238">Bridge more steps: {worse}</text>',
            '<rect x="40" y="630" width="900" height="1" fill="#d9e1e8"/>',
            '<text x="40" y="656" font-size="12" fill="#6b7780">Generated from paired A/B results.json files. Do not report aggregate claims unless N is large enough.</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Bridge A/B statistics tables and SVG figure.")
    parser.add_argument("paths", nargs="*", default=["data"], help="A/B result dirs, results.json files, or parent dirs.")
    parser.add_argument("--output-dir", default="reports/ab_statistics")
    parser.add_argument("--title", default="Bridge Advisor A/B Evaluation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = load_results([Path(path) for path in args.paths])
    pairs = paired_rows(rows)
    if not pairs:
        raise FileNotFoundError("No complete off/replan pairs found in results.json files")

    summaries = metric_summaries(pairs)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "pair_results.csv",
        pairs,
        [
            "experiment",
            "pair_id",
            "task",
            "seed",
            "success_off",
            "success_replan",
            "env_steps_off",
            "env_steps_replan",
            "env_steps_delta_replan_minus_off",
            "path_length_off",
            "path_length_replan",
            "path_length_delta_replan_minus_off",
            "final_distance_to_goal_off",
            "final_distance_to_goal_replan",
            "final_distance_to_goal_delta_replan_minus_off",
            "num_replans_replan",
            "done_reason_off",
            "done_reason_replan",
        ],
    )
    write_csv(
        output_dir / "metrics_summary.csv",
        summaries,
        [
            "metric",
            "label",
            "unit",
            "higher_is_better",
            "n",
            "baseline_mean",
            "bridge_mean",
            "delta_replan_minus_off",
            "delta_percent",
            "baseline_ci95",
            "bridge_ci95",
            "paired_delta_ci95",
        ],
    )
    (output_dir / "metrics_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    write_svg(output_dir / "ab_statistics.svg", summaries, pairs, args.title)
    write_report(output_dir / "ab_statistics.md", summaries, pairs, "ab_statistics.svg")
    print(f"[stats] wrote {output_dir}")


if __name__ == "__main__":
    main()
