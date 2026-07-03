from __future__ import annotations

import json

from scripts.plot_ab_statistics import load_results, metric_summaries, paired_rows


def test_plot_ab_statistics_uses_episode_metrics(tmp_path):
    exp = tmp_path / "ab_runtime_100"
    exp.mkdir()
    (exp / "results.json").write_text(
        json.dumps(
            [
                {
                    "pair_id": "pair_001_pantry",
                    "condition": "off",
                    "task": "pantry",
                    "seed": 1,
                    "episode_success": False,
                    "episode_env_steps": 2000,
                    "path_length": 30.0,
                    "final_distance_to_goal": 4.0,
                    "done_reason": "timeout",
                },
                {
                    "pair_id": "pair_001_pantry",
                    "condition": "replan",
                    "task": "pantry",
                    "seed": 1,
                    "episode_success": True,
                    "episode_env_steps": 1200,
                    "path_length": 18.0,
                    "final_distance_to_goal": 0.8,
                    "num_replans": 1,
                    "done_reason": "success",
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = load_results([tmp_path])
    pairs = paired_rows(rows)
    summaries = {row["metric"]: row for row in metric_summaries(pairs)}

    assert len(pairs) == 1
    assert pairs[0]["env_steps_delta_replan_minus_off"] == -800
    assert pairs[0]["path_length_delta_replan_minus_off"] == -12.0
    assert summaries["success"]["delta_percent"] == 100.0
    assert summaries["env_steps"]["delta_percent"] == -40.0
