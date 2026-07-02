# LLM-to-SAC Cognitive Bridge

This repository is a research extension of the robot agent system in
`/home/chen/code/IsaacLabExtensionTemplate/scripts/agent_system_complex_version`.
It was developed as an independent training workspace for a cross-attention
bridge between high-level LLM task planning and the SAC navigation stack.

The original agent system remains the runtime source of robot state, memory,
topological map context and navigation callbacks. This project focuses on the
middle bridge layer: collecting multimodal navigation snapshots, training a
world-model-like advisor, and testing when the high-level planner should keep,
interrupt, or revise the current navigation skill.

The bridge is not a low-level controller. SAC still executes `go_to(x, y)` and local navigation. The bridge learns the task-level monitoring layer:

- whether the current skill should continue
- whether navigation is stuck or low value
- whether a visual/semantic event should trigger scanning
- which candidate subgoal is worth giving to SAC
- when the LLM should be asked for high-level replanning

## Architecture

`BridgeNet-v1` uses separate encoders per modality, then fuses them with cross-attention:

```text
map layers       -> CNN map tokens
robot state      -> MLP state token
TaskSpec ids     -> task token
memory/topology  -> MLP memory token
candidate goals  -> MLP candidate tokens

query tokens:   [CLS, task, state, skill]
context tokens: [map tokens, memory token, candidate tokens]

cross-attention + transformer fusion
        -> GRU over recent snapshots
        -> event / replan / outcome / candidate-score heads
```

This is intentionally closer to a neuro-inspired bridge than a direct VLA: the model predicts events, progress and skill gates, while SAC keeps the fast motor loop.

The current runtime format uses a 22-D state token. The first 16 dimensions are
instantaneous pose/task/progress features; the last 6 dimensions are segment
memory for the active `go_to`: elapsed callbacks, start distance, best distance,
progress from segment start, distance regret from the best point, and recent
window progress. These features give the bridge a denser channel between
high-level task intent and low-level SAC execution.

## Episode Format

Training episodes are compressed `.npz` files under:

```text
data/<dataset_name>/episodes/episode_000001.npz
```

Required arrays:

```text
maps: [T, C, H, W] float32
state: [T, state_dim] float32
memory: [T, memory_dim] float32
task: [T, 3] int64  # intent, target, constraint ids
skill: [T] int64
candidates: [T, K, candidate_dim] float32
candidate_mask: [T, K] bool
event: [T] int64
replan: [T] int64
success/stuck/target_found/cost/info_gain: [T] float32
candidate_score_target: [T, K] float32
```

## Smoke Test

Generate synthetic data:

```bash
python make_synthetic_dataset.py --output-dir data/synthetic --episodes 32 --steps 96
```

Train:

```bash
python train.py --data-dir data/synthetic --epochs 3 --batch-size 8
```

Validate a dataset:

```bash
python validate_episode.py data/synthetic
```

Run the standalone recorder smoke test:

```bash
python smoke_recorder.py
```

Run a sidecar collector smoke test with fake runtime objects:

```bash
python smoke_sidecar.py
```

Convert existing `agent_system_complex_version/memory_logs` into weakly labeled
bridge episodes:

```bash
python convert_memory_logs.py \
  --memory-logs ../agent_system_complex_version/memory_logs \
  --output-dir data/from_memory_logs

python validate_episode.py data/from_memory_logs
python train.py --data-dir data/from_memory_logs --epochs 3 --batch-size 8
```

Evaluate a checkpoint with per-class reports:

```bash
python evaluate.py \
  --checkpoint runs/pretrain_v1_cuda_episode_split/best.pt \
  --data-dir data/from_memory_logs_all \
  --split val
```

The memory-log converter uses final saved maps, topological nodes and landmark
memory. It is useful for representation pretraining, but it is not a substitute
for runtime recorder data because the historical logs do not contain true
per-step skill outcomes.

Run tests:

```bash
pytest
```

Or run a dependency-light forward/loss smoke check:

```bash
python smoke_forward.py
```

Use the MuJoCo environment Python if the system Python does not have Torch:

```bash
export MUJOCO_PYTHON=/path/to/mujoco_env/bin/python
$MUJOCO_PYTHON make_synthetic_dataset.py
$MUJOCO_PYTHON train.py --data-dir data/synthetic --epochs 3
```

## Next Integration Step

Use `BridgeEpisodeRecorder` from a wrapper, optional callback, or offline log converter. Keep the existing robot stack as the source of snapshots; the recorder only receives arrays and writes `.npz` training episodes.

Minimal external usage:

```python
from bridge.recorder import BridgeEpisodeRecorder, StepSignals, TaskSpec

recorder = BridgeEpisodeRecorder(
    "data/real_runs",
    task_spec=TaskSpec(intent="search_place", target="meeting_room", constraint="stop_when_found"),
)

recorder.record_step(
    maps=local_map_layers,        # [5, 64, 64]
    state=state_vector,           # [22]
    memory=memory_vector,         # [12]
    skill="explore_frontier",
    candidates=candidate_features, # [K, 8]
    signals=StepSignals(distance_to_subgoal=dist, stuck_score=stuck, info_gain=map_gain),
)

episode_path = recorder.save()
```

Useful snapshot sources when adding a non-invasive hook later:

- `NavigationSkill.go_to()`
- `FrontierExplorationSkill.execute()`
- dashboard/VLM keyframe hooks later

The sidecar helper keeps this non-invasive:

```python
from bridge.recorder import TaskSpec
from bridge.sidecar import BridgeSidecarCollector

collector = BridgeSidecarCollector(
    "data/runtime",
    task_spec=TaskSpec(intent="search_place", target="meeting_room", constraint="stop_when_found"),
)

# Call from an external loop or wrapper; this only reads env/memory/topo_map.
collector.observe(env, memory=memory, topo_map=topo_map, skill="explore_frontier")

episode_path = collector.save()
```

Collect a real runtime episode without editing `agent_system_complex_version`:

```bash
$MUJOCO_PYTHON collect_runtime.py \
  --agent-root ../agent_system_complex_version \
  --output-dir data/runtime \
  --max-rounds 2 \
  --nav-steps-per-round 800 \
  --target-place 会议室
```

`collect_runtime.py` wraps only the live `nav_skill.go_to` instance so its
existing `step_callback` also records bridge snapshots. It restores the method
when the run exits.

Collect a small batch:

```bash
$MUJOCO_PYTHON collect_runtime_batch.py \
  --output-dir data/runtime_v3 \
  --episodes 10 \
  --max-rounds 1 \
  --nav-steps-per-round 400
```

Merge several runtime datasets without filename collisions:

```bash
$MUJOCO_PYTHON merge_episodes.py \
  --output-dir data/runtime_v2_v3_v4 \
  --overwrite \
  data/runtime_v2 data/runtime_v3 data/runtime_v4
```

Fine-tune from the memory-log pretrain checkpoint:

```bash
$MUJOCO_PYTHON train.py \
  --data-dir data/runtime_v2_v3_v4 \
  --output-dir runs/runtime_v2_v3_v4_finetune \
  --init-checkpoint runs/pretrain_v2_cuda/best.pt \
  --epochs 80 \
  --batch-size 8 \
  --sequence-len 16 \
  --hidden-dim 128 \
  --fusion-layers 2 \
  --num-heads 4 \
  --device cuda \
  --split-by episode \
  --val-fraction 0.2 \
  --balanced-class-loss \
  --patience 12 \
  --lr 0.0001
```

Run the trained bridge as a passive advisor on a recorded episode:

```bash
$MUJOCO_PYTHON smoke_advisor.py \
  --checkpoint runs/runtime_v2_v3_v4_finetune/best.pt \
  --data-dir data/runtime_v2_v3_v4 \
  --device cpu
```

Run the advisor during real runtime collection:

```bash
$MUJOCO_PYTHON collect_runtime.py \
  --output-dir data/runtime_advisor_smoke \
  --episode-id advisor_smoke_001 \
  --max-rounds 1 \
  --nav-steps-per-round 250 \
  --render-mode rgb_array \
  --mujoco-gl egl \
  --target-place 会议室 \
  --intent search_place \
  --target meeting_room \
  --constraint stop_when_found \
  --advisor-checkpoint runs/runtime_v2_v3_v4_finetune/best.pt \
  --advisor-device cpu \
  --advisor-log-every 5
```

Enable optional advisor arbitration:

```bash
$MUJOCO_PYTHON collect_runtime.py \
  --output-dir data/runtime_arbitration_smoke \
  --episode-id arbitration_smoke_001 \
  --max-rounds 1 \
  --nav-steps-per-round 250 \
  --render-mode rgb_array \
  --mujoco-gl egl \
  --target-place 会议室 \
  --intent search_place \
  --target meeting_room \
  --constraint stop_when_found \
  --advisor-checkpoint runs/runtime_v2_v3_v4_finetune/best.pt \
  --advisor-device cpu \
  --advisor-log-every 4 \
  --advisor-control replan \
  --advisor-stop-confidence 0.92 \
  --advisor-stop-consecutive 2 \
  --advisor-warmup-steps 6
```

`--advisor-control off` keeps the bridge passive. `safe` only stops on target or
subgoal completion. `replan` also allows repeated stuck/low-information signals
to stop the current `go_to`; the wrapper then returns `success=False` so the
frontier logic can choose a new subgoal.

Run a paired A/B experiment:

```bash
$MUJOCO_PYTHON run_ab_experiment.py \
  --output-dir data/ab_runtime_v1 \
  --episodes 5 \
  --seed-base 12000 \
  --max-rounds 1 \
  --nav-steps-per-round 300 \
  --render-mode rgb_array \
  --mujoco-gl egl \
  --callback-freq 20 \
  --observe-interval-s 0.1 \
  --advisor-checkpoint runs/runtime_v2_v3_v4_finetune/best.pt \
  --advisor-device cpu \
  --advisor-stop-confidence 0.90 \
  --advisor-stop-consecutive 2 \
  --advisor-warmup-steps 6
```

The A/B runner writes per-episode logs plus `results.json` and `summary.json`.
Seeded pairs use the same task and initial random seed for `off` and `replan`.

Summarize a dataset or audit split coverage:

```bash
python summarize_episodes.py data/runtime_hindsight_segment_v1

$MUJOCO_PYTHON audit_splits.py \
  --data-dir data/runtime_hindsight_segment_v1 \
  --seed-start 1 \
  --seed-end 30
```

Runtime collection now adds hindsight labels by default for frontier-like
`go_to` segments that already had a high-level callback. If a segment returns
failure, the segment tail is relabeled as `path_invalidated -> switch_subgoal`.
This gives the bridge supervision for "this subgoal is becoming bad" instead of
only instantaneous stuck/low-information labels. Use
`--disable-hindsight-labels` to keep purely online rule labels.

Optional `--progress-guard` stops obviously unproductive frontier segments and
records the trigger in episode metadata. Its tail frames are also relabeled as
`path_invalidated -> switch_subgoal`, so later training can learn this reflex
from data instead of relying on the hand rule forever.

## Latest Segment-Progress Run

`data/runtime_segment_v1` adds 10 real episodes with 22-D state features and 8
progress-guard triggers. Merged with the earlier hindsight set:

```text
data/runtime_hindsight_segment_v1
episodes=30 steps=2503 state_dims={16: 20, 22: 10}
events: path_invalidated=174, low_information_gain=551, navigation_stuck=205
```

The current best checkpoint is:

```text
runs/runtime_hindsight_segment_v1_riskhead_seed1/best.pt
```

This checkpoint adds a future-failure risk head. It predicts whether a
`switch_subgoal`-style failure will occur within the next 8 bridge snapshots.

On the merged dataset, all-window metrics are:

```text
event F1: path_invalidated=0.680, navigation_stuck=0.769, low_information_gain=0.772
replan F1: switch_subgoal=0.806
failure_risk_acc=0.839
```

A 5-pair learned-only A/B (`data/ab_runtime_segment_model_v1`) produced no
advisor-control stops. The same seeds with hybrid control
(`data/ab_runtime_segment_hybrid_v1`) reduced steps from 182 to 138 and
navigation timeouts from 15 to 12 via 3 progress-guard stops.

The risk-head model changed this result. With learned `risk` control only
(`data/ab_runtime_segment_riskhead_v1`, no progress guard), the same 5 pairs
reduced steps from 182 to 143, navigation timeouts from 15 to 13, and produced
2 advisor-control stops. This is close to the hand-rule hybrid result while
keeping online intervention learned. The current research conclusion is:

The larger 10-pair check on new seeds is more conservative:

```text
data/ab_runtime_segment_riskhead_v2
off steps=354, replan steps=323
off nav_timeouts=29, replan nav_timeouts=26
advisor_control_stops=3, progress_guard_stops=0
frontier_successes unchanged: 4
target_complete_rate unchanged: 0.1

data/ab_runtime_segment_hybrid_v2
off steps=354, replan steps=294
off nav_timeouts=29, replan nav_timeouts=23
advisor_control_stops=3, progress_guard_stops=3
frontier_successes unchanged: 4
target_complete_rate unchanged: 0.1
```

So the risk-head advisor is now useful online, but the analytic progress reflex
still catches failure modes the learned model misses.

- keep hybrid as the robust fallback for now
- treat the risk-head advisor as the main learned middle-layer direction
- collect more diverse guard-triggered and non-triggered segments to calibrate
  risk thresholds before replacing the analytic reflex

The first real-data labels can be rule-generated:

- `navigation_stuck`: low displacement + low progress over a window
- `low_information_gain`: visited-grid increase below threshold
- `target_candidate_found`: target landmark or VLM semantic candidate appears
- `path_invalidated`: planner cannot produce a valid path
- `need_scan`: new topological node with insufficient visual scans
