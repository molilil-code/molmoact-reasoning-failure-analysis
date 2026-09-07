# Task-state and GT progress outputs

These files are derived from `results/formal/*/condition_*/steps.jsonl` and the
episode `summary.json` reset metadata. They are privileged environment state
artifacts for task-progress and failure-mode analysis only.

They are not predictor features. No `Depth`, `Trace`, or `Action` signal is
added, and no failure-onset label or threshold is defined here.

- `pick_coke_task_state.csv`, `move_near_task_state.csv`, `open_drawer_task_state.csv`: raw `gt_*` fields.
- `pick_coke_gt_progress.csv`, `move_near_gt_progress.csv`: milestone/streak/failure-mode derivations.
- `pick_coke_gt_episode_summary.csv`, `move_near_gt_episode_summary.csv`: one row per episode.
- `task_state_inventory.json`, `task_state_availability.md`: field coverage and missing dynamic-state audit.
- `dynamic_raw_state_search.json`: schema search for per-step object/TCP/source/target geometry.
- `task_gt_progress_metadata.json`: derivation scope and aggregate counts.

Initial pose columns are explicitly episode-constant (`*_init_*`); they are not
per-step trajectories.
