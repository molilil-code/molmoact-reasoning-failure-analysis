# GitHub release manifest

This repository should publish the reproducible analysis code and a curated
set of derived tables and figures. Raw rollout artifacts are intentionally
kept local.

## Push by default

- `readme.md`
- `experiment_design.md`
- `log.md`
- `analysis/`
- `src/`
- `experiments/`
- `configs/`

The new verification script is `analysis/verify_opendrawer_depth_rules.py`.

`scripts/` can be included after replacing machine-specific checkpoint and
output paths with command-line arguments or environment variables. Otherwise,
leave the collection scripts out of the teacher-facing release.

## Curated data to include

### Feature tables and metadata

```text
data/formal_features/feature_metadata.json
data/formal_features/episode_features.csv
data/formal_features/step_features_trace_depth.csv
data/formal_features/step_features_trace_depth.metadata.json
data/formal_features/step_features_drawer_geometry.csv
data/formal_features/step_features_drawer_geometry.metadata.json
data/formal_features/trace_depth_episode_associations.csv
data/formal_features/trace_depth_opendrawer_oof.csv
data/formal_features/trace_depth_predictivity_report.md
```

These are the derived features used by the main Depth/Trace analyses. The
duplicate backup file `step_features.csv.dupbackup` is not part of the release.
The larger base `step_features.csv` is optional; include it only if the
teacher needs to reproduce every feature from the raw formal rollouts.

### OpenDrawer evidence tables

```text
data/opendrawer_analysis/README.md
data/opendrawer_analysis/opendrawer_analysis_metadata.json
data/opendrawer_analysis/opendrawer_onsets.csv
data/opendrawer_analysis/opendrawer_onset_validation.csv
data/opendrawer_analysis/opendrawer_depth_dynamics_episode.csv
data/opendrawer_analysis/opendrawer_depth_dynamics_failure_modes.csv
data/opendrawer_analysis/opendrawer_depth_dynamics_report.md
data/opendrawer_analysis/opendrawer_amplitude_dynamics_episode.csv
data/opendrawer_analysis/opendrawer_amplitude_dynamics_group_summary.csv
data/opendrawer_analysis/opendrawer_amplitude_dynamics_report.md
data/opendrawer_analysis/opendrawer_depth_rule_episode.csv
data/opendrawer_analysis/opendrawer_depth_rule_metric_summary.csv
data/opendrawer_analysis/opendrawer_depth_rule_stats.csv
data/opendrawer_analysis/opendrawer_depth_rule_verification_report.md
data/opendrawer_analysis/failure_mode_cluster_assignments.csv
data/opendrawer_analysis/failure_mode_cluster_centroids.csv
data/opendrawer_analysis/failure_mode_cluster_quality.csv
data/opendrawer_analysis/failure_mode_clustering_report.md
data/opendrawer_analysis/opendrawer_action_mechanism_group_summary.csv
data/opendrawer_analysis/opendrawer_action_mechanism_failure_mode_summary.csv
data/opendrawer_analysis/opendrawer_peak_interval_episode.csv
data/opendrawer_analysis/ablation_results_opendrawer.csv
data/opendrawer_analysis/modality_results_opendrawer.csv
data/opendrawer_analysis/risk_score_distributions_opendrawer.csv
data/opendrawer_analysis/opendrawer_oof_scores.csv
data/opendrawer_analysis/online_labels_opendrawer.csv
```

The `.txt` versions of the ablation and modality tables are duplicate exports
and can be omitted.

### Representative figures

Keep figures that communicate the main story rather than every diagnostic:

```text
data/opendrawer_analysis/opendrawer_depth_rule_verification.png
data/opendrawer_analysis/opendrawer_depth_dynamics_classification.png
data/opendrawer_analysis/opendrawer_amplitude_dynamics.png
data/opendrawer_analysis/opendrawer_onset_validation.png
data/opendrawer_analysis/opendrawer_drawer_max_qpos_progress.png
data/opendrawer_analysis/opendrawer_qpos_trajectories.png
data/opendrawer_analysis/opendrawer_temporal_change_ratio_trajectories.png
data/opendrawer_analysis/opendrawer_model_action_jump_trajectories.png
data/opendrawer_analysis/opendrawer_endpoint_drift_trajectories.png
data/opendrawer_analysis/opendrawer_trace_features_all_episodes.png
data/opendrawer_analysis/failure_mode_cluster_map.png
data/opendrawer_analysis/risk_score_distributions_opendrawer.png
data/opendrawer_analysis/modality_risk_score_distributions_opendrawer.png
```

The episode-facet trace figure and raw per-step trajectory exports are useful
for local inspection but are optional for the public repository.

## Optional cross-task progress package

Include this package if the repository should show that GT extraction was also
checked for PickCokeCan and MoveNear:

```text
data/derived/task_progress/README.md
data/derived/task_progress/task_state_availability.md
data/derived/task_progress/task_state_inventory.json
data/derived/task_progress/task_gt_progress_metadata.json
data/derived/task_progress/pick_coke_gt_episode_summary.csv
data/derived/task_progress/pick_coke_gt_predictivity_episode.csv
data/derived/task_progress/pick_coke_gt_progress.csv
data/derived/task_progress/move_near_gt_episode_summary.csv
data/derived/task_progress/move_near_gt_predictivity_episode.csv
data/derived/task_progress/move_near_gt_progress.csv
data/derived/task_progress/gt_predictivity_checkpoints.csv
data/derived/task_progress/gt_failure_predictivity_report.md
```

The full `*_task_state.csv` files are optional because they are privileged
per-step state tables and are not predictor inputs.

## Do not push

- `results/` raw rollout directories, `steps.jsonl`, rollout JSON, videos,
  and repeated backend outputs;
- `logs/`, `outputs/`, `models/`, `rollouts/`, and checkpoint files;
- `.claude/`, `_probe_tmp.py`, and temporary or duplicate files;
- `data/formal_features/step_features.csv.dupbackup`;
- `data/opendrawer_analysis/amplitude_eps01/` and `amplitude_eps02/` unless
  epsilon sensitivity is explicitly discussed;
- raw `*_task_state.csv` files unless a detailed GT audit is needed.

## Staging note

The current `.gitignore` excludes all of `data/` and `results/`. Curated files
must therefore be staged explicitly with `git add -f`; do not use `git add -f
data results` because that would reintroduce raw rollouts.
