# OpenDrawer onset-aware analysis

Reproduce all artifacts with:

```text
python analysis/opendrawer_failure_onset.py
```

The frozen onset rule is best-so-far progress plateau detection on
`label_drawer_progress`, with `W=10`, `eps=0.02`, and `min_start=20`. The
future-recovery check prevents a temporary pause from being called a failure
onset. Steps at or after `t_onset` have `valid_mask=0`; they are excluded from
online training rather than relabelled as negatives.

The validation artifact contains 5 success and 5 failure episodes. All 10
initial curve-review annotations agree with the frozen rule, and none of the
12 successful episodes receives an automatic onset.

The A0--A4 table uses episode-grouped out-of-fold logistic regression at
`H=5, 10, 20`. The formal feature table has no `action_logprob` column, so A0 is
explicitly named `A0_action_magnitude_proxy` and uses only
`feat_model_translation_norm`; this is not a claim that token log-probability
was available. The table reports episode AUROC under `max`, `mean`, and
`last` aggregation. `max_valid` is the retrospective any-alert score; it is
not used to claim an H-step lead time. Episode thresholds are searched on the
success episodes and constrained to empirical `FAR_episode <= 10%` (with 12
success episodes this permits at most one false alarm).

Because the online positives are sparse, each result also reports step-level
and episode-level AUPRC with the corresponding positive-prevalence baseline.
Confidence intervals are 95% percentile bootstrap intervals resampling whole
episodes, not individual correlated steps.

For the operational metrics, a failure is detected only when an alert falls
in `[t_onset-H, t_onset-1]`. Alerts earlier than that window are reported as
premature alerts and are not counted as successful H-window detections. Lead
time is computed only from alerts inside that window. Thresholds are selected
with inner OOF predictions inside each outer training fold.

Premature-alert summaries use one first threshold crossing per episode for
episode-level rates and distances. Repeated crossings from the same episode
remain available in the event-level detail file and are not mixed into the
episode denominator.

Files:

* `opendrawer_onset_validation.csv` and `.png`: curve-review validation;
* `opendrawer_onsets.csv`: onset for all 20 episodes;
* `online_labels_opendrawer.csv`: H=5, 10, 20 labels and masks;
* `ablation_results_opendrawer.csv` / `.txt`: A0--A4 metrics;
* `opendrawer_oof_scores.csv`: fold-held-out scores and risk groups;
* `risk_score_distributions_opendrawer.csv` / `.png`: success, pre-window, and H-window score distributions;
* `modality_results_opendrawer.csv`: formal Action / Action+Depth / Action+Trace / Action+Depth+Trace / Depth+Trace ablation;
* `risk_trajectories_modality_H*.png`: episode-level OOF trajectories with onset, onset-H, and fold threshold;
* `premature_alert_distances_opendrawer.csv` and summary: distances from premature alerts to onset;
* `premature_episode_first_crossing_opendrawer.csv`: one first premature crossing per failure episode;
* `detector_results_opendrawer.csv`: fixed T0 raw, T1 causal rolling-mean, and T2 consecutive-step detectors;
* `opendrawer_analysis_metadata.json`: parameters and provenance.

## Raw trajectory figures

Generate the six descriptive plots with:

```text
python analysis/plot_opendrawer_raw_trajectories.py
```

Outputs use shared y-axis ranges within each metric and show the independently
defined operational onset (red) and onset-minus-10 (orange dotted):

* `opendrawer_qpos_trajectories.png`;
* `opendrawer_temporal_change_ratio_trajectories.png`;
* `opendrawer_trace_num_points_trajectories.png`;
* `opendrawer_endpoint_drift_trajectories.png`;
* `opendrawer_model_action_jump_trajectories.png`;
* `opendrawer_drawer_max_qpos_progress.png`.

These figures are descriptive only. The task-state qpos/progress curves are
retrospective GT signals and are not predictor inputs; the reasoning-feature
curves are not used to redefine onset or select a threshold.

To plot the three Trace geometry trajectories for all 20 OpenDrawer episodes:

```text
python analysis/plot_opendrawer_trace_features.py
```

This writes `opendrawer_trace_features_all_episodes.png` (success/failure
overlaid) and `opendrawer_trace_features_episode_facets.png` (one row per
episode). Orange markers/lines show the independently computed operational
onsets for failure episodes.

For the action-mechanism check, run:

```text
python analysis/analyze_opendrawer_action_mechanism.py
```

This decomposes full action jump into translation jump, rotation jump, and
gripper change, and writes `opendrawer_action_mechanism_episode.csv`,
`opendrawer_action_mechanism_steps.csv`,
`opendrawer_action_mechanism_group_summary.csv`,
`opendrawer_action_mechanism_failure_mode_summary.csv`, and two diagnostic
figures.
The logged action is a delta-pose command under a PD controller; it is not a
force measurement. No wrench/contact-force signal is currently stored.

Drawer-specific reasoning features are extracted with:

```text
python analysis/add_drawer_geometry_features.py \
  --step-features data/formal_features/step_features.csv \
  --output data/formal_features/step_features_drawer_geometry.csv \
  --ckpt-path <path-to-vae-final.pt> \
  --molmoact-scripts <path-to-molmoact-scripts>
```

The output contains Trace backtracking ratio, canonical-direction alignment,
Trace-path depth sign changes, endpoint-vs-border depth contrast, and endpoint
vs. interior-Trace-point depth contrast (the latter is a drawer-front proxy,
not a GT handle mask). The canonical direction is currently estimated from
successful OpenDrawer traces for exploratory use; formal OOF evaluation must
re-estimate it within each training fold.

T0/T1/T2 share the fold-specific raw-score threshold. No smoothing window or
consecutive-step parameter is tuned on held-out episodes.

## Failure-only clustering (exploratory)

Run:

```text
python analysis/cluster_opendrawer_failures.py
```

This is a separate retrospective Path-A analysis. It clusters only the eight
failure episodes using episode-level mean and p90 aggregates of Depth, Trace,
and Action instability signals. It does not use success labels, task-state GT,
post-action fields, or onset timestamps as clustering inputs, and it does not
claim online prediction. The current K=2 partition is intentionally reported
as exploratory because the sample is small and leave-one-episode-out
stability is weak.

Outputs:

* `failure_mode_cluster_assignments.csv`: one row per failure episode;
* `failure_mode_cluster_episode_aggregates.csv`: input episode aggregates;
* `failure_mode_cluster_family_scores.csv`: standardized family scores;
* `failure_mode_cluster_centroids.csv` and `failure_mode_cluster_quality.csv`;
* `failure_mode_clustering_report.md` and `failure_mode_cluster_map.png`.

## Partial-progress stall onset (Trace-only exploratory analysis)

Run:

```text
python analysis/analyze_partial_progress_stall_onset.py
```

This analysis is restricted to the four episodes currently labelled
`partial_progress_stall` (C00, C02, C09, C10). It aligns the independently
computed operational onset and examines only:

* `feat_trace_direction_change_prev_rad`;
* `feat_trace_straightness`;
* `feat_trace_backtracking_ratio`;
* `feat_trace_direction_alignment`.

Outputs are `partial_progress_stall_onset_aligned.csv`,
`partial_progress_stall_onset_episode_summary.csv`,
`partial_progress_stall_onset_feature_summary.csv`,
`partial_progress_stall_trace_onset.png`, and
`partial_progress_stall_onset_report.md`. The four episodes are heterogeneous;
this output is descriptive and does not define a final online threshold.

## Depth-change dynamics classification without onset

Run:

```text
python analysis/classify_opendrawer_depth_dynamics.py
```

This computes four episode-level descriptors from the complete
`feat_depth_change_ratio_prev` trajectory: low-change fraction, longest
low-change run ratio, turning rate, and total variation. It writes
`opendrawer_depth_dynamics_episode.csv`,
`opendrawer_depth_dynamics_failure_modes.csv`,
`opendrawer_depth_dynamics_classification.png`, and
`opendrawer_depth_dynamics_report.md`. The stagnation/oscillation labels are
exploratory and use no onset timestamps.

Peak intervals are computed with:

```text
python analysis/calculate_opendrawer_peak_interval.py
```

The raw `peak_interval_median` uses every interior local maximum. The output
also includes prominence-filtered variants (`0.05` and `0.10`) to separate
small-amplitude wiggles from larger peaks.

## Amplitude-thresholded turning and prominent peaks

Run:

```text
python analysis/analyze_opendrawer_amplitude_dynamics.py --epsilon 0.05
```

This is an onset-free, episode-level analysis of
`feat_depth_change_ratio_prev`. A turning event requires a sign reversal
between adjacent changes and requires both change magnitudes to exceed
`epsilon`. The primary turning rate is conditional on amplitude-qualified
pairs; an all-pairs rate, prominence-filtered peak rate, and median
inter-peak interval are also reported. Outputs are
`opendrawer_amplitude_dynamics_episode.csv`,
`opendrawer_amplitude_dynamics_group_summary.csv`,
`opendrawer_amplitude_dynamics.png`, and
`opendrawer_amplitude_dynamics_report.md`.

With the current 8-failure/12-success sample and `epsilon=0.05`, failures
show a slightly higher qualified-pair turning rate, but the difference is not
significant. They show fewer prominent peaks per ten steps and slightly longer
prominent inter-peak intervals. Thus the visual high-frequency wiggles are
mostly small-amplitude noise rather than a validated failure-specific burst
signature. The all-pairs rate is lower for failures because their long
low-change tails reduce the event density; it is a different denominator and
should not be conflated with the conditional turning rate.

## Verification of two exploratory episode rules

Run:

```text
python analysis/verify_opendrawer_depth_rules.py
```

The script verifies the rules `episode median < 0.22` and
`fraction(depth-change ratio < 0.10) > 0.14` on all 20 complete episodes.
Both rules currently separate the eight failures from the twelve successes
perfectly (8/8 failure recall and 12/12 success specificity). This is a
retrospective, full-episode result: the rules use the whole trajectory and do
not define an online onset. The two metrics are also correlated descriptions
of the same low-change/stagnation behavior, not independent evidence.
