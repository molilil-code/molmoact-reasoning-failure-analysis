# Trace–Depth feature predictivity audit

The feature definitions are decoder-based and online-safe: current/past pre-action Depth and Trace only.
This report is exploratory. It does not change the frozen OpenDrawer formal ablation and does not add GT task-state fields to the predictor.

## Coverage

- Endpoint depth percentile is available on 99.8% of 6,356 steps.
- Path statistics (std/gradient/range) are available on 89.0%; 362 steps have single-point traces and are intentionally NaN for path variation.
- Fourteen malformed depth-token rows are excluded by the existing decoder extraction metadata.

## Episode-level association (retrospective)

| Task | Feature | Success mean | Failure mean | Cohen d (failure-success) | Episode AUC for failure |
|---|---|---:|---:|---:|---:|
| move_near | `feat_trace_depth_std_norm` | 0.3544 | 0.3442 | -0.082 | 0.480 |
| move_near | `feat_trace_depth_gradient_norm` | 0.0996 | 0.0975 | -0.060 | 0.497 |
| move_near | `feat_trace_depth_range_norm` | 1.1104 | 1.0613 | -0.149 | 0.467 |
| move_near | `feat_trace_endpoint_depth_percentile` | 0.5023 | 0.5628 | 0.531 | 0.647 |
| move_near | `feat_trace_endpoint_depth_percentile_drift_prev` | 0.0317 | 0.0505 | 0.531 | 0.717 |
| open_drawer | `feat_trace_depth_std_norm` | 0.3708 | 0.3021 | -0.787 | 0.271 |
| open_drawer | `feat_trace_depth_gradient_norm` | 0.1069 | 0.0709 | -1.798 | 0.083 |
| open_drawer | `feat_trace_depth_range_norm` | 1.2512 | 0.9904 | -0.923 | 0.240 |
| open_drawer | `feat_trace_endpoint_depth_percentile` | 0.5126 | 0.5237 | 0.132 | 0.448 |
| open_drawer | `feat_trace_endpoint_depth_percentile_drift_prev` | 0.0930 | 0.0460 | -1.696 | 0.125 |
| pick_coke_can | `feat_trace_depth_std_norm` | 0.2726 | 0.2701 | -0.027 | 0.498 |
| pick_coke_can | `feat_trace_depth_gradient_norm` | 0.0826 | 0.0872 | 0.157 | 0.537 |
| pick_coke_can | `feat_trace_depth_range_norm` | 0.9337 | 0.9165 | -0.052 | 0.465 |
| pick_coke_can | `feat_trace_endpoint_depth_percentile` | 0.4925 | 0.4199 | -0.623 | 0.321 |
| pick_coke_can | `feat_trace_endpoint_depth_percentile_drift_prev` | 0.0714 | 0.0459 | -0.826 | 0.265 |

These whole-episode means are descriptors, not online prediction scores; they can use information after the eventual failure and should not be interpreted as causal precursors.

## OpenDrawer grouped OOF checkpoint results

| H | Feature set | Coverage | Step AUROC | Step AUPRC | Prevalence baseline | ΔAUPRC |
|---:|---|---:|---:|---:|---:|---:|
| 5 | `feat_trace_depth_std_norm` | 100.0% | 0.497 | 0.053 | 0.057 | -0.004 |
| 5 | `feat_trace_depth_gradient_norm` | 100.0% | 0.513 | 0.055 | 0.057 | -0.002 |
| 5 | `feat_trace_depth_range_norm` | 100.0% | 0.492 | 0.053 | 0.057 | -0.004 |
| 5 | `feat_trace_endpoint_depth_percentile` | 100.0% | 0.313 | 0.039 | 0.057 | -0.018 |
| 5 | `feat_trace_endpoint_depth_percentile_drift_prev` | 97.2% | 0.473 | 0.052 | 0.058 | -0.006 |
| 5 | `TraceDepthCombined` | 97.2% | 0.456 | 0.051 | 0.058 | -0.008 |
| 10 | `feat_trace_depth_std_norm` | 100.0% | 0.432 | 0.092 | 0.114 | -0.021 |
| 10 | `feat_trace_depth_gradient_norm` | 100.0% | 0.451 | 0.096 | 0.114 | -0.018 |
| 10 | `feat_trace_depth_range_norm` | 100.0% | 0.453 | 0.096 | 0.114 | -0.018 |
| 10 | `feat_trace_endpoint_depth_percentile` | 100.0% | 0.374 | 0.086 | 0.114 | -0.028 |
| 10 | `feat_trace_endpoint_depth_percentile_drift_prev` | 97.2% | 0.486 | 0.108 | 0.117 | -0.009 |
| 10 | `TraceDepthCombined` | 97.2% | 0.447 | 0.098 | 0.117 | -0.019 |
| 20 | `feat_trace_depth_std_norm` | 100.0% | 0.428 | 0.190 | 0.227 | -0.037 |
| 20 | `feat_trace_depth_gradient_norm` | 100.0% | 0.416 | 0.184 | 0.227 | -0.043 |
| 20 | `feat_trace_depth_range_norm` | 100.0% | 0.416 | 0.182 | 0.227 | -0.046 |
| 20 | `feat_trace_endpoint_depth_percentile` | 100.0% | 0.260 | 0.149 | 0.227 | -0.078 |
| 20 | `feat_trace_endpoint_depth_percentile_drift_prev` | 97.2% | 0.489 | 0.207 | 0.228 | -0.021 |
| 20 | `TraceDepthCombined` | 97.2% | 0.450 | 0.200 | 0.228 | -0.028 |

In this small 20-episode OpenDrawer OOF check, the new Trace–Depth features alone do not consistently exceed the prevalence baseline. They are therefore promising feature candidates, not yet evidence of a useful standalone predictor.

## Recommended feature families

1. Path geometry: normalized depth std, gradient, range.
2. Endpoint semantics: endpoint depth percentile and its causal drift.
3. Temporal consistency: profile-level Trace–Depth drift, signed endpoint-start depth change, monotonicity, and local extrema count.
4. Missingness/degeneracy indicators: single-point Trace, malformed depth row, and invalid decoder alignment.

Only the first two families are currently implemented. Raw decoder depth diagnostics should remain exploratory rather than primary predictor inputs.
