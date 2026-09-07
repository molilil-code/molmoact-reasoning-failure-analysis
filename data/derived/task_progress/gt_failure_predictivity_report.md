# Privileged GT failure-predictivity analysis

This report characterizes task-state GT only. It does not evaluate the MolmoAct reasoning predictor and does not define failure onset.
Terminal milestones are explicitly marked as endpoint/leakage-prone; initial geometry is treated as a separate control baseline.

## pick_coke_can

Episodes: 45 (18 success, 27 failure)

| GT signal | Success episodes with signal | Failure episodes with signal | Interpretation |
|---|---:|---:|---|
| `any grasp` | 18/18 | 0/27 | Perfect retrospective milestone; usually late. |
| `any lift` | 18/18 | 0/27 | Same endpoint-adjacent information as grasp. |
| `significant lift` | 10/18 | 0/27 | Incomplete milestone; only a subset of successes. |

Initial object XY radius mean: success=0.578, failure=0.644; exploratory failure AUC=0.757.
This suggests a possible condition-difficulty confound, not a dynamic failure precursor.
First grasp in successful episodes: median=38, range=14–57; no failure episode ever grasps.

### Online checkpoints

Absence is interpreted as a risk rule only for this descriptive table; it is not an onset label.

| Step | Field | Event true in success | Event true in failure | Absence recall for failure | Absence false alarm on success |
|---:|---|---:|---:|---:|---:|
| 10 | `gt_is_grasped` | 0/18 | 0/27 | 100.0% | 100.0% |
| 10 | `gt_lifted_object` | 0/18 | 0/27 | 100.0% | 100.0% |
| 10 | `gt_lifted_object_significantly` | 0/18 | 0/27 | 100.0% | 100.0% |
| 20 | `gt_is_grasped` | 2/18 | 0/27 | 100.0% | 88.9% |
| 20 | `gt_lifted_object` | 2/18 | 0/27 | 100.0% | 88.9% |
| 20 | `gt_lifted_object_significantly` | 1/18 | 0/27 | 100.0% | 94.4% |
| 30 | `gt_is_grasped` | 7/18 | 0/27 | 100.0% | 61.1% |
| 30 | `gt_lifted_object` | 7/18 | 0/27 | 100.0% | 61.1% |
| 30 | `gt_lifted_object_significantly` | 4/18 | 0/27 | 100.0% | 77.8% |
| 40 | `gt_is_grasped` | 9/18 | 0/27 | 100.0% | 50.0% |
| 40 | `gt_lifted_object` | 9/18 | 0/27 | 100.0% | 50.0% |
| 40 | `gt_lifted_object_significantly` | 6/18 | 0/27 | 100.0% | 66.7% |
| 50 | `gt_is_grasped` | 14/18 | 0/27 | 100.0% | 22.2% |
| 50 | `gt_lifted_object` | 14/18 | 0/27 | 100.0% | 22.2% |
| 50 | `gt_lifted_object_significantly` | 7/18 | 0/27 | 100.0% | 61.1% |
| 60 | `gt_is_grasped` | 18/18 | 0/27 | 100.0% | 0.0% |
| 60 | `gt_lifted_object` | 18/18 | 0/27 | 100.0% | 0.0% |
| 60 | `gt_lifted_object_significantly` | 10/18 | 0/27 | 100.0% | 44.4% |
| 70 | `gt_is_grasped` | 18/18 | 0/27 | 100.0% | 0.0% |
| 70 | `gt_lifted_object` | 18/18 | 0/27 | 100.0% | 0.0% |
| 70 | `gt_lifted_object_significantly` | 10/18 | 0/27 | 100.0% | 44.4% |


## move_near

Episodes: 35 (15 success, 20 failure)

| GT signal | Success episodes with signal | Failure episodes with signal | Interpretation |
|---|---:|---:|---|
| `correct-object movement` | 15/15 | 11/20 | Useful progress milestone, but 11 failures also reach it. |
| `near target` | 15/15 | 0/20 | Strong terminal milestone; not early warning. |
| `height violation` | 0/15 | 2/20 | Specific but rare: only 2/20 failures. |

Initial source-target distance mean: success=0.271, failure=0.267; exploratory failure AUC=0.472.
Initial geometry is close to chance in this sample.
First correct movement: success median=31 (range 18–52); the 11 failures that reached it have median=44 (range 23–78).

### Online checkpoints

Absence is interpreted as a risk rule only for this descriptive table; it is not an onset label.

| Step | Field | Event true in success | Event true in failure | Absence recall for failure | Absence false alarm on success |
|---:|---|---:|---:|---:|---:|
| 10 | `gt_moved_correct_obj` | 0/15 | 0/20 | 100.0% | 100.0% |
| 10 | `gt_near_tgt_obj` | 0/15 | 0/20 | 100.0% | 100.0% |
| 20 | `gt_moved_correct_obj` | 3/15 | 0/20 | 100.0% | 80.0% |
| 20 | `gt_near_tgt_obj` | 0/15 | 0/20 | 100.0% | 100.0% |
| 30 | `gt_moved_correct_obj` | 7/15 | 2/20 | 90.0% | 53.3% |
| 30 | `gt_near_tgt_obj` | 4/15 | 0/20 | 100.0% | 73.3% |
| 40 | `gt_moved_correct_obj` | 11/15 | 3/20 | 85.0% | 26.7% |
| 40 | `gt_near_tgt_obj` | 10/15 | 0/20 | 100.0% | 33.3% |
| 50 | `gt_moved_correct_obj` | 14/15 | 6/20 | 70.0% | 6.7% |
| 50 | `gt_near_tgt_obj` | 13/15 | 0/20 | 100.0% | 13.3% |
| 60 | `gt_moved_correct_obj` | 15/15 | 8/20 | 60.0% | 0.0% |
| 60 | `gt_near_tgt_obj` | 15/15 | 0/20 | 100.0% | 0.0% |
| 70 | `gt_moved_correct_obj` | 15/15 | 8/20 | 60.0% | 0.0% |
| 70 | `gt_near_tgt_obj` | 15/15 | 0/20 | 100.0% | 0.0% |


## Conclusion

PickCokeCan GT milestones identify the eventual outcome retrospectively, but they appear too late and are absent throughout every failure episode. MoveNear has a more useful phase split: no correct movement versus transport not completed, with rare height violations. Neither task currently has per-step geometry, so these are milestone/failure-mode analyses rather than continuous task-progress onset labels.

Privileged GT fields should remain outside the main Depth/Trace/Action predictor. They can support task-specific labels, a geometry-only control baseline, or a clearly named privileged-state oracle.
