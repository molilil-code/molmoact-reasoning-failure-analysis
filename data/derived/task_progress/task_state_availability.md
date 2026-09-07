# Task-state GT availability

Source: raw `results/formal/*/condition_*/steps.jsonl` plus episode `summary.json` reset metadata.
This is a raw-state inventory only. Progress transforms, onset labels, predictor features, and model fitting are intentionally undefined.

## pick_coke_can

Episodes: 45  
Steps: 2853  
Success episodes: 18  
Failure/truncated episodes: 27

| Field | Source | Scope | Per-step coverage | Episode coverage | Temporal variation |
|---|---|---|---:|---:|---:|
| `gt_consecutive_grasp` | `steps.post_action.info.consecutive_grasp` | per_step | 100.0% | 100.0% | 0 episodes |
| `gt_elapsed_steps` | `steps.post_action.info.elapsed_steps` | per_step | 100.0% | 100.0% | 45 episodes |
| `gt_is_grasped` | `steps.post_action.info.is_grasped` | per_step | 100.0% | 100.0% | 18 episodes |
| `gt_lifted_object` | `steps.post_action.info.lifted_object` | per_step | 100.0% | 100.0% | 18 episodes |
| `gt_lifted_object_significantly` | `steps.post_action.info.lifted_object_significantly` | per_step | 100.0% | 100.0% | 10 episodes |
| `gt_object_init_x` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_object_init_y` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_object_init_z` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_reward` | `steps.post_action.reward` | per_step | 100.0% | 100.0% | 18 episodes |
| `gt_success` | `steps.post_action.info.success` | per_step | 100.0% | 100.0% | 18 episodes |
| `gt_terminated` | `steps.post_action.terminated` | per_step | 100.0% | 100.0% | 18 episodes |
| `gt_truncated` | `steps.post_action.truncated` | per_step | 100.0% | 100.0% | 27 episodes |

Missing candidate fields:

- `per_step_object_position`
- `per_step_object_z`
- `per_step_tcp_position`
- `per_step_tcp_object_distance`
- `per_step_gripper_state`


## move_near

Episodes: 35  
Steps: 2187  
Success episodes: 15  
Failure/truncated episodes: 20

| Field | Source | Scope | Per-step coverage | Episode coverage | Temporal variation |
|---|---|---|---:|---:|---:|
| `gt_all_obj_keep_height` | `steps.post_action.info.all_obj_keep_height` | per_step | 100.0% | 100.0% | 2 episodes |
| `gt_elapsed_steps` | `steps.post_action.info.elapsed_steps` | per_step | 100.0% | 100.0% | 35 episodes |
| `gt_is_closest_to_tgt` | `steps.post_action.info.is_closest_to_tgt` | per_step | 100.0% | 100.0% | 10 episodes |
| `gt_moved_correct_obj` | `steps.post_action.info.moved_correct_obj` | per_step | 100.0% | 100.0% | 26 episodes |
| `gt_moved_wrong_obj` | `steps.post_action.info.moved_wrong_obj` | per_step | 100.0% | 100.0% | 0 episodes |
| `gt_near_tgt_obj` | `steps.post_action.info.near_tgt_obj` | per_step | 100.0% | 100.0% | 15 episodes |
| `gt_reward` | `steps.post_action.reward` | per_step | 100.0% | 100.0% | 15 episodes |
| `gt_source_init_x` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_source_init_y` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_source_init_z` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_success` | `steps.post_action.info.success` | per_step | 100.0% | 100.0% | 15 episodes |
| `gt_target_init_x` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_target_init_y` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_target_init_z` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_terminated` | `steps.post_action.terminated` | per_step | 100.0% | 100.0% | 15 episodes |
| `gt_truncated` | `steps.post_action.truncated` | per_step | 100.0% | 100.0% | 20 episodes |

Missing candidate fields:

- `per_step_source_object_position`
- `per_step_target_object_position`
- `per_step_object_target_distance`
- `per_step_tcp_position`
- `per_step_tcp_object_distance`
- `per_step_gripper_state`


## open_drawer

Episodes: 20  
Steps: 1316  
Success episodes: 12  
Failure/truncated episodes: 8

| Field | Source | Scope | Per-step coverage | Episode coverage | Temporal variation |
|---|---|---|---:|---:|---:|
| `gt_cabinet_init_x` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_cabinet_init_y` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_cabinet_init_z` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_drawer_init_x` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_drawer_init_y` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_drawer_init_z` | `summary.reset_info` | episode_initial_only | — | 100.0% | 0 episodes |
| `gt_drawer_qpos` | `steps.post_action.info.qpos` | per_step | 100.0% | 100.0% | 20 episodes |
| `gt_elapsed_steps` | `steps.post_action.info.elapsed_steps` | per_step | 100.0% | 100.0% | 20 episodes |
| `gt_reward` | `steps.post_action.reward` | per_step | 100.0% | 100.0% | 12 episodes |
| `gt_success` | `steps.post_action.info.success` | per_step | 100.0% | 100.0% | 12 episodes |
| `gt_terminated` | `steps.post_action.terminated` | per_step | 100.0% | 100.0% | 12 episodes |
| `gt_truncated` | `steps.post_action.truncated` | per_step | 100.0% | 100.0% | 8 episodes |

Missing candidate fields:

- `per_step_tcp_position`
- `per_step_handle_position`

