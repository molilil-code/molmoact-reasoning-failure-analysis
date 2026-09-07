#!/usr/bin/env python3
"""
Deep analysis of label_coke_stage across tasks.

Why this matters:
  - OpenDrawer has clear label_drawer_progress
  - But both PickCokeCan and MoveNear have label_coke_stage
  - Need to understand: Is coke_stage task-specific? Or universal?
  - This determines if we can use it for onset definition

Key questions:
  1. What does coke_stage=[0,1,2,3,4] represent?
  2. Does it have different meanings in different tasks?
  3. Can we use it to detect "stalled progress"?
  4. Why does MoveNear have coke_stage at all?
"""

import pandas as pd
import numpy as np
from collections import defaultdict


def analyze_coke_stage(step_features_csv):
    df = pd.read_csv(step_features_csv)
    
    print("\n" + "=" * 100)
    print("DEEP ANALYSIS: label_coke_stage ACROSS TASKS")
    print("=" * 100)
    
    # ==================== Per-Task Analysis ====================
    
    for task in sorted(df['task'].unique()):
        sub = df[df['task'] == task]
        
        print(f"\n{'=' * 100}")
        print(f"TASK: {task.upper()}")
        print(f"{'=' * 100}")
        
        # Overall distribution
        print(f"\nOverall distribution of label_coke_stage:")
        print(sub['label_coke_stage'].value_counts().sort_index())
        print(f"  Mean: {sub['label_coke_stage'].mean():.4f}")
        print(f"  Std:  {sub['label_coke_stage'].std():.4f}")
        
        # Analyze by success/failure
        print(f"\n--- Success vs Failure ---\n")
        
        for success in [0, 1]:
            subset = sub[sub['label_episode_success'] == success]
            if len(subset) > 0:
                label_text = "SUCCESS" if success == 1 else "FAILURE"
                print(f"{label_text}:")
                print(f"  Count: {len(subset)} steps (~{len(subset)//50} episodes)")
                print(f"  Coke stage distribution:")
                dist = subset['label_coke_stage'].value_counts().sort_index()
                for stage, count in dist.items():
                    pct = 100 * count / len(subset)
                    print(f"    stage {int(stage)}: {count:4d} ({pct:5.1f}%)")
                print(f"  Mean stage: {subset['label_coke_stage'].mean():.3f}")
                print()
        
        # Analyze progression within episodes
        print(f"\n--- Stage Progression Within Episodes ---\n")
        
        # Group by episode
        for episode_id in sub['condition_id'].unique()[:3]:  # Show first 3 episodes
            episode = sub[sub['condition_id'] == episode_id].sort_values('step_id')
            success = episode['label_episode_success'].iloc[0]
            label_text = "✓ SUCCESS" if success == 1 else "✗ FAILURE"
            
            stages = episode['label_coke_stage'].values
            ep_len = len(stages)
            
            print(f"Episode {episode_id:04d} ({label_text}, {ep_len} steps):")
            print(f"  Stages: {list(stages[:20])}" + ("..." if ep_len > 20 else ""))
            
            # Find first stage transition
            stage_changes = np.where(np.diff(stages) != 0)[0]
            if len(stage_changes) > 0:
                first_change = stage_changes[0]
                print(f"  First stage change at step {first_change}: {stages[first_change]} → {stages[first_change+1]}")
                print(f"    ({100*first_change/ep_len:.1f}% through episode)")
            else:
                print(f"  No stage changes (stuck at stage {stages[0]})")
            
            # Find max stage reached
            max_stage = np.max(stages)
            print(f"  Max stage reached: {int(max_stage)}")
            print()
    
    # ==================== Cross-Task Comparison ====================
    
    print(f"\n{'=' * 100}")
    print("CROSS-TASK COMPARISON")
    print(f"{'=' * 100}\n")
    
    print("Stage 4 (likely 'success') distribution:")
    for task in sorted(df['task'].unique()):
        sub = df[df['task'] == task]
        stage4_pct = 100 * (sub['label_coke_stage'] == 4).sum() / len(sub)
        print(f"  {task:20s}: {stage4_pct:5.1f}% (n={len(sub)})")
    
    print("\nCorrelation: label_coke_stage vs label_episode_success")
    for task in sorted(df['task'].unique()):
        sub = df[df['task'] == task]
        corr = sub['label_coke_stage'].corr(sub['label_episode_success'])
        print(f"  {task:20s}: r = {corr:+.4f}")
    
    # ==================== Hypothesis Testing ====================
    
    print(f"\n{'=' * 100}")
    print("HYPOTHESIS: Is coke_stage a universal task progression encoding?")
    print(f"{'=' * 100}\n")
    
    print("""Possible interpretations:
    
1. Task-specific milestone:
   OpenDrawer: stage = drawer open progress level?
   PickCokeCan: stage = can grasp progress level?
   MoveNear: stage = ... ??? (contradictory)
   
2. Universal success indicator:
   0 = not attempted / not grasped
   1 = grasped / contacted
   2 = lifted / in progress
   3 = lifted significantly / major progress
   4 = success
   
3. Task-agnostic timestep encoding:
   Just a marker of which phase of execution we're in?
   
4. Data collection artifact:
   All tasks assigned same stage field, even if not meaningful?

EVIDENCE TO COLLECT:
  - Do success episodes consistently reach stage 4?
  - Do failure episodes get stuck at low stages?
  - Does stage follow a monotonic increasing pattern?
  - What's the typical stage_progression_rate (stages/step)?
""")
    
    print("\nSTAGE MONOTONICITY CHECK:")
    for task in sorted(df['task'].unique()):
        sub = df[df['task'] == task]
        
        non_monotonic_count = 0
        for episode_id in sub['condition_id'].unique():
            episode = sub[sub['condition_id'] == episode_id]['label_coke_stage'].values
            diffs = np.diff(episode)
            if np.any(diffs < -0.1):  # allow small numerical noise
                non_monotonic_count += 1
        
        pct = 100 * non_monotonic_count / len(sub['condition_id'].unique())
        print(f"  {task:20s}: {non_monotonic_count} / {len(sub['condition_id'].unique())} episodes have backward stage transitions ({pct:.1f}%)")
    
    # ==================== Recommendations ====================
    
    print(f"\n{'=' * 100}")
    print("RECOMMENDATIONS FOR ONSET DEFINITION")
    print(f"{'=' * 100}\n")
    
    print("""Based on label_coke_stage inspection:

For PickCokeCan:
  ✓ If stage has clear progression → can use for onset
  Rules:
    onset = first time stage stays constant for W steps
    (similar to drawer_progress, but discrete)

For MoveNear:
  ? If coke_stage is meaningful → might be usable
  ? If not → must find alternative GT signal
  
  Check: Are most successful move_near episodes reaching stage 4?
  If YES → can use stage-based onset
  If NO  → stage is not task-progress indicator

For OpenDrawer:
  ✓ Use drawer_progress (already clear)
  
NEXT STEPS:
  1. Determine if stage progression is monotonic
  2. Check if stage=4 ↔ success (for each task)
  3. Decide onset rule based on findings
  4. Validate on 5+5 episodes manually
""")
    
    print("\n" + "=" * 100 + "\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = 'data/formal_features/step_features.csv'
    
    try:
        analyze_coke_stage(csv_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
