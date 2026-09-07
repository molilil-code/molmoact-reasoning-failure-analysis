#!/usr/bin/env python3
"""
Inspect available GT fields for task progress definition.

Before implementing onset rules, we need to understand:
- What label_* columns are available?
- What GT signal can we use for each task?
- Are the signals continuous/discrete/binary?
- How complete are they?

This is the first step before designing failure_onset rules.
"""

import pandas as pd
import numpy as np
from collections import defaultdict


def inspect_fields(step_features_csv):
    df = pd.read_csv(step_features_csv)
    
    print("\n" + "=" * 100)
    print("GROUND TRUTH FIELD INSPECTION FOR FAILURE ONSET DEFINITION")
    print("=" * 100)
    
    # Summary
    print(f"\nTotal episodes: {len(df) // 50}")  # rough estimate
    print(f"Total steps: {len(df)}")
    print(f"Tasks: {sorted(df['task'].unique().tolist())}")
    
    # ==================== Per-Task Inspection ====================
    
    for task in sorted(df['task'].unique()):
        sub = df[df['task'] == task]
        n_episodes = len(sub) // 50  # rough estimate
        
        print(f"\n{'=' * 100}")
        print(f"TASK: {task.upper()}")
        print(f"{'=' * 100}")
        print(f"Episodes: ~{n_episodes}   Steps: {len(sub)}")
        
        # Get all label_* columns
        label_cols = [c for c in df.columns if c.startswith('label_')]
        
        if not label_cols:
            print("  [No label_* columns found]")
            continue
        
        # Categorize by availability
        available = []
        sparse = []
        empty = []
        
        for col in label_cols:
            non_null = sub[col].notna().sum()
            pct = 100 * non_null / len(sub)
            
            if pct > 80:
                available.append((col, non_null, pct))
            elif pct > 0:
                sparse.append((col, non_null, pct))
            else:
                empty.append(col)
        
        # Display available fields
        if available:
            print(f"\n  📊 PRIMARY FIELDS (>80% coverage):\n")
            for col, count, pct in sorted(available, key=lambda x: -x[2]):
                col_short = col.replace(f'label_', '').replace(f'{task}_', '').replace('_', ' ')
                
                non_null_vals = sub[col].dropna()
                dtype = sub[col].dtype
                
                print(f"    {col}")
                print(f"      Coverage: {count}/{len(sub)} ({pct:.1f}%)")
                print(f"      Type: {dtype}")
                
                if dtype in ['float64', 'int64']:
                    print(f"      Range: [{non_null_vals.min():.4f}, {non_null_vals.max():.4f}]")
                    print(f"      Mean/Std: {non_null_vals.mean():.4f} / {non_null_vals.std():.4f}")
                elif dtype == 'object':
                    unique_vals = non_null_vals.unique()
                    print(f"      Unique values: {len(unique_vals)}")
                    print(f"      Sample: {unique_vals[:5].tolist()}")
                else:
                    # Binary or boolean
                    print(f"      Unique values: {sorted(non_null_vals.unique())}")
                print()
        
        # Display sparse fields (might still be useful)
        if sparse:
            print(f"\n  ⚠️  SPARSE FIELDS (1-80% coverage):\n")
            for col, count, pct in sorted(sparse, key=lambda x: -x[2]):
                col_short = col.replace(f'label_', '').replace(f'{task}_', '').replace('_', ' ')
                print(f"    {col}: {count}/{len(sub)} ({pct:.1f}%)")
        
        # Display empty fields
        if empty:
            print(f"\n  ❌ EMPTY FIELDS (0% coverage):\n")
            for col in empty[:10]:  # limit display
                col_short = col.replace(f'label_', '').replace(f'{task}_', '')
                print(f"    {col}")
            if len(empty) > 10:
                print(f"    ... and {len(empty) - 10} more")
        
        # Task-specific recommendations
        print(f"\n  💡 RECOMMENDATIONS FOR {task.upper()}:\n")
        
        if task == 'open_drawer':
            print("""    Required for drawer-progress onset:
      ✓ label_drawer_progress (door opening progress)
      
    Current status:""")
            qpos_col = [c for c in available if 'progress' in c[0].lower()]
            if qpos_col:
                print(f"      ✓ FOUND: {qpos_col[0][0]}")
            else:
                print(f"      ✗ NOT FOUND - drawer onset definition BLOCKED")
        
        elif task == 'pick_coke_can':
            print("""    Required for stage-based onset:
      - Approach phase: TCP-to-object distance OR object position
      - Grasp phase: is_grasped / gripper_closed / gripper_state / coke_stage
      - Lift phase: object_lifted / lifted / object_height
      - Success: label_episode_success
      
    Currently available:""")
            needed = ['grasped', 'lifted', 'object_height', 'tcp', 'distance', 'stage', 'coke']
            found_any = False
            for col in available:
                col_lower = col[0].lower()
                if any(n in col_lower for n in needed):
                    print(f"      ✓ {col[0]}")
                    found_any = True
            
            if not found_any:
                print(f"      ✗ No stage indicators found")
                print(f"      → Fallback: Use z-height only (high false-positive risk)")
        
        elif task == 'move_near':
            print("""    Required for transport-progress onset:
      - Approach phase: TCP-to-source distance OR source position
      - Grasp phase: is_grasped / source_grasped / coke_stage
      - Transport phase: source-to-target distance
      - Success: label_episode_success
      
    Currently available:""")
            needed = ['source', 'target', 'object', 'position', 'distance', 'grasped', 'stage', 'coke']
            found_any = False
            for col in available:
                col_lower = col[0].lower()
                if any(n in col_lower for n in needed):
                    print(f"      ✓ {col[0]}")
                    found_any = True
            
            if not found_any:
                print(f"      ✗ No position info found")
                print(f"      → Cannot define meaningful onset")
    
    # ==================== Cross-Task Summary ====================
    
    print(f"\n{'=' * 100}")
    print("CROSS-TASK SUMMARY & FEASIBILITY")
    print(f"{'=' * 100}\n")
    
    feasibility = {}
    
    # OpenDrawer
    progress_available = any('drawer_progress' in c.lower() for c in df.columns if c.startswith('label_'))
    feasibility['open_drawer'] = {
        'status': '✅ READY' if progress_available else '❌ BLOCKED',
        'reason': 'Have drawer_progress signal' if progress_available else 'Missing drawer_progress data',
        'priority': 1
    }
    
    # PickCokeCan
    grasp_available = any(
        any(x in c.lower() for x in ['grasped', 'gripper', 'lifted'])
        for c in df.columns if c.startswith('label_')
    )
    feasibility['pick_coke_can'] = {
        'status': '🟡 PARTIAL' if grasp_available else '🔴 FALLBACK',
        'reason': 'Have stage info' if grasp_available else 'Need z-height fallback (risky)',
        'priority': 2
    }
    
    # MoveNear
    pos_available = any(
        any(x in c.lower() for x in ['position', 'source', 'target', 'distance'])
        for c in df.columns if c.startswith('label_')
    )
    feasibility['move_near'] = {
        'status': '✅ READY' if pos_available else '🔴 UNCERTAIN',
        'reason': 'Have position info' if pos_available else 'Unknown position signal',
        'priority': 3 if pos_available else 99
    }
    
    for task, info in sorted(feasibility.items(), key=lambda x: x[1]['priority']):
        print(f"{info['status']} {task:20s}: {info['reason']}")
    
    # ==================== Implementation Roadmap ====================
    
    print(f"\n{'=' * 100}")
    print("RECOMMENDED IMPLEMENTATION ROADMAP")
    print(f"{'=' * 100}\n")
    
    print("""1. 🟢 START WITH: open_drawer
     Reason: Clearest GT signal (qpos)
     Steps:
       a. Select 5 success + 5 failure episodes
       b. Manually inspect qpos curves
       c. Design onset rule (W=10, eps=0.02)
       d. Verify NO false-positive onsets in success episodes
       e. Freeze parameters
     
2. 🟡 THEN: pick_coke_can
     Decision point:
       IF have stage info (grasped/lifted):
         → Design stage-based onset
       ELSE:
         → Use z-height fallback with caution
       
3. 🟡 FINALLY: move_near
     Decision point:
       IF have position info:
         → Design transport-progress onset
       ELSE:
         → Determine if task-specific onset is feasible
    
⚠️  DO NOT implement all three simultaneously.
    Complete OpenDrawer validation first, then proceed.""")
    
    print(f"\n{'=' * 100}\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = 'data/formal_features/step_features.csv'
    
    try:
        inspect_fields(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
