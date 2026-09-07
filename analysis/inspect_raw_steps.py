#!/usr/bin/env python3
"""
Inspect raw steps.jsonl to find TRUE GT fields.

NOT the extracted step_features.csv (which may have data pollution),
but the original rollout logs where actions and observations are recorded.
"""

import json
from pathlib import Path

def inspect_raw_steps(task, condition_id=0):
    """Inspect one episode's raw steps.jsonl"""
    
    steps_file = Path(f'results/formal/{task}/condition_{condition_id:04d}/steps.jsonl')
    
    if not steps_file.exists():
        print(f"File not found: {steps_file}")
        return
    
    print(f"\n{'='*100}")
    print(f"RAW STEPS.JSONL INSPECTION: {task} / condition_{condition_id:04d}")
    print(f"{'='*100}\n")
    
    # Read first 3 steps to understand structure
    with open(steps_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            
            step = json.loads(line)
            print(f"Step {i}:")
            print(f"  Top-level keys: {list(step.keys())}")
            
            # Inspect post_action (the state AFTER action was executed)
            if 'post_action' in step:
                post = step['post_action']
                print(f"\n  post_action structure:")
                print(f"    Type: {type(post)}")
                if isinstance(post, dict):
                    print(f"    Keys: {list(post.keys())}")
                    
                    # Look for GT signals
                    for key in post.keys():
                        if any(x in key.lower() for x in ['state', 'info', 'obs', 'grasp', 'lifted', 'progress', 'qpos', 'position']):
                            val = post[key]
                            if isinstance(val, dict):
                                print(f"\n    [{key}] (dict with keys: {list(val.keys())})")
                                # Sample a few values
                                for k, v in list(val.items())[:3]:
                                    print(f"      {k}: {v}")
                            elif isinstance(val, (list, tuple)):
                                print(f"\n    [{key}] (sequence, len={len(val)})")
                                if len(val) > 0:
                                    print(f"      sample: {val[:2]}")
                            else:
                                print(f"\n    [{key}] = {val}")
            
            print("\n" + "-"*100 + "\n")
    
    # Also check summary.json for episode-level metadata
    summary_file = Path(f'results/formal/{task}/condition_{condition_id:04d}/summary.json')
    if summary_file.exists():
        print(f"\nSummary.json contents:")
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            print(json.dumps(summary, indent=2)[:1000])  # First 1000 chars
    
    print(f"\n{'='*100}\n")


if __name__ == '__main__':
    # Inspect one episode from each task
    for task in ['open_drawer', 'pick_coke_can', 'move_near']:
        print(f"\n\n{'#'*100}")
        print(f"# {task.upper()}")
        print(f"{'#'*100}")
        inspect_raw_steps(task, condition_id=0)
