#!/usr/bin/env python3
"""Debug script to analyze episode data."""

import torch
from pathlib import Path
from collections import defaultdict

def analyze_episodes(rollout_dir: str):
    """Analyze episode structure in saved rollouts."""
    rollout_path = Path(rollout_dir)
    files = sorted([f for f in rollout_path.glob("*.pt")])
    
    print(f"Total files: {len(files)}")
    print(f"Analyzing from: {rollout_dir}\n")
    
    episode_count = 0
    current_episode_steps = 0
    episode_lengths = []
    done_count = 0
    success_count = 0
    
    per_goal_episodes = defaultdict(int)
    
    for i, file in enumerate(files):
        try:
            data = torch.load(file, weights_only=False)
            info = data['info']
            
            current_episode_steps += 1
            
            # Check if episode is done
            done = info.get('done', False)
            if isinstance(done, torch.Tensor):
                done = done.item() if done.numel() == 1 else done[0].item()
            
            if done:
                done_count += 1
                episode_count += 1
                episode_lengths.append(current_episode_steps)
                
                # Check success
                success = info.get('success', False)
                if isinstance(success, torch.Tensor):
                    success = success.item() if success.numel() == 1 else success[0].item()
                if success:
                    success_count += 1
                
                # Track goal
                goal_name = info.get('goal_names', 'unknown')
                per_goal_episodes[goal_name] += 1
                
                print(f"Episode {episode_count}: {current_episode_steps} steps, "
                      f"done={done}, success={success}, goal={goal_name}")
                
                current_episode_steps = 0
                
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total steps/files:     {len(files)}")
    print(f"Total episodes (done): {episode_count}")
    print(f"Success count:         {success_count}")
    print(f"Incomplete steps:      {current_episode_steps}")
    
    if episode_lengths:
        import numpy as np
        print(f"\nEpisode lengths:")
        print(f"  Mean: {np.mean(episode_lengths):.1f}")
        print(f"  Min:  {np.min(episode_lengths)}")
        print(f"  Max:  {np.max(episode_lengths)}")
    
    if per_goal_episodes:
        print(f"\nEpisodes per goal:")
        for goal, count in sorted(per_goal_episodes.items()):
            print(f"  {goal}: {count}")

if __name__ == "__main__":
    analyze_episodes("eval_runs/barlow_twins_100eps")
