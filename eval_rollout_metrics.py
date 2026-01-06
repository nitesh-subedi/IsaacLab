#!/usr/bin/env python3
"""
Script to calculate evaluation metrics from rollout data.
Computes success rate and success path length from saved rollout files.
"""

import torch
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def load_rollouts(rollout_dir: str) -> List[Dict]:
    """Load all rollout files from a directory."""
    rollout_path = Path(rollout_dir)
    files = sorted([f for f in rollout_path.glob("*.pt")])

    print(f"Loading {len(files)} files from {rollout_dir}...")

    rollouts = []
    for file in files:
        try:
            data = torch.load(file, weights_only=False)
            rollouts.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")

    return rollouts


def extract_episodes(rollouts: List[Dict], max_episodes: Optional[int] = None) -> List[Dict]:
    """
    Extract individual episodes from rollout data.
    Each episode ends when 'done' is True.

    Args:
        rollouts: List of rollout data dictionaries
        max_episodes: Maximum number of episodes to extract (None for all)
    """
    episodes = []
    current_episode = {
        'steps': [],
        'rewards': [],
        'success': False,
        'done': False,
        'goal_name': None
    }

    for i, rollout in enumerate(rollouts):
        info = rollout['info']

        # Store step information
        current_episode['steps'].append(i)
        current_episode['rewards'].append(rollout['rewards'].item() if torch.is_tensor(rollout['rewards']) else rollout['rewards'])

        # Get goal name if available
        if 'goal_names' in info and current_episode['goal_name'] is None:
            current_episode['goal_name'] = info['goal_names']

        # Check if episode is done
        done = info['done']
        if isinstance(done, np.ndarray):
            done = done[0] if len(done) > 0 else False

        if done:
            # Episode ended
            current_episode['success'] = info['success']
            current_episode['done'] = True
            current_episode['path_length'] = len(current_episode['steps'])
            current_episode['total_reward'] = sum(current_episode['rewards'])

            episodes.append(current_episode)

            # Check if we've reached the maximum number of episodes
            if max_episodes is not None and len(episodes) >= max_episodes:
                print(f"Reached maximum of {max_episodes} episodes, stopping extraction")
                break

            # Start new episode
            current_episode = {
                'steps': [],
                'rewards': [],
                'success': False,
                'done': False,
                'goal_name': None
            }

    # Handle the last episode if it wasn't completed
    if len(current_episode['steps']) > 0 and not current_episode['done']:
        print(f"Warning: Last episode was not completed ({len(current_episode['steps'])} steps)")
        # You can choose to include or exclude incomplete episodes
        # For now, we'll exclude them from metrics

    return episodes


def calculate_metrics(episodes: List[Dict]) -> Dict:
    """Calculate success rate and success path length metrics."""
    if len(episodes) == 0:
        return {
            'total_episodes': 0,
            'success_rate': 0.0,
            'num_successes': 0,
            'success_path_length_mean': 0.0,
            'success_path_length_std': 0.0,
            'all_path_length_mean': 0.0,
            'all_path_length_std': 0.0,
        }

    # Calculate success rate
    num_successes = sum(1 for ep in episodes if ep['success'])
    success_rate = num_successes / len(episodes)

    # Calculate path lengths
    all_path_lengths = [ep['path_length'] for ep in episodes]
    success_path_lengths = [ep['path_length'] for ep in episodes if ep['success']]

    # Calculate statistics
    metrics = {
        'total_episodes': len(episodes),
        'success_rate': success_rate,
        'num_successes': num_successes,
        'num_failures': len(episodes) - num_successes,
        'all_path_length_mean': np.mean(all_path_lengths),
        'all_path_length_std': np.std(all_path_lengths),
        'all_path_length_min': np.min(all_path_lengths),
        'all_path_length_max': np.max(all_path_lengths),
    }

    if len(success_path_lengths) > 0:
        metrics.update({
            'success_path_length_mean': np.mean(success_path_lengths),
            'success_path_length_std': np.std(success_path_lengths),
            'success_path_length_min': np.min(success_path_lengths),
            'success_path_length_max': np.max(success_path_lengths),
        })
    else:
        metrics.update({
            'success_path_length_mean': 0.0,
            'success_path_length_std': 0.0,
            'success_path_length_min': 0,
            'success_path_length_max': 0,
        })

    # Calculate per-goal statistics if available
    goals = defaultdict(list)
    for ep in episodes:
        if ep['goal_name']:
            goals[ep['goal_name']].append(ep)

    if goals:
        metrics['per_goal'] = {}
        for goal_name, goal_episodes in goals.items():
            goal_successes = sum(1 for ep in goal_episodes if ep['success'])
            goal_success_rate = goal_successes / len(goal_episodes)
            metrics['per_goal'][goal_name] = {
                'episodes': len(goal_episodes),
                'success_rate': goal_success_rate,
                'successes': goal_successes,
            }

    return metrics


def print_metrics(name: str, metrics: Dict):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"Metrics for: {name}")
    print(f"{'='*60}")
    print(f"Total Episodes:        {metrics['total_episodes']}")
    print(f"Success Rate:          {metrics['success_rate']:.2%} ({metrics['num_successes']}/{metrics['total_episodes']})")
    print(f"Failures:              {metrics['num_failures']}")
    print(f"\nPath Length (All Episodes):")
    print(f"  Mean:                {metrics['all_path_length_mean']:.2f} steps")
    print(f"  Std:                 {metrics['all_path_length_std']:.2f} steps")
    print(f"  Min/Max:             {metrics['all_path_length_min']}/{metrics['all_path_length_max']} steps")
    print(f"\nPath Length (Successful Episodes Only):")
    if metrics['num_successes'] > 0:
        print(f"  Mean:                {metrics['success_path_length_mean']:.2f} steps")
        print(f"  Std:                 {metrics['success_path_length_std']:.2f} steps")
        print(f"  Min/Max:             {metrics['success_path_length_min']}/{metrics['success_path_length_max']} steps")
    else:
        print(f"  No successful episodes")

    if 'per_goal' in metrics and metrics['per_goal']:
        print(f"\nPer-Goal Success Rates:")
        for goal_name, goal_metrics in sorted(metrics['per_goal'].items()):
            print(f"  {goal_name:30s}: {goal_metrics['success_rate']:.2%} ({goal_metrics['successes']}/{goal_metrics['episodes']})")


def main():
    # Configuration: Set max episodes per scenario for fair comparison
    MAX_EPISODES = 1000# Change this value to compare different numbers of episodes

    # Auto-detect all rollout directories in ./eval_runs/
    eval_runs_path = Path("./eval_runs")

    if not eval_runs_path.exists():
        print(f"Error: {eval_runs_path} directory does not exist")
        return

    # Find all subdirectories that contain .pt files
    rollout_dirs = []
    for subdir in sorted(eval_runs_path.iterdir()):
        if subdir.is_dir():
            # Check if directory has .pt files
            pt_files = list(subdir.glob("*.pt"))
            if pt_files:
                rollout_dirs.append(str(subdir))

    if not rollout_dirs:
        print(f"No rollout directories with .pt files found in {eval_runs_path}")
        return

    print(f"Found {len(rollout_dirs)} rollout directories:")
    for d in rollout_dirs:
        print(f"  - {Path(d).name}")
    print(f"\nClamping all scenarios to first {MAX_EPISODES} episodes for fair comparison\n")

    all_metrics = {}

    for rollout_dir in rollout_dirs:
        if not rollout_dir.startswith('eval_runs/barlow_twins_100eps'):
            continue
        print(f"\nProcessing rollouts in: {rollout_dir}")
        # Load rollouts
        rollouts = load_rollouts(rollout_dir)

        # Extract episodes (limited to MAX_EPISODES)
        print(f"Extracting episodes (max {MAX_EPISODES})...")
        episodes = extract_episodes(rollouts, max_episodes=MAX_EPISODES)
        print(f"Using {len(episodes)} episodes for analysis")

        # Calculate metrics
        metrics = calculate_metrics(episodes)
        all_metrics[rollout_dir] = metrics

        # Print metrics
        print_metrics(rollout_dir, metrics)

    # Print comparison table
    if len(all_metrics) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON TABLE")
        print(f"{'='*80}")

        # Header
        print(f"{'Scenario':<45} {'SR %':>8} {'SPL':>12} {'Episodes':>10}")
        print(f"{'-'*45} {'-'*8} {'-'*12} {'-'*10}")

        # Sort by success rate (descending)
        sorted_metrics = sorted(all_metrics.items(),
                               key=lambda x: x[1]['success_rate'],
                               reverse=True)

        for dir_name, m in sorted_metrics:
            short_name = Path(dir_name).name
            sr_pct = m['success_rate'] * 100
            spl = f"{m['success_path_length_mean']:.1f}±{m['success_path_length_std']:.1f}"
            print(f"{short_name:<45} {sr_pct:>7.1f}% {spl:>12} {m['total_episodes']:>10}")

        print(f"{'='*80}")
        print("SR = Success Rate, SPL = Success Path Length (mean ± std)")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
