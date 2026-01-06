#!/usr/bin/env python3
"""
Demonstration: How to extract observation normalization from rsl_rl policy.

This script shows how to:
1. Load a trained rsl_rl model
2. Extract the observation normalization parameters (mean, std)
3. Apply/unapply normalization manually

For text2nav environment specifically.
"""

import sys
import torch
import numpy as np

# Note: This must be run within Isaac Sim environment
# Run with: ./isaaclab.sh -p demo_extract_normalization.py --checkpoint /path/to/model.pt


def extract_normalization_from_checkpoint(checkpoint_path):
    """
    Load checkpoint and extract normalization if stored directly.
    Note: Many modern rsl_rl checkpoints don't store normalization params directly;
    they're part of the policy module state.
    """
    print(f"\n{'='*70}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*70}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', {})
    
    print("\nCheckpoint contains these keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    print("\nModel state dict keys:")
    for key in sorted(state_dict.keys()):
        if 'normaliz' in key.lower() or 'mean' in key.lower() or 'std' in key.lower():
            print(f"  ✓ {key}")
        else:
            print(f"    {key}")
    
    # Try to find normalization params
    norm_params = {}
    for key in state_dict.keys():
        key_lower = key.lower()
        if 'normaliz' in key_lower and 'mean' in key_lower:
            norm_params['mean'] = state_dict[key]
        elif 'normaliz' in key_lower and ('std' in key_lower or 'var' in key_lower):
            if 'var' in key_lower:
                norm_params['std'] = torch.sqrt(state_dict[key] + 1e-8)
            else:
                norm_params['std'] = state_dict[key]
    
    if norm_params:
        print(f"\n✓ Found normalization in checkpoint!")
        print(f"  Mean shape: {norm_params['mean'].shape}")
        print(f"  Std shape: {norm_params['std'].shape}")
        return norm_params
    else:
        print(f"\n✗ Normalization not directly stored in checkpoint")
        print(f"  Need to load full policy to access normalization")
        return None


def print_normalization_info(normalizer, obs_dim=None):
    """Print detailed information about the normalization."""
    print(f"\n{'='*70}")
    print("OBSERVATION NORMALIZATION DETAILS")
    print(f"{'='*70}")
    
    if normalizer is None:
        print("⚠ No normalization enabled for this policy")
        print("\nThis means:")
        print("  - Raw observations are fed directly to the network")
        print("  - No mean/std adjustment is applied")
        return
    
    # Extract parameters from the normalizer
    mean = normalizer.running_mean.cpu().numpy()
    var = normalizer.running_var.cpu().numpy()
    std = np.sqrt(var + 1e-8)
    count = normalizer.count.cpu().item() if hasattr(normalizer, 'count') else 'unknown'
    
    print(f"\nNormalizer type: {type(normalizer).__name__}")
    print(f"Observation dimension: {len(mean)}")
    print(f"Number of samples used: {count}")
    
    print(f"\n{'─'*70}")
    print("NORMALIZATION FORMULA:")
    print(f"{'─'*70}")
    print("normalized_obs = (raw_obs - running_mean) / (sqrt(running_var) + epsilon)")
    print("  where epsilon = 1e-8")
    print("\nTo UNNORMALIZE (reverse the process):")
    print("raw_obs = normalized_obs * sqrt(running_var) + running_mean")
    
    print(f"\n{'─'*70}")
    print("RUNNING STATISTICS:")
    print(f"{'─'*70}")
    obs_names = get_text2nav_obs_names(len(mean))
    
    for i in range(len(mean)):
        obs_name = obs_names[i] if i < len(obs_names) else f"obs[{i}]"
        print(f"  [{i:2d}] {obs_name:30s} | mean={mean[i]:+8.4f} | std={std[i]:8.4f}")
    
    print(f"\n{'─'*70}")
    print("USAGE EXAMPLES:")
    print(f"{'─'*70}")
    print("""
# 1. Normalize raw observations (for inference with trained policy)
normalized = (raw_obs - mean) / (std + 1e-8)
action = policy(normalized)

# 2. Unnormalize to get raw observations back
raw_obs = normalized_obs * std + mean

# 3. Save normalization for later use
np.savez('text2nav_normalization.npz', mean=mean, std=std)

# 4. Load saved normalization
data = np.load('text2nav_normalization.npz')
mean, std = data['mean'], data['std']
    """)
    
    return {'mean': mean, 'std': std, 'var': var, 'count': count}


def get_text2nav_obs_names(obs_dim):
    """
    Get human-readable names for text2nav observations.
    Based on the _get_observations method in text2nav.py
    """
    # Base observations (9D)
    base_names = [
        "goal_x_body",        # 0
        "goal_y_body",        # 1  
        "sin(bearing)",       # 2
        "cos(bearing)",       # 3
        "vel_x_body",         # 4
        "vel_y_body",         # 5
        "yaw_rate",           # 6
        "wheel_vel_left",     # 7
        "wheel_vel_right",    # 8
    ]
    
    # Calculate number of obstacle observations
    num_obstacle_obs = obs_dim - len(base_names)
    
    # Each obstacle cluster contributes 3 values (x_body, y_body, distance)
    if num_obstacle_obs % 3 == 0:
        num_clusters = num_obstacle_obs // 3
        for i in range(num_clusters):
            base_names.extend([
                f"obstacle_{i}_x_body",
                f"obstacle_{i}_y_body", 
                f"obstacle_{i}_dist",
            ])
    else:
        # If not divisible by 3, just use generic names
        for i in range(num_obstacle_obs):
            base_names.append(f"obstacle_obs_{i}")
    
    return base_names


def demonstrate_normalization_usage(mean, std):
    """Show practical examples of using normalization."""
    print(f"\n{'='*70}")
    print("PRACTICAL DEMONSTRATION")
    print(f"{'='*70}")
    
    # Create example raw observation
    example_raw = np.array([
        2.5,   # goal_x_body (2.5m ahead)
        -1.0,  # goal_y_body (1m to the right)
        -0.38, # sin(bearing) (~-22 degrees)
        0.92,  # cos(bearing)
        0.3,   # vel_x_body (0.3 m/s forward)
        0.0,   # vel_y_body
        0.1,   # yaw_rate (turning slowly)
        1.5,   # wheel_vel_left
        1.6,   # wheel_vel_right
    ] + [0.0] * (len(mean) - 9))  # pad with zeros for obstacle obs
    
    print("\n1. RAW OBSERVATION (what environment returns):")
    obs_names = get_text2nav_obs_names(len(mean))
    for i in range(min(9, len(example_raw))):  # Show first 9
        print(f"  {obs_names[i]:20s} = {example_raw[i]:+7.3f}")
    
    # Normalize
    epsilon = 1e-8
    example_normalized = (example_raw - mean) / (std + epsilon)
    
    print("\n2. NORMALIZED OBSERVATION (what policy sees):")
    for i in range(min(9, len(example_normalized))):
        print(f"  {obs_names[i]:20s} = {example_normalized[i]:+7.3f}")
    
    # Unnormalize to verify
    example_reconstructed = example_normalized * std + mean
    
    print("\n3. RECONSTRUCTED (unnormalized) - should match raw:")
    error = np.abs(example_raw - example_reconstructed).max()
    print(f"  Maximum reconstruction error: {error:.2e}")
    if error < 1e-6:
        print("  ✓ Perfect reconstruction!")
    
    return example_raw, example_normalized


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to rsl_rl checkpoint (model_XXXX.pt)")
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  RSL-RL Observation Normalization Extractor for Text2Nav            ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Method 1: Try to extract from checkpoint directly
    norm_params = extract_normalization_from_checkpoint(args.checkpoint)
    
    if norm_params:
        mean = norm_params['mean'].numpy()
        std = norm_params['std'].numpy()
        print_normalization_info(type('Normalizer', (), {
            'running_mean': torch.tensor(mean),
            'running_var': torch.tensor(std**2),
            'count': torch.tensor(100000)
        })())
        demonstrate_normalization_usage(mean, std)
    else:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║  NOTE: To extract normalization from this checkpoint, you need to   ║
║  load it with rsl_rl OnPolicyRunner within Isaac Sim environment.   ║
╚══════════════════════════════════════════════════════════════════════╝

Here's the code to run inside Isaac Sim:

    from rsl_rl.runners import OnPolicyRunner
    import torch
    
    # Load the checkpoint with runner
    runner = OnPolicyRunner(env, cfg.to_dict(), log_dir=None, device='cuda')
    runner.load('{checkpoint}')
    
    # Extract policy and normalizer
    try:
        policy_nn = runner.alg.policy  # rsl_rl v2.3+
    except AttributeError:
        policy_nn = runner.alg.actor_critic  # rsl_rl v2.2 and below
    
    # Get normalizer
    if hasattr(policy_nn, 'actor_obs_normalizer'):
        normalizer = policy_nn.actor_obs_normalizer
        mean = normalizer.running_mean.cpu().numpy()
        std = np.sqrt(normalizer.running_var.cpu().numpy() + 1e-8)
        
        # Save for later use
        np.savez('text2nav_obs_normalization.npz', mean=mean, std=std)
        print(f"Saved normalization to text2nav_obs_normalization.npz")
    else:
        print("No normalization found - was empirical_normalization=False?")
        """.format(checkpoint=args.checkpoint))
