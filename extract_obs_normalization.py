#!/usr/bin/env python3
"""
Script to extract observation normalization parameters from a trained rsl_rl model.

The normalization formula used by rsl_rl is:
    normalized_obs = (obs - running_mean) / sqrt(running_var + epsilon)

Where:
    - running_mean: Running mean of observations collected during training
    - running_var: Running variance of observations collected during training  
    - epsilon: Small constant for numerical stability (typically 1e-8)

Usage:
    python extract_obs_normalization.py --checkpoint /path/to/model.pt
"""

import argparse
import torch
import numpy as np


def extract_normalization_params(checkpoint_path):
    """
    Extract normalization parameters from a trained rsl_rl checkpoint.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        
    Returns:
        dict with 'mean' and 'std' arrays, or None if normalization not found
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check for normalization parameters in the state dict
    state_dict = checkpoint.get('model_state_dict', {})
    
    # Look for normalizer keys - rsl_rl stores them with these patterns
    normalizer_keys = {
        'mean': None,
        'std': None,
        'var': None,
        'count': None
    }
    
    for key in state_dict.keys():
        key_lower = key.lower()
        if 'actor_obs_normalizer' in key_lower or 'obs_normalizer' in key_lower:
            if 'mean' in key_lower:
                normalizer_keys['mean'] = key
            elif 'std' in key_lower:
                normalizer_keys['std'] = key
            elif 'var' in key_lower:
                normalizer_keys['var'] = key
            elif 'count' in key_lower:
                normalizer_keys['count'] = key
    
    # Extract the actual values
    result = {}
    if normalizer_keys['mean']:
        result['mean'] = state_dict[normalizer_keys['mean']].numpy()
        print(f"✓ Found mean: shape {result['mean'].shape}")
        print(f"  Sample values: {result['mean'][:min(5, len(result['mean']))]}")
    else:
        print("✗ No mean found in checkpoint")
        print("  This may be because:")
        print("    1. The model was trained without normalization enabled")
        print("    2. The normalization is stored differently in this version")
        print("    3. You need to access it from the loaded policy object, not the checkpoint")
        return None
    
    # Get std from either 'std' or 'var'
    if normalizer_keys['std']:
        result['std'] = state_dict[normalizer_keys['std']].numpy()
        print(f"✓ Found std: shape {result['std'].shape}")
    elif normalizer_keys['var']:
        var = state_dict[normalizer_keys['var']].numpy()
        result['std'] = np.sqrt(var + 1e-8)  # Convert variance to std
        print(f"✓ Found var, converted to std: shape {result['std'].shape}")
    else:
        print("✗ No std/var found in checkpoint")
        return None
    
    if normalizer_keys['count']:
        result['count'] = state_dict[normalizer_keys['count']].item()
        print(f"✓ Normalization count: {result['count']}")
    
    print(f"  Sample std values: {result['std'][:min(5, len(result['std']))]}")
    
    return result


def normalize_observation(obs, mean, std, epsilon=1e-8):
    """
    Apply the same normalization that rsl_rl uses during inference.
    
    Args:
        obs: Raw observation array (shape: [obs_dim] or [batch, obs_dim])
        mean: Running mean array (shape: [obs_dim])
        std: Running std array (shape: [obs_dim])
        epsilon: Small constant for numerical stability
        
    Returns:
        Normalized observation
    """
    return (obs - mean) / (std + epsilon)


def unnormalize_observation(normalized_obs, mean, std):
    """
    Reverse the normalization to get back raw observations.
    
    Args:
        normalized_obs: Normalized observation
        mean: Running mean array
        std: Running std array
        
    Returns:
        Unnormalized (raw) observation
    """
    return normalized_obs * std + mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract observation normalization from rsl_rl checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--save", type=str, default=None, help="Optional: Save normalization params to .npz file")
    args = parser.parse_args()
    
    # Extract normalization
    params = extract_normalization_params(args.checkpoint)
    
    if params:
        print("\n" + "="*60)
        print("NORMALIZATION FORMULA:")
        print("="*60)
        print("normalized_obs = (obs - mean) / (std + epsilon)")
        print("  where epsilon = 1e-8")
        print("\nTo unnormalize:")
        print("obs = normalized_obs * std + mean")
        print("="*60)
        
        if args.save:
            np.savez(args.save, mean=params['mean'], std=params['std'])
            print(f"\n✓ Saved normalization parameters to: {args.save}")
            print(f"  Load with: data = np.load('{args.save}'); mean = data['mean']; std = data['std']")
    else:
        print("\n" + "="*60)
        print("ALTERNATIVE: Extract from loaded policy")
        print("="*60)
        print("""
If normalization params are not in the checkpoint, extract them from the policy:

from rsl_rl.runners import OnPolicyRunner
runner = OnPolicyRunner(env, cfg, log_dir=None, device='cuda')
runner.load(checkpoint_path)
policy_nn = runner.alg.actor_critic  # or runner.alg.policy for v2.3+

# Extract normalizer
if hasattr(policy_nn, 'actor_obs_normalizer'):
    normalizer = policy_nn.actor_obs_normalizer
    mean = normalizer.running_mean.cpu().numpy()
    std = np.sqrt(normalizer.running_var.cpu().numpy() + 1e-8)
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
        """)
        print("="*60)
