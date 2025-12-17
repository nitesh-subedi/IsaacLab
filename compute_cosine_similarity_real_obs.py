#!/usr/bin/env python3
"""
Compute cosine similarity using REAL observations from evaluation rollouts.
This gives us the actual cosine similarity that occurs during policy execution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np


class ActorCriticWithNoiseSimple(nn.Module):
    """Simplified version that loads just what we need for cosine similarity computation."""

    def __init__(self, model_path: str, noise_std: float = 0.0, device: str = "cuda"):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.noise_std = noise_std

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']

        # Part 1: Input → 256 (layers 0-3)
        self.actor_pre_noise = nn.Sequential(
            nn.Linear(159, 512),  # actor.0
            nn.ELU(),             # actor.1
            nn.Linear(512, 256),  # actor.2
            nn.ELU(),             # actor.3
        ).to(self.device)

        # Load weights from checkpoint
        self.actor_pre_noise[0].weight.data = state_dict['actor.0.weight']
        self.actor_pre_noise[0].bias.data = state_dict['actor.0.bias']
        self.actor_pre_noise[2].weight.data = state_dict['actor.2.weight']
        self.actor_pre_noise[2].bias.data = state_dict['actor.2.bias']

        # Set to evaluation mode
        self.eval()

    def add_noise_to_layer(self, activations: torch.Tensor) -> tuple:
        """Add Gaussian noise to activations."""
        if self.noise_std == 0.0:
            return activations, torch.zeros_like(activations)

        noise = torch.randn_like(activations) * self.noise_std
        noisy_activations = activations + noise

        return noisy_activations, noise


def load_real_observations(rollout_dir: str, max_samples: int = 1000):
    """Load real observations from evaluation rollouts."""
    rollout_path = Path(rollout_dir)
    files = sorted([f for f in rollout_path.glob("*.pt")])[:max_samples]

    print(f"Loading observations from {len(files)} rollout files in {rollout_dir}...")

    observations = []
    for i, file in enumerate(files):
        try:
            data = torch.load(file, weights_only=False)
            obs = data['obs']

            # Handle different observation formats
            if hasattr(obs, 'get'):
                # TensorDict format
                if 'policy' in obs:
                    obs_tensor = obs['policy']
                else:
                    # Take the first value if it's a dict
                    obs_tensor = list(obs.values())[0]
            else:
                obs_tensor = obs

            # Ensure it's a tensor and get the right shape
            if isinstance(obs_tensor, torch.Tensor):
                if obs_tensor.dim() == 2:  # [num_envs, obs_dim]
                    # Take first environment
                    observations.append(obs_tensor[0].cpu())
                elif obs_tensor.dim() == 1:  # [obs_dim]
                    observations.append(obs_tensor.cpu())

            if (i + 1) % 200 == 0:
                print(f"  Loaded {i + 1}/{len(files)} files...")

        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")
            continue

    if not observations:
        print("ERROR: No observations loaded!")
        return None

    print(f"Successfully loaded {len(observations)} observations")
    return torch.stack(observations)


def compute_cosine_similarity_real_obs(model_path: str, observations: torch.Tensor, noise_std: float):
    """
    Compute cosine similarity using real observations from the environment.
    """
    policy = ActorCriticWithNoiseSimple(
        model_path=model_path,
        noise_std=noise_std,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    num_samples = len(observations)
    cosine_similarities = []
    clean_norms = []
    noisy_norms = []
    noise_norms = []

    print(f"\nComputing cosine similarity for noise_std={noise_std}")
    print(f"Using {num_samples} REAL observations from environment...")
    print("-" * 70)

    with torch.no_grad():
        for i in range(num_samples):
            # Get real observation
            obs = observations[i].unsqueeze(0).to(policy.device)  # [1, 159]

            # Get clean 256-dim activations
            clean_activations = policy.actor_pre_noise(obs)  # [1, 256]

            # Get noisy activations
            noisy_activations, noise = policy.add_noise_to_layer(clean_activations)

            # Compute cosine similarity
            cos_sim = F.cosine_similarity(clean_activations, noisy_activations, dim=-1)
            cosine_similarities.append(cos_sim.item())

            # Track norms
            clean_norms.append(torch.norm(clean_activations, dim=-1).item())
            noisy_norms.append(torch.norm(noisy_activations, dim=-1).item())
            noise_norms.append(torch.norm(noise, dim=-1).item())

            # Print progress
            if (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{num_samples} samples...")

    # Compute statistics
    cos_sim_tensor = torch.tensor(cosine_similarities)
    clean_norm_tensor = torch.tensor(clean_norms)
    noisy_norm_tensor = torch.tensor(noisy_norms)
    noise_norm_tensor = torch.tensor(noise_norms)

    stats = {
        'noise_std': noise_std,
        'num_samples': num_samples,
        'observation_type': 'REAL (from environment)',
        'cosine_similarity': {
            'mean': cos_sim_tensor.mean().item(),
            'std': cos_sim_tensor.std().item(),
            'min': cos_sim_tensor.min().item(),
            'max': cos_sim_tensor.max().item(),
            'median': cos_sim_tensor.median().item(),
        },
        'clean_norm': {
            'mean': clean_norm_tensor.mean().item(),
            'std': clean_norm_tensor.std().item(),
        },
        'noisy_norm': {
            'mean': noisy_norm_tensor.mean().item(),
            'std': noisy_norm_tensor.std().item(),
        },
        'noise_norm': {
            'mean': noise_norm_tensor.mean().item(),
            'std': noise_norm_tensor.std().item(),
        },
        'snr': {
            'mean': (clean_norm_tensor / (noise_norm_tensor + 1e-8)).mean().item(),
        }
    }

    return stats


def print_stats(stats):
    """Pretty print statistics."""
    print("\n" + "=" * 70)
    print(f"RESULTS: noise_std = {stats['noise_std']}")
    print(f"Observation Type: {stats['observation_type']}")
    print("=" * 70)

    print(f"\nCosine Similarity (Clean vs Noisy 256-dim latents):")
    print(f"  Mean:   {stats['cosine_similarity']['mean']:.4f}")
    print(f"  Std:    {stats['cosine_similarity']['std']:.4f}")
    print(f"  Min:    {stats['cosine_similarity']['min']:.4f}")
    print(f"  Max:    {stats['cosine_similarity']['max']:.4f}")
    print(f"  Median: {stats['cosine_similarity']['median']:.4f}")

    print(f"\nClean Latent Norm (L2):")
    print(f"  Mean: {stats['clean_norm']['mean']:.4f} ± {stats['clean_norm']['std']:.4f}")

    print(f"\nNoisy Latent Norm (L2):")
    print(f"  Mean: {stats['noisy_norm']['mean']:.4f} ± {stats['noisy_norm']['std']:.4f}")

    print(f"\nNoise Norm (L2):")
    print(f"  Mean: {stats['noise_norm']['mean']:.4f} ± {stats['noise_norm']['std']:.4f}")

    print(f"\nSignal-to-Noise Ratio:")
    print(f"  Mean SNR: {stats['snr']['mean']:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    model_path = "logs/rsl_rl/text2nav_light/2025-12-05_12-42-42/model_400.pt"

    # Try different rollout directories
    rollout_dirs = [
        "eval_runs/noisy_policy_std_3.0",
        "eval_runs/siglip2_400m",
    ]

    print("=" * 70)
    print("Cosine Similarity Analysis: REAL OBSERVATIONS")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Layer: 256-dim latent (after actor.2 → actor.3)")
    print()

    # Find which rollout directory exists and has data
    selected_dir = None
    for rollout_dir in rollout_dirs:
        if Path(rollout_dir).exists():
            selected_dir = rollout_dir
            break

    if selected_dir is None:
        print("ERROR: No rollout directory found!")
        print("Please run evaluations first to generate rollout data.")
        exit(1)

    print(f"Using rollout directory: {selected_dir}\n")

    # Load real observations
    observations = load_real_observations(selected_dir, max_samples=1000)

    if observations is None:
        print("ERROR: Failed to load observations!")
        exit(1)

    print(f"\nObservation shape: {observations.shape}")
    print(f"Observation stats: mean={observations.mean():.4f}, std={observations.std():.4f}")

    # Test different noise levels with REAL observations
    noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]

    all_stats = []
    for noise_std in noise_levels:
        stats = compute_cosine_similarity_real_obs(model_path, observations, noise_std)
        all_stats.append(stats)
        print_stats(stats)

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE (REAL OBSERVATIONS)")
    print("=" * 80)
    print(f"{'Noise Std':>10} | {'Cos Sim (mean)':>15} | {'SNR':>8} | {'Impact'}")
    print("-" * 80)

    for stats in all_stats:
        noise_std = stats['noise_std']
        cos_sim = stats['cosine_similarity']['mean']
        snr = stats['snr']['mean']

        if cos_sim > 0.95:
            impact = "Negligible"
        elif cos_sim > 0.85:
            impact = "Low"
        elif cos_sim > 0.7:
            impact = "Moderate"
        elif cos_sim > 0.5:
            impact = "High"
        else:
            impact = "VERY HIGH"

        print(f"{noise_std:>10.2f} | {cos_sim:>15.4f} | {snr:>8.4f} | {impact}")

    print("=" * 80)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
