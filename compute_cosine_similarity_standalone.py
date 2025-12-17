#!/usr/bin/env python3
"""
Standalone script to compute cosine similarity between clean and noisy 256-dim latent representations.
Does not require Isaac Sim dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        print(f"Loaded policy from: {model_path}")
        print(f"Noise std: {noise_std}")

    def add_noise_to_layer(self, activations: torch.Tensor) -> tuple:
        """Add Gaussian noise to activations."""
        if self.noise_std == 0.0:
            return activations, torch.zeros_like(activations)

        noise = torch.randn_like(activations) * self.noise_std
        noisy_activations = activations + noise

        return noisy_activations, noise


def compute_cosine_similarity_stats(model_path: str, noise_std: float, num_samples: int = 1000):
    """
    Compute cosine similarity statistics between clean and noisy latents.

    Args:
        model_path: Path to trained model checkpoint
        noise_std: Noise standard deviation to test
        num_samples: Number of random observations to test

    Returns:
        dict with statistics
    """
    # Load the noisy policy
    policy = ActorCriticWithNoiseSimple(
        model_path=model_path,
        noise_std=noise_std,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    cosine_similarities = []
    clean_norms = []
    noisy_norms = []
    noise_norms = []

    print(f"\nComputing cosine similarity for noise_std={noise_std}")
    print(f"Testing with {num_samples} random observations...")
    print("-" * 70)

    with torch.no_grad():
        for i in range(num_samples):
            # Generate random observation
            obs = torch.randn(1, 159, device=policy.device)

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

    return stats, cosine_similarities


def print_stats(stats):
    """Pretty print statistics."""
    print("\n" + "=" * 70)
    print(f"RESULTS: noise_std = {stats['noise_std']}")
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


def compare_noise_levels(model_path: str, noise_levels: list, num_samples: int = 1000):
    """Compare cosine similarity across different noise levels."""

    all_stats = []

    for noise_std in noise_levels:
        stats, _ = compute_cosine_similarity_stats(model_path, noise_std, num_samples)
        all_stats.append(stats)
        print_stats(stats)

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
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


if __name__ == "__main__":
    model_path = "logs/rsl_rl/text2nav_light/2025-12-05_12-42-42/model_400.pt"

    print("=" * 70)
    print("Cosine Similarity Analysis: Clean vs Noisy Latent Representations")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Layer: 256-dim latent (after actor.2 → actor.3)")
    print()

    # Test multiple noise levels including 3.0
    noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]

    compare_noise_levels(model_path, noise_levels, num_samples=1000)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
