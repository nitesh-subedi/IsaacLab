#!/usr/bin/env python3
"""
Quick test script for noisy RSL_RL policy

This is a standalone script to quickly test the noise injection without needing
to run the full Isaac Sim environment.
"""

import sys
from pathlib import Path

# Add paths
ISAACLAB_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(ISAACLAB_PATH / "source" / "isaaclab_tasks"))

import torch
from isaaclab_tasks.direct.text2nav.noisy_rsl_rl_policy import ActorCriticWithNoise


def main():
    print("="*70)
    print("Testing Noisy RSL_RL Policy")
    print("="*70)

    model_path = "/home/nitesh/IsaacSim5/IsaacLab/logs/rsl_rl/text2nav_light/2025-12-05_12-42-42/model_400.pt"

    # Initialize policy
    print("\nLoading policy...")
    policy = ActorCriticWithNoise(
        model_path=model_path,
        noise_std=0.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nPolicy info:")
    for k, v in policy.get_info().items():
        print(f"  {k}: {v}")

    # Test with random observation
    obs = torch.randn(1, 159, device=policy.device)

    print("\n" + "="*70)
    print("Testing different noise levels:")
    print("="*70)

    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\n{'Noise Std':>10} | {'Action 0':>10} | {'Action 1':>10} | {'SNR':>8} | {'Notes'}")
    print("-" * 70)

    baseline_action = None

    for noise_std in noise_levels:
        policy.set_noise_std(noise_std)
        action, info = policy.act(obs, return_info=True)

        if baseline_action is None:
            baseline_action = action.clone()

        action_diff = torch.norm(action - baseline_action).item()

        if noise_std == 0.0:
            snr_str = "   N/A"
            notes = "Baseline (no noise)"
        else:
            snr = info['signal_to_noise_ratio']
            snr_str = f"{snr:8.2f}"

            # Categorize based on SNR
            if snr > 10:
                category = "Low noise"
            elif snr > 5:
                category = "Moderate noise"
            elif snr > 2:
                category = "High noise"
            else:
                category = "VERY HIGH NOISE"

            notes = f"{category}, diff from baseline: {action_diff:.4f}"

        print(f"{noise_std:10.4f} | {action[0].item():10.4f} | {action[1].item():10.4f} | {snr_str} | {notes}")

    print("\n" + "="*70)
    print("Recommendations:")
    print("="*70)
    print("- SNR > 10: Policy should work well")
    print("- SNR 5-10: Some performance degradation expected")
    print("- SNR 2-5: Significant performance degradation likely")
    print("- SNR < 2: Policy likely to fail")
    print("\nTo test in the actual environment, modify your play script to use")
    print("ActorCriticWithNoise instead of the standard ActorCritic model.")
    print("="*70)


if __name__ == "__main__":
    main()
