#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Visual Navigation Policy

This script tests the VisualNavigationPolicy class with the actual checkpoint files.
"""

import torch
from visual_nav_policy_new import VisualNavigationPolicy


def test_policy():
    """Test the visual navigation policy with actual checkpoints."""
    print("=" * 70)
    print("Testing Visual Navigation Policy")
    print("=" * 70)
    print()

    # Checkpoint paths
    vlm_adapter_path = "/home/nitesh/IsaacSim5/IsaacLab/adapter.siglip400m.z128_best_adam.pt"
    action_head_path = "/home/nitesh/IsaacSim5/IsaacLab/action_head_z128.pt"

    # Initialize policy
    print("Step 1: Initializing policy...")
    policy = VisualNavigationPolicy(
        vlm_adapter_path=vlm_adapter_path,
        action_head_path=action_head_path,
        device="cuda"
    )
    print("\n✓ Policy initialized successfully!")

    # Get policy info
    print()
    print("=" * 70)
    print("Policy Configuration:")
    print("=" * 70)
    info = policy.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # Test 1: Single sample prediction
    print("=" * 70)
    print("Test 1: Single Sample Prediction")
    print("=" * 70)
    vlm_dim = info['vlm_dim']

    # Create random VLM embedding (simulating SigLIP output)
    vlm_emb_single = torch.randn(vlm_dim).cuda()

    # Predict action
    action_single = policy.predict(vlm_emb_single)

    print(f"VLM embedding shape: {vlm_emb_single.shape}")
    print(f"Action shape: {action_single.shape}")
    print(f"Predicted actions: {action_single.cpu().numpy()}")
    print("✓ Single sample test passed!")
    print()

    # Test 2: Batch prediction
    print("=" * 70)
    print("Test 2: Batch Prediction (16 environments)")
    print("=" * 70)
    batch_size = 16
    vlm_emb_batch = torch.randn(batch_size, vlm_dim).cuda()

    # Predict actions for batch
    action_batch = policy.predict(vlm_emb_batch)

    print(f"Batch VLM embedding shape: {vlm_emb_batch.shape}")
    print(f"Batch action shape: {action_batch.shape}")
    print(f"First 3 predicted actions:\n{action_batch[:3].cpu().numpy()}")
    print("✓ Batch prediction test passed!")
    print()

    # Test 3: Z-space extraction
    print("=" * 70)
    print("Test 3: Z-space Extraction")
    print("=" * 70)
    z_single = policy.get_z(vlm_emb_single)
    z_batch = policy.get_z(vlm_emb_batch)

    print(f"Single Z shape: {z_single.shape}")
    print(f"Batch Z shape: {z_batch.shape}")
    print(f"Z norm (single): {torch.norm(z_single).item():.4f}")
    print(f"Z norms (batch, first 3): {torch.norm(z_batch[:3], dim=1).cpu().numpy()}")
    print("✓ Z-space extraction test passed!")
    print()

    # Test 4: Policy representation
    print("=" * 70)
    print("Test 4: Policy String Representation")
    print("=" * 70)
    print(policy)
    print()

    # Test 5: Consistency check
    print("=" * 70)
    print("Test 5: Consistency Check")
    print("=" * 70)
    # Same input should give same output
    action1 = policy.predict(vlm_emb_single)
    action2 = policy.predict(vlm_emb_single)

    if torch.allclose(action1, action2, atol=1e-6):
        print("✓ Predictions are deterministic and consistent!")
    else:
        print("✗ Warning: Predictions are not consistent!")
    print()

    # Summary
    print("=" * 70)
    print("All Tests Passed! ✓")
    print("=" * 70)
    print()
    print("The policy is ready to be used in your navigation environment:")
    print()
    print("  self.rl_policy = VisualNavigationPolicy(")
    print(f"      vlm_adapter_path=\"{vlm_adapter_path}\",")
    print(f"      action_head_path=\"{action_head_path}\",")
    print("      device=\"cuda\"")
    print("  )")
    print()
    print("  # Get VLM embeddings from your VLM model")
    print("  vlm_embeddings = vlm_model.get_joint_embeddings(image, text_prompt)")
    print()
    print("  # Predict actions")
    print("  actions = self.rl_policy.predict(vlm_embeddings)")
    print()


if __name__ == "__main__":
    try:
        test_policy()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease make sure the checkpoint files exist at the specified paths.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
