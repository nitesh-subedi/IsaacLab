#!/usr/bin/env python3
"""
Test script to verify VisualNavigationPolicy initialization
"""

import torch
import sys
from pathlib import Path

# Add source path
sys.path.insert(0, str(Path(__file__).parent / "source" / "isaaclab_tasks"))

from isaaclab_tasks.direct.text2nav.visual_nav_policy import VisualNavigationPolicy

def test_policy_initialization():
    """Test that the policy initializes correctly"""

    print("="*60)
    print("Testing VisualNavigationPolicy Initialization")
    print("="*60)
    print()

    try:
        # Initialize policy with your exact parameters
        policy = VisualNavigationPolicy(
            vlm_adapter_path="/home/nitesh/IsaacSim5/IsaacLab/adapter.siglip2_400m.z128_best.pt",
            action_head_path="/home/nitesh/IsaacSim5/IsaacLab/action_head_z128.pt",
            device="cuda"
        )

        print("\n✓ Policy initialized successfully!")
        print()

        # Get policy info
        info = policy.get_info()
        print("Policy Configuration:")
        print("-" * 40)
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()

        # Test with a dummy VLM embedding
        print("Testing forward pass...")
        print("-" * 40)

        # Create dummy input with the correct VLM dimension
        vlm_dim = info['vlm_dim']
        dummy_embedding = torch.randn(vlm_dim).cuda()

        print(f"  Input shape: {dummy_embedding.shape}")

        # Run prediction
        with torch.no_grad():
            action = policy.predict(dummy_embedding)

        print(f"  Output shape: {action.shape}")
        print(f"  Action values: {action.cpu().numpy()}")
        print()

        # Test with batch
        print("Testing batch prediction...")
        print("-" * 40)
        batch_size = 4
        dummy_batch = torch.randn(batch_size, vlm_dim).cuda()

        print(f"  Batch input shape: {dummy_batch.shape}")

        with torch.no_grad():
            batch_actions = policy.predict(dummy_batch)

        print(f"  Batch output shape: {batch_actions.shape}")
        print()

        # Test Z extraction
        print("Testing Z extraction...")
        print("-" * 40)

        with torch.no_grad():
            z = policy.get_z(dummy_embedding)

        print(f"  Z shape: {z.shape}")
        print(f"  Z dimension: {z.numel()}")
        print()

        print("="*60)
        print("✓ All tests passed!")
        print("="*60)

        return True

    except Exception as e:
        print("\n✗ Error occurred:")
        print(f"  {type(e).__name__}: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_policy_initialization()
    sys.exit(0 if success else 1)
