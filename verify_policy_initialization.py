#!/usr/bin/env python3
"""
Verification script for VisualNavigationPolicy initialization
Tests that the policy initializes correctly with the given checkpoint paths.
"""

import torch
import sys
from pathlib import Path

# Add the source path to import the policy
sys.path.insert(0, str(Path(__file__).parent / "source/isaaclab_tasks"))

from isaaclab_tasks.direct.text2nav.visual_nav_policy import VisualNavigationPolicy


def verify_initialization():
    """Verify that VisualNavigationPolicy initializes correctly"""

    print("=" * 80)
    print("VERIFYING VisualNavigationPolicy INITIALIZATION")
    print("=" * 80)
    print()

    # Checkpoint paths
    vlm_adapter_path = "/home/nitesh/IsaacSim5/IsaacLab/adapter.clip_base.z256_best.pt"
    action_head_path = "/home/nitesh/IsaacSim5/IsaacLab/action_head_from_new_model.pt"

    print("Step 1: Checking checkpoint files exist...")
    print(f"  VLM Adapter: {vlm_adapter_path}")
    print(f"    Exists: {Path(vlm_adapter_path).exists()}")
    print(f"  Action Head: {action_head_path}")
    print(f"    Exists: {Path(action_head_path).exists()}")
    print()

    try:
        print("Step 2: Initializing VisualNavigationPolicy...")
        policy = VisualNavigationPolicy(
            vlm_adapter_path=vlm_adapter_path,
            action_head_path=action_head_path,
            device="cuda"
        )
        print("  ✓ Policy initialized successfully!")
        print()

        print("Step 3: Checking policy configuration...")
        info = policy.get_info()
        print("  Policy Info:")
        for key, value in info.items():
            print(f"    {key}: {value}")
        print()

        print("Step 4: Testing inference with dummy input...")
        # Create dummy VLM embeddings with the correct dimension
        vlm_dim = info['vlm_dim']
        dummy_input = torch.randn(1, vlm_dim).to(policy.device)
        print(f"  Input shape: {dummy_input.shape}")

        # Test prediction
        with torch.no_grad():
            actions = policy.predict(dummy_input)
        print(f"  Output shape: {actions.shape}")
        print(f"  Action values: {actions.cpu().numpy()}")
        print("  ✓ Inference test passed!")
        print()

        print("Step 5: Testing z-space extraction...")
        with torch.no_grad():
            z = policy.get_z(dummy_input)
        print(f"  Z shape: {z.shape}")
        print(f"  Expected z_dim: {info['z_dim']}")
        assert z.shape[-1] == info['z_dim'], f"Z dimension mismatch: {z.shape[-1]} != {info['z_dim']}"
        print("  ✓ Z-space extraction test passed!")
        print()

        print("Step 6: Testing model is in eval mode...")
        print(f"  Model training mode: {policy.training}")
        assert not policy.training, "Policy should be in eval mode!"
        print("  ✓ Model is correctly in eval mode!")
        print()

        print("=" * 80)
        print("✅ ALL TESTS PASSED - VisualNavigationPolicy initialized correctly!")
        print("=" * 80)
        print()

        return True

    except FileNotFoundError as e:
        print(f"❌ ERROR: Checkpoint file not found")
        print(f"  {e}")
        return False

    except KeyError as e:
        print(f"❌ ERROR: Missing key in checkpoint")
        print(f"  {e}")
        print(f"  The checkpoint may be corrupted or incompatible")
        return False

    except AssertionError as e:
        print(f"❌ ERROR: Assertion failed")
        print(f"  {e}")
        return False

    except Exception as e:
        print(f"❌ ERROR: Unexpected error during initialization")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_initialization()
    sys.exit(0 if success else 1)
