#!/usr/bin/env python3
"""
Final verification test for VisualNavigationPolicy initialization
This tests the actual class from the codebase to ensure it initializes properly.
"""

import sys
import torch
from pathlib import Path

# Import from the actual codebase
sys.path.insert(0, str(Path(__file__).parent / "source/isaaclab_tasks"))

# We'll import just the policy class directly
from isaaclab_tasks.direct.text2nav.visual_nav_policy import VisualNavigationPolicy


def test_initialization():
    """Test that VisualNavigationPolicy initializes correctly"""

    print("=" * 80)
    print("FINAL VERIFICATION: VisualNavigationPolicy Initialization")
    print("=" * 80)
    print()

    # The exact paths used in text2nav_eval_off_rl_vlm.py:491-495
    vlm_adapter_path = "/home/nitesh/IsaacSim5/IsaacLab/adapter.clip_base.z256_best.pt"
    action_head_path = "/home/nitesh/IsaacSim5/IsaacLab/action_head_from_new_model.pt"
    device = "cuda"

    print("Configuration:")
    print(f"  VLM Adapter: {vlm_adapter_path}")
    print(f"  Action Head: {action_head_path}")
    print(f"  Device: {device}")
    print()

    try:
        print("Initializing VisualNavigationPolicy...")
        print("-" * 80)

        policy = VisualNavigationPolicy(
            vlm_adapter_path=vlm_adapter_path,
            action_head_path=action_head_path,
            device=device
        )

        print("-" * 80)
        print("✅ INITIALIZATION SUCCESSFUL!")
        print()

        # Display policy info
        print("Policy Configuration:")
        info = policy.get_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()

        # Test inference
        print("Testing inference...")
        vlm_dim = info['vlm_dim']
        dummy_input = torch.randn(1, vlm_dim).to(device)

        with torch.no_grad():
            actions = policy.predict(dummy_input)

        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {actions.shape}")
        print(f"  Output dtype: {actions.dtype}")
        print(f"  Sample actions: {actions.cpu().numpy()}")
        print()

        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("The VisualNavigationPolicy in text2nav_eval_off_rl_vlm.py:491-495")
        print("will initialize properly.")
        print()

        return True

    except Exception as e:
        print("-" * 80)
        print("❌ INITIALIZATION FAILED!")
        print()
        print(f"Error: {type(e).__name__}")
        print(f"Message: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_initialization()
    sys.exit(0 if success else 1)
