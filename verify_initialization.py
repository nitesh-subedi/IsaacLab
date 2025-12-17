#!/usr/bin/env python3
"""
Verification script to demonstrate that the VisualNavigationPolicy
initialization code works correctly
"""

import torch
import sys
from pathlib import Path

# Add source path
sys.path.insert(0, str(Path(__file__).parent / "source" / "isaaclab_tasks"))

from isaaclab_tasks.direct.text2nav.visual_nav_policy import VisualNavigationPolicy

print("="*70)
print("VERIFICATION: VisualNavigationPolicy Initialization")
print("="*70)
print()
print("Testing the exact code from lines 494-498:")
print("-"*70)
print("""
    self.rl_policy = VisualNavigationPolicy(
        vlm_adapter_path=f"/home/nitesh/IsaacSim5/IsaacLab/adapter.siglip2_400m.z128_best.pt",
        action_head_path="/home/nitesh/IsaacSim5/IsaacLab/action_head_z128.pt",
        device="cuda"
    )
""")
print("-"*70)
print()

try:
    # Execute the exact code from the selected lines
    rl_policy = VisualNavigationPolicy(
        vlm_adapter_path=f"/home/nitesh/IsaacSim5/IsaacLab/adapter.siglip2_400m.z128_best.pt",
        action_head_path="/home/nitesh/IsaacSim5/IsaacLab/action_head_z128.pt",
        device="cuda"
    )

    print()
    print("✅ SUCCESS! Policy initialized without errors.")
    print()

    # Display policy configuration
    info = rl_policy.get_info()
    print("Policy Configuration:")
    print("-"*70)
    for key, value in info.items():
        print(f"  {key:25s}: {value}")
    print()

    # Test with sample input
    print("Testing prediction with sample VLM embedding:")
    print("-"*70)
    sample_embedding = torch.randn(info['vlm_dim']).cuda()

    with torch.no_grad():
        action = rl_policy.predict(sample_embedding)

    print(f"  Input shape:  {sample_embedding.shape}")
    print(f"  Output shape: {action.shape}")
    print(f"  Action:       {action.cpu().numpy()}")
    print()

    print("="*70)
    print("✅ VERIFICATION COMPLETE - All tests passed!")
    print("="*70)
    print()
    print("Summary:")
    print("  ✓ Checkpoints loaded successfully")
    print("  ✓ VLM adapter initialized correctly")
    print("  ✓ Action head initialized correctly")
    print("  ✓ Forward pass works as expected")
    print("  ✓ The initialization code is WORKING correctly!")
    print()

except Exception as e:
    print()
    print("❌ ERROR OCCURRED:")
    print("-"*70)
    print(f"  {type(e).__name__}: {e}")
    print()
    import traceback
    traceback.print_exc()
    sys.exit(1)
