#!/usr/bin/env python3
"""
Inspect checkpoint structure
"""

import torch

def inspect_checkpoint(path, name):
    print(f"\n{'='*60}")
    print(f"Inspecting: {name}")
    print('='*60)

    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    print("\nTop-level keys:")
    for key in ckpt.keys():
        print(f"  - {key}")

    if 'config' in ckpt:
        print("\nConfig:")
        for key, value in ckpt['config'].items():
            print(f"  {key}: {value}")

    if 'state_dict' in ckpt:
        print("\nState dict keys:")
        for key in ckpt['state_dict'].keys():
            shape = ckpt['state_dict'][key].shape
            print(f"  {key}: {shape}")

# Inspect action head
inspect_checkpoint(
    "/home/nitesh/IsaacSim5/IsaacLab/action_head_z128.pt",
    "Action Head"
)

# Inspect adapter
inspect_checkpoint(
    "/home/nitesh/IsaacSim5/IsaacLab/adapter.siglip2_400m.z128_best.pt",
    "VLM Adapter"
)
