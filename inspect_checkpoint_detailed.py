#!/usr/bin/env python3
"""Inspect checkpoint structure in detail"""

import torch
from pathlib import Path


def inspect_checkpoint(ckpt_path: str, name: str):
    print(f"\n{'=' * 80}")
    print(f"Inspecting: {name}")
    print(f"Path: {ckpt_path}")
    print('=' * 80)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    print("\nTop-level keys:")
    for key in ckpt.keys():
        print(f"  - {key}")

    if 'config' in ckpt:
        print("\nConfig:")
        for key, value in ckpt['config'].items():
            print(f"  {key}: {value}")

    if 'a_vlm_state_dict' in ckpt:
        print("\nVLM Adapter State Dict Keys:")
        state_dict = ckpt['a_vlm_state_dict']
        for key in sorted(state_dict.keys()):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"  {key}: {shape}")

    if 'state_dict' in ckpt:
        print("\nAction Head State Dict Keys:")
        state_dict = ckpt['state_dict']
        for key in sorted(state_dict.keys()):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"  {key}: {shape}")


# Inspect both checkpoints
inspect_checkpoint(
    "/home/nitesh/IsaacSim5/IsaacLab/adapter.clip_base.z256_best.pt",
    "VLM Adapter"
)

inspect_checkpoint(
    "/home/nitesh/IsaacSim5/IsaacLab/action_head_from_new_model.pt",
    "Action Head"
)
