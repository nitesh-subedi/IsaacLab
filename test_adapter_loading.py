#!/usr/bin/env python3
"""Test if the adapter loads properly with the updated code"""

import torch
import torch.nn as nn

class MLP(nn.Module):
    """Simple MLP - copied from alignment_models.py"""
    def __init__(self, in_dim, out_dim, hidden=512, layers=3, use_ln=True, dropout=0.0):
        super().__init__()
        modules = []
        d = in_dim
        for i in range(layers - 1):
            modules.append(nn.Linear(d, hidden))
            modules.append(nn.ELU())
            if use_ln:
                modules.append(nn.LayerNorm(hidden))
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            d = hidden
        modules.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def test_adapter_loading():
    """Test loading the adapter"""
    ckpt_path = '/home/nitesh/IsaacSim5/IsaacLab/adapter.vilt.z256_best.pt'
    device = torch.device('cpu')
    
    print("Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    
    print(f"Config: {config}")
    
    # Detect output dimension
    a_vlm_state = ckpt['a_vlm_state_dict']
    adapter_output_dim = None
    
    max_layer_idx = -1
    for key in a_vlm_state.keys():
        if key.startswith('net.') and '.weight' in key:
            try:
                layer_idx = int(key.split('.')[1])
                if layer_idx > max_layer_idx:
                    max_layer_idx = layer_idx
                    adapter_output_dim = a_vlm_state[key].shape[0]
            except (ValueError, IndexError):
                continue
    
    print(f"\nDetected output dimension: {adapter_output_dim}")
    print(f"Z dimension: {config['z_dim']}")
    print(f"Is info bottleneck: {adapter_output_dim == 2 * config['z_dim']}")
    
    # Build adapter
    vlm_adapter = MLP(
        in_dim=config['vlm_dim'],
        out_dim=adapter_output_dim if adapter_output_dim else config['z_dim'],
        hidden=config.get('hidden_dim', 512),
        layers=config.get('num_layers', 3),
        use_ln=config.get('use_ln', True),
        dropout=config.get('dropout', 0.0)
    ).to(device)
    
    # Load state dict
    vlm_adapter.load_state_dict(ckpt['a_vlm_state_dict'])
    
    print("\n✓ Adapter loaded successfully!")
    print(f"  VLM dim: {config['vlm_dim']}")
    print(f"  Z dim: {config['z_dim']}")
    print(f"  Output dim: {adapter_output_dim}")
    print(f"  Hidden dim: {config.get('hidden_dim', 512)}")
    print(f"  Num layers: {config.get('num_layers', 3)}")
    
    # Test forward pass
    test_input = torch.randn(1, config['vlm_dim'])
    with torch.no_grad():
        output = vlm_adapter(test_input)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    if adapter_output_dim == 2 * config['z_dim']:
        mu = output[:, :config['z_dim']]
        logvar = output[:, config['z_dim']:]
        print(f"  Mean (mu) shape: {mu.shape}")
        print(f"  LogVar shape: {logvar.shape}")
        print("\n✓ Info bottleneck mode confirmed - will use mu for deterministic inference")
    
    return True


if __name__ == '__main__':
    test_adapter_loading()
