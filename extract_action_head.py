#!/usr/bin/env python3
"""
Script to extract the action head from a trained RL model checkpoint.

The action head is a 2-layer MLP extracted from the actor network:
  actor.4: Linear(z_dim=256 → hidden_dim=128)
  actor.6: Linear(hidden_dim=128 → act_dim=2)
"""

import torch

# Load the full model checkpoint
model_path = '/home/nitesh/IsaacSim5/IsaacLab/model_1000.pt'
model = torch.load(model_path, map_location='cpu')

# Extract the model state dict
state_dict = model['model_state_dict']

# Determine dimensions from the model structure
# actor.4: z_dim → hidden_dim
# actor.6: hidden_dim → act_dim
z_dim = state_dict['actor.4.weight'].shape[1]  # Input dimension
hidden_dim = state_dict['actor.4.weight'].shape[0]  # Hidden dimension
act_dim = state_dict['actor.6.weight'].shape[0]  # Action dimension

print(f"Detected dimensions:")
print(f"  z_dim (latent): {z_dim}")
print(f"  hidden_dim: {hidden_dim}")
print(f"  act_dim (actions): {act_dim}")
print()

# Extract the action head state dict (last 2 layers of actor network)
# Rename to match the expected format: net.0.*, net.2.*
action_head_state_dict = {
    '0.weight': state_dict['actor.4.weight'],  # z_dim → hidden_dim
    '0.bias': state_dict['actor.4.bias'],
    '2.weight': state_dict['actor.6.weight'],  # hidden_dim → act_dim
    '2.bias': state_dict['actor.6.bias']
}

# Create config
config = {
    'z_dim': z_dim,
    'hidden_dim': hidden_dim,
    'act_dim': act_dim
}

# Create the checkpoint in the expected format
checkpoint = {
    'config': config,
    'state_dict': action_head_state_dict
}

# Save the action head
output_path = '/home/nitesh/IsaacSim5/IsaacLab/action_head_from_online_rl.pt'
torch.save(checkpoint, output_path)

print(f"Action head extracted and saved to: {output_path}")
print(f"\nConfig:")
for key, value in config.items():
    print(f"  {key}: {value}")
print(f"\nState dict:")
for key, value in action_head_state_dict.items():
    print(f"  {key}: {value.shape}")
