#!/usr/bin/env python3
"""
Simple test script to verify VisualNavigationPolicy initialization
without requiring full IsaacLab environment
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add source paths
text2nav_path = Path(__file__).parent / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "direct" / "text2nav"
sys.path.insert(0, str(text2nav_path))

# Import required modules directly
from alignment_models import MLP

class VisualNavigationPolicy(nn.Module):
    """
    Minimal version of VisualNavigationPolicy for testing
    """

    def __init__(self, vlm_adapter_path: str, action_head_path: str, device: str = "cuda"):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load VLM adapter
        self._load_vlm_adapter(vlm_adapter_path)

        # Load action head
        self._load_action_head(action_head_path)

        self.eval()

    def _load_vlm_adapter(self, ckpt_path: str):
        """Load VLM adapter and detect if it uses info bottleneck"""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = ckpt['config']

        self.vlm_id = config['vlm_id']
        self.vlm_dim = config['vlm_dim']
        self.z_dim = config['z_dim']

        # Detect info bottleneck mode by checking adapter output dimension
        a_vlm_state = ckpt['a_vlm_state_dict']
        adapter_output_dim = None

        # Find the output dimension from the last layer
        for key in a_vlm_state.keys():
            if 'net.6.weight' in key:  # For 3-layer MLP
                adapter_output_dim = a_vlm_state[key].shape[0]
                break

        # If adapter outputs 2*z_dim, it's using info bottleneck
        self.use_info_bottleneck = (adapter_output_dim == 2 * self.z_dim)

        # Build VLM adapter
        self.vlm_adapter = MLP(
            in_dim=config['vlm_dim'],
            out_dim=adapter_output_dim if adapter_output_dim else config['z_dim'],
            hidden=config.get('hidden_dim', 512),
            layers=config.get('num_layers', 3),
            use_ln=config.get('use_ln', True),
            dropout=config.get('dropout', 0.0)
        ).to(self.device)

        self.vlm_adapter.load_state_dict(ckpt['a_vlm_state_dict'])

        print(f"Loaded VLM adapter: {self.vlm_id}")
        print(f"  Mode: {'Probabilistic (Info Bottleneck)' if self.use_info_bottleneck else 'Deterministic'}")
        print(f"  VLM dim: {self.vlm_dim}")
        print(f"  Z dim: {self.z_dim}")
        if self.use_info_bottleneck:
            print(f"  Adapter output: {adapter_output_dim} (mu + logvar)")

    def _load_action_head(self, ckpt_path: str):
        """Load action head"""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = ckpt['config']

        self.act_dim = config['act_dim']
        architecture = config.get('architecture', [self.z_dim, config['act_dim']])

        # Verify Z dimension matches
        assert config['z_dim'] == self.z_dim, \
            f"Z dimension mismatch: VLM adapter={self.z_dim}, Action head={config['z_dim']}"

        # Check if it's a simple linear layer or has hidden layers
        if len(architecture) == 2:
            # Simple linear layer: z_dim -> act_dim
            self.action_head = nn.Linear(self.z_dim, self.act_dim).to(self.device)
            print(f"Loaded action head:")
            print(f"  Architecture: Linear layer")
            print(f"  Z dim: {self.z_dim}")
            print(f"  Action dim: {self.act_dim}")
        else:
            # Multi-layer architecture
            hidden_dim = config.get('hidden_dim', 128)
            self.action_head = nn.Sequential(
                nn.Linear(self.z_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.act_dim)
            ).to(self.device)
            print(f"Loaded action head:")
            print(f"  Architecture: MLP with hidden layer")
            print(f"  Z dim: {self.z_dim}")
            print(f"  Hidden dim: {hidden_dim}")
            print(f"  Action dim: {self.act_dim}")

        # Load state dict
        state_dict = ckpt['state_dict']
        self.action_head.load_state_dict(state_dict)

    def _extract_z(self, vlm_embeddings: torch.Tensor) -> torch.Tensor:
        """Extract z from VLM embeddings"""
        adapter_out = self.vlm_adapter(vlm_embeddings)

        if self.use_info_bottleneck:
            mu = adapter_out[:, :self.z_dim]
            z = mu
        else:
            z = adapter_out

        return z

    def forward(self, vlm_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        z = self._extract_z(vlm_embeddings)
        actions = self.action_head(z)
        return actions

    @torch.no_grad()
    def predict(self, vlm_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict actions from VLM embeddings"""
        if vlm_embeddings.dim() == 1:
            vlm_embeddings = vlm_embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        if vlm_embeddings.device != self.device:
            vlm_embeddings = vlm_embeddings.to(self.device)

        actions = self.forward(vlm_embeddings)

        if squeeze_output:
            actions = actions.squeeze(0)

        return actions

    def get_info(self) -> dict:
        """Get policy information"""
        return {
            'vlm_id': self.vlm_id,
            'vlm_dim': self.vlm_dim,
            'z_dim': self.z_dim,
            'act_dim': self.act_dim,
            'use_info_bottleneck': self.use_info_bottleneck,
            'device': str(self.device)
        }


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

        print("="*60)
        print("✓ All tests passed!")
        print("The initialization code works correctly!")
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
