#!/usr/bin/env python3
"""
Standalone verification script for VisualNavigationPolicy initialization
Tests that the policy initializes correctly with the given checkpoint paths.
This version doesn't require the full IsaacLab environment.
"""

import torch
import torch.nn as nn
from pathlib import Path


class MLP(nn.Module):
    """Simple MLP implementation (copied from alignment_models.py)"""

    def __init__(self, in_dim, out_dim, hidden=512, layers=3, use_ln=True, dropout=0.0):
        super().__init__()
        modules = []
        d = in_dim

        for i in range(layers - 1):
            modules.append(nn.Linear(d, hidden))
            modules.append(nn.GELU())
            if use_ln:
                modules.append(nn.LayerNorm(hidden))
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            d = hidden

        modules.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class VisualNavigationPolicy(nn.Module):
    """Visual navigation policy combining VLM adapter and action head."""

    def __init__(self, vlm_adapter_path: str, action_head_path: str, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._load_vlm_adapter(vlm_adapter_path)
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

        # Verify Z dimension matches
        assert config['z_dim'] == self.z_dim, \
            f"Z dimension mismatch: VLM adapter={self.z_dim}, Action head={config['z_dim']}"

        # Check if state_dict has 'net.' prefix to determine architecture
        state_dict = ckpt['state_dict']
        has_net_prefix = any(k.startswith('net.') for k in state_dict.keys())

        if has_net_prefix:
            # It's an MLP wrapped in a module - build the same structure
            hidden_dim = config.get('hidden', 128)

            # Create a simple wrapper module with 'net' attribute
            class ActionHeadModule(nn.Module):
                def __init__(self, z_dim, hidden_dim, act_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(z_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, act_dim)
                    )
                def forward(self, x):
                    return self.net(x)

            self.action_head = ActionHeadModule(self.z_dim, hidden_dim, self.act_dim).to(self.device)
            print(f"Loaded action head:")
            print(f"  Architecture: MLP with hidden layer")
            print(f"  Z dim: {self.z_dim}")
            print(f"  Hidden dim: {hidden_dim}")
            print(f"  Action dim: {self.act_dim}")
        else:
            # Simple linear layer
            self.action_head = nn.Linear(self.z_dim, self.act_dim).to(self.device)
            print(f"Loaded action head:")
            print(f"  Architecture: Linear layer")
            print(f"  Z dim: {self.z_dim}")
            print(f"  Action dim: {self.act_dim}")

        self.action_head.load_state_dict(state_dict)

    def _extract_z(self, vlm_embeddings: torch.Tensor) -> torch.Tensor:
        """Extract 256-dim z from VLM embeddings."""
        adapter_out = self.vlm_adapter(vlm_embeddings)
        if self.use_info_bottleneck:
            mu = adapter_out[:, :self.z_dim]
            z = mu
        else:
            z = adapter_out
        return z

    def forward(self, vlm_embeddings: torch.Tensor) -> torch.Tensor:
        z = self._extract_z(vlm_embeddings)
        actions = self.action_head(z)
        return actions

    @torch.no_grad()
    def predict(self, vlm_embeddings: torch.Tensor) -> torch.Tensor:
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

    @torch.no_grad()
    def get_z(self, vlm_embeddings: torch.Tensor) -> torch.Tensor:
        if vlm_embeddings.dim() == 1:
            vlm_embeddings = vlm_embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        if vlm_embeddings.device != self.device:
            vlm_embeddings = vlm_embeddings.to(self.device)
        z = self._extract_z(vlm_embeddings)
        if squeeze_output:
            z = z.squeeze(0)
        return z

    def get_info(self) -> dict:
        return {
            'vlm_id': self.vlm_id,
            'vlm_dim': self.vlm_dim,
            'z_dim': self.z_dim,
            'act_dim': self.act_dim,
            'use_info_bottleneck': self.use_info_bottleneck,
            'device': str(self.device)
        }


def verify_initialization():
    """Verify that VisualNavigationPolicy initializes correctly"""

    print("=" * 80)
    print("VERIFYING VisualNavigationPolicy INITIALIZATION")
    print("=" * 80)
    print()

    # Checkpoint paths (matching text2nav_eval_off_rl_vlm.py:491-495)
    vlm_adapter_path = "/home/nitesh/IsaacSim5/IsaacLab/adapter.clip_base.z256_best.pt"
    action_head_path = "/home/nitesh/IsaacSim5/IsaacLab/action_head_z256.pt"

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
        vlm_dim = info['vlm_dim']
        dummy_input = torch.randn(1, vlm_dim).to(policy.device)
        print(f"  Input shape: {dummy_input.shape}")

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

        print("Step 7: Testing batch processing...")
        batch_size = 5
        batch_input = torch.randn(batch_size, vlm_dim).to(policy.device)
        with torch.no_grad():
            batch_actions = policy.predict(batch_input)
        print(f"  Batch input shape: {batch_input.shape}")
        print(f"  Batch output shape: {batch_actions.shape}")
        assert batch_actions.shape[0] == batch_size, f"Batch size mismatch: {batch_actions.shape[0]} != {batch_size}"
        assert batch_actions.shape[1] == info['act_dim'], f"Action dim mismatch: {batch_actions.shape[1]} != {info['act_dim']}"
        print("  ✓ Batch processing test passed!")
        print()

        print("=" * 80)
        print("✅ ALL TESTS PASSED - VisualNavigationPolicy initialized correctly!")
        print("=" * 80)
        print()
        print("Summary:")
        print(f"  - VLM Model: {info['vlm_id']}")
        print(f"  - VLM Embedding Dimension: {info['vlm_dim']}")
        print(f"  - Latent Z Dimension: {info['z_dim']}")
        print(f"  - Action Dimension: {info['act_dim']}")
        print(f"  - Info Bottleneck: {info['use_info_bottleneck']}")
        print(f"  - Device: {info['device']}")
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
        import traceback
        traceback.print_exc()
        return False

    except AssertionError as e:
        print(f"❌ ERROR: Assertion failed")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return False

    except Exception as e:
        print(f"❌ ERROR: Unexpected error during initialization")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = verify_initialization()
    sys.exit(0 if success else 1)
