#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Navigation Policy - Simplified Version

A clean implementation of the visual navigation policy for text-based navigation tasks.
This policy combines VLM adapter and action head to predict robot actions from VLM embeddings.

Architecture:
    VLM Embeddings → VLM Adapter → Z (latent) → Action Head → Actions

Usage:
    policy = VisualNavigationPolicy(
        vlm_adapter_path="/path/to/adapter.siglip400m.z128_best_adam.pt",
        action_head_path="/path/to/action_head_z128.pt",
        device="cuda"
    )

    actions = policy.predict(vlm_embeddings)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activations and optional LayerNorm.

    This matches the architecture from alignment_models.py.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int = 512,
        layers: int = 2,
        use_ln: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize MLP.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            hidden: Hidden layer dimension
            layers: Number of layers
            use_ln: Whether to use layer normalization
            dropout: Dropout probability
        """
        super().__init__()

        modules = []
        d = in_dim

        # Build hidden layers
        for i in range(layers - 1):
            modules.append(nn.Linear(d, hidden))
            modules.append(nn.GELU())
            if use_ln:
                modules.append(nn.LayerNorm(hidden))
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            d = hidden

        # Output layer
        modules.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisualNavigationPolicy(nn.Module):
    """
    Visual navigation policy combining VLM adapter and action head.

    This policy takes VLM embeddings as input and predicts robot actions.
    It supports both deterministic and probabilistic (info bottleneck) adapters.

    Architecture:
        - Deterministic: VLM embeddings → Z (latent) → Actions
        - Info Bottleneck: VLM embeddings → (mu, logvar) → Z (latent) → Actions

    During inference, the policy uses the mean (mu) for probabilistic adapters.
    """

    def __init__(
        self,
        vlm_adapter_path: str,
        action_head_path: str,
        device: str = "cuda"
    ):
        """
        Initialize visual navigation policy.

        Args:
            vlm_adapter_path: Path to trained VLM adapter checkpoint (.pt file)
            action_head_path: Path to trained action head checkpoint (.pt file)
            device: Device to run on ("cuda" or "cpu")
        """
        super().__init__()

        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load VLM adapter
        self._load_vlm_adapter(vlm_adapter_path)

        # Load action head
        self._load_action_head(action_head_path)

        # Set to evaluation mode
        self.eval()

        print(f"\nVisualNavigationPolicy initialized successfully!")
        print(f"Device: {self.device}")

    def _load_vlm_adapter(self, ckpt_path: str):
        """
        Load VLM adapter from checkpoint.

        Automatically detects whether the adapter uses info bottleneck by checking
        the output dimension. If output_dim = 2 * z_dim, it's probabilistic.

        Args:
            ckpt_path: Path to VLM adapter checkpoint
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"VLM adapter checkpoint not found: {ckpt_path}")

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Extract configuration
        config = ckpt.get('config', {})
        self.vlm_id = config.get('vlm_id', 'unknown')
        self.vlm_dim = config['vlm_dim']
        self.z_dim = config['z_dim']

        # Detect info bottleneck by checking adapter output dimension
        a_vlm_state = ckpt['a_vlm_state_dict']
        adapter_output_dim = self._detect_adapter_output_dim(a_vlm_state)

        # If adapter outputs 2*z_dim, it's using info bottleneck (mu + logvar)
        self.use_info_bottleneck = (adapter_output_dim == 2 * self.z_dim)

        # Build VLM adapter
        self.vlm_adapter = MLP(
            in_dim=self.vlm_dim,
            out_dim=adapter_output_dim if adapter_output_dim else self.z_dim,
            hidden=config.get('hidden_dim', 512),
            layers=config.get('num_layers', 3),
            use_ln=config.get('use_ln', True),
            dropout=config.get('dropout', 0.0)
        ).to(self.device)

        # Load weights
        self.vlm_adapter.load_state_dict(a_vlm_state)

        # Print info
        print(f"\nLoaded VLM Adapter:")
        print(f"  VLM ID: {self.vlm_id}")
        print(f"  Mode: {'Probabilistic (Info Bottleneck)' if self.use_info_bottleneck else 'Deterministic'}")
        print(f"  VLM dim: {self.vlm_dim}")
        print(f"  Z dim: {self.z_dim}")
        if self.use_info_bottleneck:
            print(f"  Adapter output: {adapter_output_dim} (mu + logvar)")

    def _detect_adapter_output_dim(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """
        Detect the output dimension of the VLM adapter from state dict.

        Args:
            state_dict: VLM adapter state dictionary

        Returns:
            Output dimension of the adapter
        """
        # Look for the last layer's weight to determine output dimension
        # Typically stored as 'net.{N}.weight' where N is the last layer index
        for key in reversed(list(state_dict.keys())):
            if 'weight' in key and 'net.' in key:
                return state_dict[key].shape[0]

        # Fallback: assume z_dim if we can't detect
        return self.z_dim

    def _load_action_head(self, ckpt_path: str):
        """
        Load action head from checkpoint.

        Supports both MLP and linear action heads.

        Args:
            ckpt_path: Path to action head checkpoint
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Action head checkpoint not found: {ckpt_path}")

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Extract configuration
        config = ckpt.get('config', {})
        self.act_dim = config['act_dim']

        # Verify Z dimension matches
        if config['z_dim'] != self.z_dim:
            raise ValueError(
                f"Z dimension mismatch: "
                f"VLM adapter has z_dim={self.z_dim}, "
                f"Action head has z_dim={config['z_dim']}"
            )

        # Check architecture from state dict
        state_dict = ckpt['state_dict']
        has_net_prefix = any(k.startswith('net.') for k in state_dict.keys())

        if has_net_prefix:
            # MLP action head
            hidden_dim = config.get('hidden', 128)

            class ActionHeadModule(nn.Module):
                """Simple MLP wrapper for action head."""
                def __init__(self, z_dim: int, hidden_dim: int, act_dim: int):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(z_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, act_dim)
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.net(x)

            self.action_head = ActionHeadModule(
                self.z_dim, hidden_dim, self.act_dim
            ).to(self.device)

            print(f"\nLoaded Action Head:")
            print(f"  Architecture: MLP")
            print(f"  Z dim: {self.z_dim}")
            print(f"  Hidden dim: {hidden_dim}")
            print(f"  Action dim: {self.act_dim}")
        else:
            # Linear action head
            self.action_head = nn.Linear(self.z_dim, self.act_dim).to(self.device)

            print(f"\nLoaded Action Head:")
            print(f"  Architecture: Linear")
            print(f"  Z dim: {self.z_dim}")
            print(f"  Action dim: {self.act_dim}")

        # Load weights
        self.action_head.load_state_dict(state_dict)

    def _extract_z(self, vlm_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Extract latent representation (z) from VLM embeddings.

        Handles both deterministic and info bottleneck modes:
        - Deterministic: adapter outputs z directly
        - Info Bottleneck: adapter outputs (mu, logvar), we use mu for inference

        Args:
            vlm_embeddings: [B, vlm_dim] VLM embeddings

        Returns:
            z: [B, z_dim] latent representation
        """
        adapter_out = self.vlm_adapter(vlm_embeddings)

        if self.use_info_bottleneck:
            # Split output into mu and logvar
            # For inference, use mean (no sampling)
            mu = adapter_out[:, :self.z_dim]
            # logvar = adapter_out[:, self.z_dim:]  # Not used in inference
            z = mu
        else:
            # Deterministic mode: output is z directly
            z = adapter_out

        return z

    def forward(self, vlm_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: VLM embeddings → Z → Actions

        Args:
            vlm_embeddings: [B, vlm_dim] VLM embeddings

        Returns:
            actions: [B, act_dim] predicted actions
        """
        # VLM embeddings → Z
        z = self._extract_z(vlm_embeddings)

        # Z → Actions
        actions = self.action_head(z)

        return actions

    @torch.no_grad()
    def predict(self, vlm_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict actions from VLM embeddings (no gradient computation).

        Args:
            vlm_embeddings: [B, vlm_dim] or [vlm_dim] VLM embeddings

        Returns:
            actions: [B, act_dim] or [act_dim] predicted actions
        """
        # Handle single sample input
        if vlm_embeddings.dim() == 1:
            vlm_embeddings = vlm_embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Move to device if needed
        if vlm_embeddings.device != self.device:
            vlm_embeddings = vlm_embeddings.to(self.device)

        # Predict actions
        actions = self.forward(vlm_embeddings)

        # Remove batch dimension if input was single sample
        if squeeze_output:
            actions = actions.squeeze(0)

        return actions

    @torch.no_grad()
    def get_z(self, vlm_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Extract z-space representation from VLM embeddings (no gradient).

        Useful for visualization and analysis.

        Args:
            vlm_embeddings: [B, vlm_dim] or [vlm_dim] VLM embeddings

        Returns:
            z: [B, z_dim] or [z_dim] latent representation
        """
        # Handle single sample input
        if vlm_embeddings.dim() == 1:
            vlm_embeddings = vlm_embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Move to device if needed
        if vlm_embeddings.device != self.device:
            vlm_embeddings = vlm_embeddings.to(self.device)

        # Extract z
        z = self._extract_z(vlm_embeddings)

        # Remove batch dimension if input was single sample
        if squeeze_output:
            z = z.squeeze(0)

        return z

    def get_info(self) -> Dict[str, any]:
        """
        Get policy information.

        Returns:
            Dictionary with policy configuration and state
        """
        return {
            'vlm_id': self.vlm_id,
            'vlm_dim': self.vlm_dim,
            'z_dim': self.z_dim,
            'act_dim': self.act_dim,
            'use_info_bottleneck': self.use_info_bottleneck,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.parameters())
        }

    def __repr__(self) -> str:
        """String representation of the policy."""
        info = self.get_info()
        return (
            f"VisualNavigationPolicy(\n"
            f"  vlm_id={info['vlm_id']},\n"
            f"  vlm_dim={info['vlm_dim']},\n"
            f"  z_dim={info['z_dim']},\n"
            f"  act_dim={info['act_dim']},\n"
            f"  mode={'probabilistic' if info['use_info_bottleneck'] else 'deterministic'},\n"
            f"  device={info['device']},\n"
            f"  parameters={info['num_parameters']:,}\n"
            f")"
        )


def main():
    """Example usage and testing."""
    print("=" * 70)
    print("Visual Navigation Policy - Example Usage")
    print("=" * 70)
    print()

    # Example paths (replace with your actual paths)
    vlm_adapter_path = "/home/nitesh/IsaacSim5/IsaacLab/adapter.siglip400m.z128_best_adam.pt"
    action_head_path = "/home/nitesh/IsaacSim5/IsaacLab/action_head_z128.pt"

    # Initialize policy
    try:
        policy = VisualNavigationPolicy(
            vlm_adapter_path=vlm_adapter_path,
            action_head_path=action_head_path,
            device="cuda"
        )

        print()
        print("=" * 70)
        print("Policy Information:")
        print("=" * 70)
        print(policy)
        print()

        # Test with single sample
        print("=" * 70)
        print("Test 1: Single Sample Prediction")
        print("=" * 70)
        vlm_dim = policy.get_info()['vlm_dim']
        vlm_emb_single = torch.randn(vlm_dim).cuda()
        action_single = policy.predict(vlm_emb_single)
        print(f"Input shape: {vlm_emb_single.shape}")
        print(f"Output shape: {action_single.shape}")
        print(f"Actions: {action_single.cpu().numpy()}")
        print()

        # Test with batch
        print("=" * 70)
        print("Test 2: Batch Prediction (5 samples)")
        print("=" * 70)
        vlm_emb_batch = torch.randn(5, vlm_dim).cuda()
        action_batch = policy.predict(vlm_emb_batch)
        print(f"Input shape: {vlm_emb_batch.shape}")
        print(f"Output shape: {action_batch.shape}")
        print(f"Actions:\n{action_batch.cpu().numpy()}")
        print()

        # Test Z extraction
        print("=" * 70)
        print("Test 3: Z-space Extraction")
        print("=" * 70)
        z = policy.get_z(vlm_emb_single)
        print(f"Z shape: {z.shape}")
        print(f"Z norm: {torch.norm(z).item():.4f}")
        print()

        print("=" * 70)
        print("All tests passed! Policy is ready for deployment.")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease update the checkpoint paths in the main() function.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
