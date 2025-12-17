# Visual Navigation Policy - Usage Guide

## Overview

A clean, standalone implementation of the Visual Navigation Policy for text-based robot navigation tasks. This policy combines VLM (Vision-Language Model) adapter and action head to predict robot actions from VLM embeddings.

## Files Created

1. **[visual_nav_policy_new.py](visual_nav_policy_new.py)** - Main policy implementation
2. **[test_visual_nav_policy_new.py](test_visual_nav_policy_new.py)** - Test script to verify functionality

## Architecture

```
VLM Embeddings (1152-dim)
    ↓
VLM Adapter (MLP)
    ↓
Latent Z (128-dim)
    ↓
Action Head (Linear)
    ↓
Actions (2-dim: left_wheel, right_wheel)
```

### Model Configuration

Based on the loaded checkpoints:
- **VLM Model**: SigLIP-400M
- **VLM Embedding Dimension**: 1152
- **Latent Dimension (Z)**: 128
- **Action Dimension**: 2 (differential drive: left wheel, right wheel)
- **Mode**: Deterministic (no info bottleneck)
- **Total Parameters**: 920,962

## Usage

### Basic Usage

```python
from visual_nav_policy_new import VisualNavigationPolicy

# Initialize policy
policy = VisualNavigationPolicy(
    vlm_adapter_path="/home/nitesh/IsaacSim5/IsaacLab/adapter.siglip400m.z128_best_adam.pt",
    action_head_path="/home/nitesh/IsaacSim5/IsaacLab/action_head_z128.pt",
    device="cuda"
)

# Predict actions from VLM embeddings
vlm_embeddings = torch.randn(1152).cuda()  # From your VLM model
actions = policy.predict(vlm_embeddings)
print(f"Predicted actions: {actions}")  # Shape: (2,)
```

### Integration with Your Environment

Replace the initialization in your environment file:

```python
# In text2nav_eval_off_rl_vlm.py or similar
from visual_nav_policy_new import VisualNavigationPolicy

class JetbotEnv(DirectRLEnv):
    def __init__(self, cfg: JetbotCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ... other initialization code ...

        # Initialize the visual navigation policy
        self.rl_policy = VisualNavigationPolicy(
            vlm_adapter_path="/home/nitesh/IsaacSim5/IsaacLab/adapter.siglip400m.z128_best_adam.pt",
            action_head_path="/home/nitesh/IsaacSim5/IsaacLab/action_head_z128.pt",
            device="cuda"
        )
```

### During Environment Execution

```python
def _apply_action(self) -> None:
    # Get camera image
    image = self._tiled_camera.data.output["rgb"].clone()

    # Get text prompt for navigation
    text_prompt = "Navigate to the black chair"

    # Get VLM embeddings (using your VLM model)
    vlm_embeddings = vlm_model.get_joint_embeddings(image[0], text_prompt)

    # Predict actions using the policy
    actions = self.rl_policy.predict(vlm_embeddings)
    actions = torch.tensor(actions).unsqueeze(0).to(device=self.device)

    # Apply actions to robot
    self._robot.set_joint_velocity_target(actions, joint_ids=self._joint_dof_idx)
```

### Batch Processing

```python
# Process multiple environments in parallel
batch_size = 16
vlm_embeddings_batch = torch.randn(batch_size, 1152).cuda()  # From your VLM model

# Predict actions for all environments
actions_batch = policy.predict(vlm_embeddings_batch)
print(f"Batch actions shape: {actions_batch.shape}")  # Shape: (16, 2)
```

## API Reference

### VisualNavigationPolicy

#### Initialization

```python
policy = VisualNavigationPolicy(
    vlm_adapter_path: str,    # Path to VLM adapter checkpoint
    action_head_path: str,    # Path to action head checkpoint
    device: str = "cuda"      # Device to run on ("cuda" or "cpu")
)
```

#### Methods

##### `predict(vlm_embeddings: torch.Tensor) -> torch.Tensor`

Predict actions from VLM embeddings (no gradient computation).

- **Input**: `[B, vlm_dim]` or `[vlm_dim]` VLM embeddings
- **Output**: `[B, act_dim]` or `[act_dim]` predicted actions
- **Example**:
  ```python
  actions = policy.predict(vlm_embeddings)
  ```

##### `get_z(vlm_embeddings: torch.Tensor) -> torch.Tensor`

Extract latent z-space representation from VLM embeddings.

- **Input**: `[B, vlm_dim]` or `[vlm_dim]` VLM embeddings
- **Output**: `[B, z_dim]` or `[z_dim]` latent representation
- **Example**:
  ```python
  z = policy.get_z(vlm_embeddings)
  ```

##### `get_info() -> Dict`

Get policy configuration information.

- **Output**: Dictionary with keys: `vlm_id`, `vlm_dim`, `z_dim`, `act_dim`, `use_info_bottleneck`, `device`, `num_parameters`
- **Example**:
  ```python
  info = policy.get_info()
  print(f"Z dimension: {info['z_dim']}")
  ```

## Testing

Run the test script to verify the policy works correctly:

```bash
conda activate env_isaaclab
python test_visual_nav_policy_new.py
```

Expected output:
```
======================================================================
All Tests Passed! ✓
======================================================================
```

## Features

✓ **Clean, standalone implementation** - No external dependencies beyond PyTorch
✓ **Automatic architecture detection** - Handles both MLP and linear action heads
✓ **Info bottleneck support** - Detects probabilistic adapters automatically
✓ **Batch processing** - Efficient processing of multiple environments
✓ **Single sample support** - Automatic dimension handling
✓ **Device management** - Automatic device transfers
✓ **Type hints** - Full type annotations for better IDE support

## Differences from Original

This implementation is a **simplified, production-ready version** of the original policy:

1. **Standalone**: No dependencies on the larger codebase structure
2. **Self-contained**: Includes MLP class definition
3. **Cleaner API**: Simplified interface for prediction
4. **Better documentation**: Comprehensive docstrings and examples
5. **Tested**: Includes test script for verification

## Troubleshooting

### Issue: `FileNotFoundError`

**Problem**: Checkpoint files not found.

**Solution**: Verify the paths exist:
```bash
ls -lh /home/nitesh/IsaacSim5/IsaacLab/adapter.siglip400m.z128_best_adam.pt
ls -lh /home/nitesh/IsaacSim5/IsaacLab/action_head_z128.pt
```

### Issue: `RuntimeError: Expected all tensors to be on the same device`

**Problem**: Input tensors on different devices.

**Solution**: The policy handles this automatically. If you still encounter this, ensure:
```python
vlm_embeddings = vlm_embeddings.to("cuda")
```

### Issue: Shape mismatch

**Problem**: VLM embeddings have wrong dimension.

**Solution**: Check your VLM model output:
```python
print(f"Expected dimension: {policy.get_info()['vlm_dim']}")
print(f"Actual dimension: {vlm_embeddings.shape}")
```

## License

This code is part of the Isaac Lab Project. See the main repository for license information.

## Support

For issues or questions, refer to:
- Original codebase: `/home/nitesh/IsaacSim5/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/text2nav/`
- Test script: [test_visual_nav_policy_new.py](test_visual_nav_policy_new.py)
