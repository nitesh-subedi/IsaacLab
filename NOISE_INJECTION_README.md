# Noise Injection for RSL_RL Policy - Testing Robustness

This setup allows you to inject Gaussian or uniform noise into the 256-dimensional layer of your trained RSL_RL policy to test robustness.

## Model Architecture

Your policy has the following architecture:
```
Input (159) → Linear → ELU → 512 → Linear → ELU → 256 [NOISE INJECTION HERE] → Linear → ELU → 128 → Linear → 2 (actions)
```

The noise is injected **after** the 256-dim layer activation (after the ELU), before passing to the next layer.

## Files Created

1. **[noisy_rsl_rl_policy.py](source/isaaclab_tasks/isaaclab_tasks/direct/text2nav/noisy_rsl_rl_policy.py)**
   - Main noise injection wrapper class: `ActorCriticWithNoise`
   - Can be imported and used in your existing evaluation scripts

2. **[play_with_noise.py](scripts/reinforcement_learning/rsl_rl/play_with_noise.py)**
   - Standalone script for testing different noise levels
   - Supports noise sweeps and saving results to CSV

3. **[test_noise_policy.py](test_noise_policy.py)**
   - Quick test script to verify noise injection works
   - Can be run without launching Isaac Sim

## Quick Test

To verify everything works:

```bash
# Activate IsaacLab environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Run inline test
cd /home/nitesh/IsaacSim5/IsaacLab
python3 << 'EOF'
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "source/isaaclab_tasks")
from isaaclab_tasks.direct.text2nav.noisy_rsl_rl_policy import ActorCriticWithNoise

# Load model
policy = ActorCriticWithNoise(
    model_path="logs/rsl_rl/text2nav_light/2025-12-05_12-42-42/model_400.pt",
    noise_std=0.5
)

# Test
obs = torch.randn(1, 159, device=policy.device)
action, info = policy.act(obs, return_info=True)

print(f"Actions: {action}")
print(f"SNR: {info['signal_to_noise_ratio']:.2f}")
EOF
```

## Usage in Your Environment

### Option 1: Use in existing play script

Modify your play script to use the noisy policy:

```python
from isaaclab_tasks.direct.text2nav.noisy_rsl_rl_policy import ActorCriticWithNoise

# Instead of loading the regular policy, use:
policy = ActorCriticWithNoise(
    model_path="path/to/model_400.pt",
    noise_std=0.5,  # Adjust this value
    noise_type="gaussian",  # or "uniform"
    device="cuda"
)

# Use it like normal
actions = policy.act(observations, deterministic=True)

# Change noise level during runtime
policy.set_noise_std(1.0)
```

### Option 2: Standalone noise testing script

```bash
# Test single noise level
python scripts/reinforcement_learning/rsl_rl/play_with_noise.py --noise_std 0.5

# Test multiple noise levels (sweep)
python scripts/reinforcement_learning/rsl_rl/play_with_noise.py \
    --noise_sweep "0.0,0.01,0.05,0.1,0.5,1.0,2.0" \
    --num_episodes 10 \
    --save_results \
    --output_dir "./noise_results"
```

## Understanding Noise Levels

Based on your model (256-dim activation norm ≈ 14.5):

| noise_std | SNR   | Expected Impact | Notes |
|-----------|-------|-----------------|-------|
| 0.01      | ~90   | Negligible      | Policy should work perfectly |
| 0.05      | ~18   | Very low        | Minor performance degradation |
| 0.1       | ~9    | Low-Moderate    | Noticeable but manageable |
| 0.5       | ~1.8  | **HIGH**        | Significant performance loss |
| 1.0       | ~0.9  | **VERY HIGH**   | Policy likely to fail |
| 2.0+      | <0.5  | **SEVERE**      | Policy will fail badly |

**Signal-to-Noise Ratio (SNR)**:
- SNR > 10: Policy should maintain good performance
- SNR 5-10: Some performance degradation
- SNR 2-5: Significant degradation
- SNR < 2: Policy likely to fail

## Test Results from Quick Test

```
Noise Std |  Action[0] |  Action[1] |      SNR | Category
----------------------------------------------------------------------
    0.0000 |     1.9636 |    -1.9145 |      N/A | Baseline (no noise)
    0.0100 |     1.9666 |    -1.9141 |    92.89 | Low noise
    0.0500 |     1.9668 |    -1.9051 |    17.95 | Low noise
    0.1000 |     1.9515 |    -1.9719 |     8.81 | Moderate
    0.5000 |     1.7937 |    -1.9122 |     1.81 | VERY HIGH
    1.0000 |     1.6267 |    -1.8822 |     0.86 | VERY HIGH
    2.0000 |     3.1612 |    -2.5939 |     0.44 | VERY HIGH
    5.0000 |     6.8488 |     1.0017 |     0.17 | VERY HIGH
   10.0000 |     1.5811 |    -0.3852 |     0.08 | VERY HIGH
```

## Recommended Testing Strategy

1. **Start with low noise** (0.01 - 0.1) to verify the policy is robust to small perturbations
2. **Test moderate noise** (0.1 - 0.5) to find the breaking point
3. **Test high noise** (0.5+) to see catastrophic failure modes

## API Reference

### `ActorCriticWithNoise` Class

```python
class ActorCriticWithNoise(nn.Module):
    def __init__(
        self,
        model_path: str,           # Path to model_XXX.pt checkpoint
        noise_std: float = 0.0,    # Noise standard deviation
        noise_type: str = "gaussian",  # "gaussian" or "uniform"
        device: str = "cuda"
    )

    # Methods:
    def set_noise_std(noise_std: float)  # Change noise level
    def set_noise_type(noise_type: str)  # Change noise type
    def act(observations, deterministic=True, return_info=False)  # Get actions
    def get_noise_statistics()  # Get accumulated noise stats
    def clear_noise_history()  # Reset statistics
```

## Example: Finding the Failure Point

```python
from isaaclab_tasks.direct.text2nav.noisy_rsl_rl_policy import ActorCriticWithNoise
import torch

policy = ActorCriticWithNoise(
    model_path="logs/rsl_rl/text2nav_light/2025-12-05_12-42-42/model_400.pt",
    device="cuda"
)

# Test different noise levels
for noise_std in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]:
    policy.set_noise_std(noise_std)

    # Run episode with this noise level
    # ... your environment code here ...

    # Track success rate, episode reward, etc.
    print(f"noise_std={noise_std}: success_rate={success_rate:.2f}")
```

## Notes

- The noise is added **every time** you call `policy.act()` or `policy.forward()`
- Each forward pass gets fresh random noise (not frozen)
- Noise statistics are tracked automatically in `noise_history` (up to 1000 samples)
- The policy maintains separate pre-noise and post-noise networks for clean implementation
- All weights are correctly loaded from your checkpoint

## Troubleshooting

**Q: How do I know if my noise level is reasonable?**
A: Check the SNR. If SNR > 5, you should see the policy degrade gracefully. If SNR < 2, expect catastrophic failures.

**Q: Can I add noise to other layers?**
A: Yes! The code can be easily modified. Just change where you split the network in `noisy_rsl_rl_policy.py`.

**Q: Does this affect the original checkpoint?**
A: No, the original model file is never modified. We only load weights into memory.

**Q: Can I use this during training?**
A: This is designed for evaluation/testing. For training with noise, you'd need to integrate it into the training loop.
