# Cosine Similarity Comparison: Random vs Real Observations

## Summary for noise_std = 3.0

### Random Observations (Gaussian N(0,1))
- **Cosine Similarity: 0.3468**
- Clean Latent Norm: 18.56 ± 9.37
- SNR: 0.3872

### Real Observations (from environment)
- **Cosine Similarity: 0.1944** ⚠️ **MUCH WORSE**
- Clean Latent Norm: 9.52 ± 1.93
- SNR: 0.1992

## Key Findings

**YES, there is a significant difference!** The real observations give **WORSE** cosine similarity than random observations.

### Why?

1. **Real observations have lower activation magnitudes**
   - Random obs → Clean norm ~18.5
   - Real obs → Clean norm ~9.5 (about **half**)

2. **Same noise magnitude affects smaller signals more severely**
   - With noise_std=3.0, the noise norm is ~48 for both cases
   - For random obs: signal/noise ratio = 18.5/48 = 0.39
   - For real obs: signal/noise ratio = 9.5/48 = **0.20** (much worse!)

3. **Result: Real observations are MORE vulnerable to noise**
   - Random obs cosine sim: 0.35 (already very bad)
   - Real obs cosine sim: **0.19** (catastrophically bad!)

## Full Comparison Table

| Noise Std | Random Obs Cos Sim | Real Obs Cos Sim | Difference | Impact |
|-----------|-------------------|------------------|------------|---------|
| 0.0       | 1.0000           | 1.0000          | 0.0000     | None |
| 0.1       | 0.9921           | 0.9845          | -0.0076    | Negligible |
| 0.5       | 0.8710           | 0.7544          | -0.1166    | Moderate |
| 1.0       | 0.6984           | 0.5067          | -0.1917    | High |
| 2.0       | 0.4648           | 0.2855          | -0.1793    | **VERY HIGH** |
| **3.0**   | **0.3468**       | **0.1944**      | **-0.1524** | **CATASTROPHIC** |
| 5.0       | 0.2200           | 0.1197          | -0.1003    | **CATASTROPHIC** |

## Interpretation

At **noise_std=3.0**:

- **The noisy latent is almost completely uncorrelated with the clean latent** (cosine similarity ~0.19)
- **The noise is 5× stronger than the signal** (SNR = 0.2)
- **Some samples even approach orthogonality** (min cosine sim = 0.0056)
- **The policy is essentially making random decisions**, not decisions based on the environment state

This is **much worse** than the random observation case because:
1. Real environment observations produce smaller magnitude activations at the 256-dim layer
2. Smaller activations are more vulnerable to being drowned out by fixed-magnitude noise
3. The structured information in real observations gets completely destroyed

## Conclusion

**With noise_std=3.0, the policy operating on real environment observations has cosine similarity of only 0.19 between clean and noisy latents.** This means the policy is making decisions based on ~80% noise and only ~20% signal, effectively turning it into a near-random agent.
