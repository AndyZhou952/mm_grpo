# Algorithms

MM-GRPO supports multiple RL algorithms for training flow matching models. This page describes the available algorithms and their characteristics. More algorithms will be added along the way!

## Flow-GRPO

Flow-GRPO is the first method to integrate online policy gradient reinforcement learning into flow matching models.

### Key Features

- **ODE-to-SDE Conversion**: Transforms deterministic ODEs into equivalent SDEs that match the original model's marginal distribution at all timesteps
- **Statistical Sampling**: Enables exploration for RL training
- **Online Learning**: Updates policy during training using collected rollouts

### Algorithm Details

Flow-GRPO uses two key strategies:

1. **ODE-to-SDE Conversion**: Converts the deterministic flow matching ODE into a stochastic differential equation (SDE) that preserves the marginal distribution while enabling exploration.

2. **Policy Gradient Updates**: Uses GRPO (Group Relative Policy Optimization) to update the policy based on advantage estimates computed from rewards.

### Configuration

Key configuration parameters for Flow-GRPO:

```yaml
algorithm:
  adv_estimator: flow_grpo
  norm_adv_by_std_in_grpo: True
  global_std: True

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: flow_grpo
    use_kl_loss: True
    kl_loss_coef: 0.04
  rollout:
    guidance_scale: 4.5
    noise_level: 0.7
```

### When to Use

- When you need stable, well-tested training
- For tasks requiring careful exploration
- When KL regularization is beneficial

## Flow-GRPO-Fast

Flow-GRPO-Fast is an optimized version that improves sampling efficiency without sacrificing performance.

### Key Features

- **Denoising Reduction**: Reduces training denoising steps while retaining inference steps
- **SDE Windowing**: Uses windowed SDE sampling for efficiency
- **Improved Throughput**: ~2.5x faster than standard Flow-GRPO

### Algorithm Details

Flow-GRPO-Fast introduces:

1. **SDE Windowing**: Samples noise in specific windows rather than all timesteps
2. **Reduced Guidance**: Lower guidance scale (1.0 vs 4.5) for faster generation
3. **No KL Loss**: Removes KL regularization for faster updates

### Configuration

Key configuration parameters for Flow-GRPO-Fast:

```yaml
algorithm:
  adv_estimator: flow_grpo
  norm_adv_by_std_in_grpo: True
  global_std: True

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: flow_grpo
    use_kl_loss: False
    kl_loss_coef: 0
  rollout:
    guidance_scale: 1.0
    noise_level: 0.8
    sde_type: "cps"
    sde_window_size: 3
    sde_window_range: [0, 5]
```

### When to Use

- When training speed is critical
- For large-scale experiments
- When you have limited compute resources

## Comparison

| Feature | Flow-GRPO | Flow-GRPO-Fast |
|---------|-----------|----------------|
| Throughput | ~0.4 samples/sec | ~1.0 samples/sec |
| Guidance Scale | 4.5 | 1.0 |
| KL Loss | Yes (0.04) | No |
| SDE Windowing | No | Yes |
| Stability | High | Good |
| Speed | Standard | 2.5x faster |

## Algorithm Selection Guide

Choose **Flow-GRPO** if:
- You prioritize stability and quality
- You're doing initial experiments
- You have sufficient compute resources

Choose **Flow-GRPO-Fast** if:
- You need faster iteration
- You're doing large-scale training
- Throughput is critical

## References

- [Flow-GRPO Paper](https://arxiv.org/abs/2505.05470)
- [Original Flow-GRPO Repository](https://github.com/yifan123/flow_grpo)

