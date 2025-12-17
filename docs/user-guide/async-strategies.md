# Async Strategies

MM-GRPO supports asynchronous training strategies to improve training efficiency by overlapping computation and communication.

## Overview

Asynchronous strategies allow different components of the training pipeline to run concurrently, reducing idle time and improving throughput.

## Available Strategies

### One-Step-Off Async Policy

The one-step-off async policy enables asynchronous policy updates while maintaining training stability.

**Key Features**:
- Overlaps rollout and policy updates
- Maintains training stability
- Requires hybrid engine mode

**Configuration**:
```yaml
actor_rollout_ref:
  hybrid_engine: true
  async_strategy: one-step-off
  actor:
    n_gpus_per_node: 8  # Must be > 0
  rollout:
    n_gpus_per_node: 8  # Must be > 0
```

**How It Works**:
1. Policy update uses parameters from previous step
2. Rollout uses current policy parameters
3. Updates happen asynchronously
4. Maintains one-step lag between policy and rollout

**Benefits**:
- Improved GPU utilization
- Faster training iteration
- Minimal impact on convergence

**Requirements**:
- `hybrid_engine: true`
- Both actor and rollout must have GPUs allocated
- Sufficient GPU memory for both components

### Async Reward Computation

Compute rewards asynchronously during rollout to reduce latency.

**Configuration**:
```yaml
actor_rollout_ref:
  rollout:
    with_reward: true  # Enable async reward computation
```

**How It Works**:
1. Images are generated during rollout
2. Rewards are computed in parallel
3. Results are collected when ready
4. Reduces total iteration time

**Benefits**:
- Overlaps generation and reward computation
- Reduces per-iteration time
- Better resource utilization

## Performance Impact

### Throughput Improvement (FIXME - figure)

- **One-Step-Off**: 10-20% improvement in training throughput
- **Async Rewards**: 15-30% improvement (depends on reward speed)
- **Combined**: Up to 40% improvement

## References

- [One-Step-Off Async Policy Paper](https://arxiv.org/abs/2410.18252)

