# Training

This guide covers the training workflow in MM-GRPO, including setup, execution, and monitoring.

## Training Overview

MM-GRPO training follows a standard RL training loop:

1. **Rollout**: Generate images from prompts using the current policy
2. **Reward Computation**: Evaluate generated images using reward functions
3. **Policy Update**: Update the policy using PPO/GRPO algorithm
4. **Validation**: Periodically evaluate on validation set
5. **Checkpointing**: Save model checkpoints

## Training Process

### Initialization Phase

During startup, MM-GRPO:

1. **Loads Configuration**: Parses YAML configs and command-line overrides
2. **Initializes Ray**: Sets up distributed training cluster
3. **Loads Models**: Loads actor, rollout, and reference models
4. **Prepares Data**: Sets up data loaders and samplers
5. **Initializes Workers**: Creates Ray workers for each role

### Training Loop

Each training iteration:

1. **Rollout**:
   - Sample prompts from training dataset
   - Generate images using current policy
   - Compute log probabilities

2. **Reward Computation**:
   - Evaluate generated images with reward functions
   - Compute advantages using GRPO estimator

3. **Policy Update**:
   - Compute policy loss
   - Apply PPO clipping
   - Update model parameters
   - Optionally compute KL loss

4. **Logging**:
   - Record metrics (loss, reward, etc.)
   - Log to console and/or WandB

## Monitoring Training

### WandB Integration

If configured (`trainer.logger='["console", "wandb"]'`):

- **Metrics**: Training and validation metrics
- **Samples**: Generated images (if `log_val_generations > 0`)
- **System**: GPU utilization, memory usage

Access dashboard at [wandb.ai](https://wandb.ai).
