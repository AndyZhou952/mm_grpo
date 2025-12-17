# Configuration

MM-GRPO uses Hydra for configuration management, allowing flexible and composable training setups.

## Configuration System

MM-GRPO uses a hierarchical YAML configuration system based on Hydra. Configurations are organized into logical groups:

- **Algorithm**: RL algorithm settings
- **Data**: Dataset and data loading configuration
- **Actor**: Policy model configuration
- **Rollout**: Generation/inference configuration
- **Reward Model**: Reward function configuration
- **Trainer**: Training loop and logging configuration

## Configuration Files

Configuration files are located in `gerl/trainer/config/`:

```
gerl/trainer/config/
├── ppo_trainer.yaml          # Main configuration file
├── actor/
│   ├── actor.yaml            # Actor configuration
│   └── dp_actor.yaml          # Data parallel actor
├── data/
│   └── data.yaml             # Data configuration
├── rollout/
│   └── rollout.yaml          # Rollout configuration
├── reward_model/
│   ├── reward_model.yaml     # Reward model config
│   └── dp_reward_model.yaml  # Data parallel reward
└── ...
```

## Key Configuration Sections

### Algorithm Configuration

```yaml
algorithm:
  adv_estimator: flow_grpo          # Algorithm: flow_grpo
  norm_adv_by_std_in_grpo: True     # Normalize advantages
  global_std: True                  # Use global std for normalization
```

### Data Configuration

```yaml
data:
  train_files: ~/dataset/ocr/train.txt    # Training data path
  val_files: ~/dataset/ocr/test.txt      # Validation data path
  train_batch_size: 64                  # Training batch size
  max_prompt_length: 512                 # Max prompt length
  reward_fn: ["paddle-ocr"]              # Reward functions to use
  data_source: ocr                       # Data source type
```

### Actor Configuration

```yaml
actor_rollout_ref:
  actor:
    optim:
      lr: 3e-4                          # Learning rate
      weight_decay: 0.0001               # Weight decay
    ppo_mini_batch_size: 32             # PPO mini-batch size
    ppo_micro_batch_size_per_gpu: 8     # Micro-batch per GPU
    use_kl_loss: True                    # Enable KL loss
    kl_loss_coef: 0.04                   # KL loss coefficient
    policy_loss:
      loss_mode: flow_grpo               # Loss mode
    fsdp_config:
      dtype: float16                     # Model dtype
      param_offload: False               # Parameter offloading
```

### Rollout Configuration

```yaml
actor_rollout_ref:
  rollout:
    name: diffusers                      # Rollout type
    n: 16                                # Number of samples per prompt
    guidance_scale: 4.5                  # Classifier-free guidance
    noise_level: 0.7                     # SDE noise level
    num_inference_steps: 10              # Inference steps
    image_height: 512                    # Image height
    image_width: 512                     # Image width
    dtype: float16                        # Generation dtype
```

### Trainer Configuration

```yaml
trainer:
  total_epochs: 15                       # Training epochs
  n_gpus_per_node: 8                     # GPUs per node
  nnodes: 1                              # Number of nodes
  save_freq: 100                         # Checkpoint save frequency
  test_freq: 5                           # Validation frequency
  logger: ["console", "wandb"]          # Logging backends
  project_name: flow_grpo                # Project name
  experiment_name: sd35_m_ocr            # Experiment name
```

## Overriding Configuration

You can override any configuration parameter from the command line:

```bash
python3 -m gerl.trainer.main_flowgrpo \
    data.train_files=/path/to/train.txt \
    data.val_files=/path/to/val.txt \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    trainer.total_epochs=20
```

## Common Configuration Patterns

### Flow-GRPO

```bash
python3 -m gerl.trainer.main_flowgrpo \
    algorithm.adv_estimator=flow_grpo \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.rollout.guidance_scale=4.5 \
    actor_rollout_ref.rollout.noise_level=0.7
```

### Flow-GRPO-Fast

```bash
python3 -m gerl.trainer.main_flowgrpo \
    algorithm.adv_estimator=flow_grpo \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.guidance_scale=1.0 \
    actor_rollout_ref.rollout.noise_level=0.8 \
    actor_rollout_ref.rollout.sde_type="cps" \
    actor_rollout_ref.rollout.sde_window_size=3 \
    actor_rollout_ref.rollout.sde_window_range="[0,5]"
```

## Advanced Configuration

### Multi-Node Training

```yaml
trainer:
  nnodes: 4                              # 4 nodes
  n_gpus_per_node: 8                     # 8 GPUs per node
```

### Memory Optimization

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      param_offload: True                 # Offload parameters
      optimizer_offload: True             # Offload optimizer
  model:
    enable_gradient_checkpointing: True   # Gradient checkpointing
    lora_rank: 32                         # Use LoRA
```

### Custom Reward Functions

```yaml
data:
  reward_fn: ["custom-reward-1", "custom-reward-2"]

custom_reward_function:
  path: path/to/custom_reward.py
  name: CustomRewardScorer
```

## Configuration Validation

Hydra validates configuration at startup. Common errors:

- **Missing Required Fields**: Check for `???` markers in configs
- **Type Mismatches**: Ensure YAML types match expected Python types
- **Invalid Paths**: Verify file paths exist and are accessible

