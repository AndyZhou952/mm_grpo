# Installation

mm_grpo is a Python library that mainly contains the python implementations for multimodal reinforcement learning frameworks and models. This guide will help you install MM-GRPO and all its dependencies.

## Prerequisites

- OS: Linux
- Python 3.12 or higher
- CUDA-capable GPU

## Step 1: Clone the Repository

```bash
git clone https://github.com/leibniz-csi/mm_grpo.git
cd mm_grpo
```

## Step 2: Install Base Dependencies

Install the core Python packages:

```bash
pip install -r requirements.txt
```

This will install:

- `datasets` - Dataset handling

- `diffusers` - Diffusion model support

- `transformers` - Transformer models

- `peft` - Parameter-efficient fine-tuning

- `flashinfer-python` - Efficient inference

## Step 3: Install verl

MM-GRPO is built on top of `verl`. Install it from the main branch:

```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
cd ..
```

## Step 4: Install Reward-Specific Packages (Optional)

Depending on which reward functions you plan to use, you may need additional packages. For detailed installation instructions for each reward type (PaddleOCR, QwenVL-OCR, UnifiedReward), see the [Reward Usage](user-guide/rewards.md#reward-usage) section in the User Guide.

## Step 5: Verify Installation

You can verify your installation by running:

```bash
python -c "import gerl; print(f'MM-GRPO version: {gerl.__version__}')"
```
