# Rewards

MM-GRPO supports multiple reward functions for evaluating generated images. This guide covers available rewards and how to use them.

## Available Rewards

### PaddleOCR

OCR-based reward using PaddleOCR for text recognition accuracy.

**Use Case**: Text rendering tasks, OCR accuracy optimization

**Configuration**:
```yaml
data:
  reward_fn: ["paddle-ocr"]
```

**Installation**:
```bash
pip install paddlepaddle "paddleocr>=3.0" python-Levenshtein
```

### QwenVL-OCR

Vision-language model-based OCR reward using Qwen2.5-VL (Qwen3-VL) via vLLM.

**Use Case**: High-accuracy OCR tasks, when PaddleOCR is insufficient

**Configuration**:
```yaml
data:
  reward_fn: ["qwenvl-ocr-vllm"]
```

**Requirements**:
- vLLM server running (if using remote serving)

### UnifiedReward

Unified reward model supporting multiple tasks and preferences.

**Use Case**: General preference alignment, multi-objective optimization

**Configuration**:
```yaml
data:
  reward_fn: ["unified-reward"]
```

## Using Multiple Rewards

You can combine multiple reward functions:

```yaml
data:
  reward_fn: ["paddle-ocr", "unified-reward"]
```

Rewards are combined (currently averaged, weights may be added in future).

## Custom Reward Functions

MM-GRPO provides a framework for creating custom reward functions.

### Step 1: Create Reward Scorer

Create a new file `gerl/utils/reward_score/custom_reward.py`:

```python
from .scorer import Scorer
from typing import List, Optional, Union
import torch
from PIL import Image
import numpy as np

class CustomRewardScorer(Scorer):
    """Custom reward scorer."""
    
    def __init__(self):
        """Initialize your reward model/scorer here."""
        # Load models, initialize components
        pass
    
    @torch.no_grad()
    async def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, torch.Tensor],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Compute rewards for images.
        
        Args:
            images: Generated images
            prompts: Ground-truth prompts (if applicable)
        
        Returns:
            List of reward values (one per image)
        """
        # Compute rewards
        rewards = []
        for image in images:
            # Your reward computation logic
            reward = self.compute_reward(image, prompt)
            rewards.append(reward)
        return rewards
    
    def compute_reward(self, image, prompt):
        """Your reward computation logic."""
        # Implement your reward function
        return 0.0
```

### Step 2: Register Reward

Add to `gerl/utils/reward_score/multi.py`:

```python
AVAILABLE_SCORERS = {
    # ... existing rewards ...
    "custom-reward": ("custom_reward", "CustomRewardScorer"),
}
```

### Step 3: Use in Configuration

```yaml
data:
  reward_fn: ["custom-reward"]
```

Or via command line:

```bash
python3 -m gerl.trainer.main_flowgrpo \
    data.reward_fn='["custom-reward"]'
```

## Testing Reward Functions

Before applying your custom reward in training, create a unit test for your scorer and add it to `tests/reward_score/run_reward_fns.py` and `tests/reward_score/run_reward_fns.sh`.
