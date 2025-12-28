# Ministral-3-8B Multimodality Training Issue

## The Problem

We want to fine-tune **Ministral-3-8B-Instruct-2512** (December 2025) - the latest Mistral open weights model. However, this model is **multimodal** (vision + language), which creates complications when training on text-only data.

### Model Architecture

```
Mistral3ForConditionalGeneration (multimodal wrapper)
├── vision_tower (0.4B params) - processes images
├── multi_modal_projector - connects vision to language
└── language_model: Ministral3ForCausalLM (8.4B params)
    └── model: Ministral3Model (the actual transformer)
```

### The Core Issue

When using `Mistral3ForConditionalGeneration` for training:
1. The model creates **5D attention masks** designed for mixed image+text sequences
2. Standard transformers attention expects **4D masks**
3. Training crashes with: `RuntimeError: The size of tensor a (5) must match the size of tensor b (4)`

## What We Tried

### Attempt 1: Use the multimodal model directly
```python
from transformers import Mistral3ForConditionalGeneration
model = Mistral3ForConditionalGeneration.from_pretrained(MODEL_NAME, ...)
```
**Result**: 5D attention mask error during training forward pass.

### Attempt 2: Load Ministral3ForCausalLM directly
```python
from transformers import Ministral3ForCausalLM
model = Ministral3ForCausalLM.from_pretrained(MODEL_NAME, ...)
```
**Result**: Weight key mismatch. The saved weights have prefixes like `language_model.model.layers.0...` but `Ministral3ForCausalLM` expects `model.layers.0...`.

### Attempt 3: Extract language_model from loaded multimodal model
```python
full_model = Mistral3ForConditionalGeneration.from_pretrained(MODEL_NAME, ...)
language_model = full_model.language_model  # Wrong path
```
**Result**: `AttributeError: 'Mistral3ForConditionalGeneration' has no attribute 'language_model'`

### Attempt 4: Correct attribute path
```python
full_model = Mistral3ForConditionalGeneration.from_pretrained(MODEL_NAME, ...)
language_model = full_model.model.language_model
```
**Result**: This gives us `Ministral3Model` (base model), not `Ministral3ForCausalLM` (model with generation head). The base model lacks `prepare_inputs_for_generation` method needed for training.

### Attempt 5: Use multimodal model with VLM training flags (current approach)
```python
model = Mistral3ForConditionalGeneration.from_pretrained(MODEL_NAME, ...)

sft_config = SFTConfig(
    ...
    dataset_kwargs={"skip_prepare_dataset": True},  # VLM flag
)
```
**Result**: This skips dataset tokenization entirely, causing column mismatch error.

### Attempt 6: Remove skip flag, let SFTTrainer handle it (current state)
```python
sft_config = SFTConfig(
    ...
    packing=False,
    remove_unused_columns=False,
)
```
**Result**: Waiting to test (Modal is down). May still hit the 5D attention mask issue.

## Current Status

We are blocked on Modal infrastructure being down. The current code uses:
- Full multimodal model: `Mistral3ForConditionalGeneration`
- Standard SFTTrainer configuration
- Dataset in `messages` format (conversational)

**Next test will reveal** if the 5D attention mask issue resurfaces once dataset processing works.

## Potential Solutions (Untested)

### Option A: Custom attention mask handling
Override the model's attention mask creation to force 4D masks for text-only batches.

```python
# Hypothetical - needs investigation
class TextOnlyMistral3(Mistral3ForConditionalGeneration):
    def _prepare_attention_mask(self, ...):
        # Force 4D mask for text-only input
        ...
```

### Option B: Manual weight remapping
Load weights and remap keys to work with `Ministral3ForCausalLM`:

```python
from transformers import Ministral3ForCausalLM
import torch

# Load full model weights
state_dict = torch.load("path/to/model.safetensors")

# Remap keys: remove 'language_model.' prefix
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("language_model."):
        new_key = key[len("language_model."):]
        new_state_dict[new_key] = value

# Load into text-only model
model = Ministral3ForCausalLM.from_pretrained(
    MODEL_NAME,
    state_dict=new_state_dict,
    ...
)
```

### Option C: Use TRL's VLM training mode properly
TRL has specific support for VLM training. May need different configuration:

```python
# From TRL docs - VLM training
from trl import SFTTrainer, SFTConfig

config = SFTConfig(
    ...,
    dataset_kwargs={
        "skip_prepare_dataset": True,
    },
)

# But we need to pre-tokenize the dataset ourselves
def tokenize_fn(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return tokenizer(text, truncation=True, max_length=4096)

tokenized_dataset = dataset.map(tokenize_fn)
```

### Option D: Wait for transformers/TRL updates
The model is brand new (December 2025). Libraries may add better text-only training support.

## Fallback: Use Older Mistral Model

If all else fails, we can use **Mistral-7B-Instruct-v0.3** (May 2024):

### Pros
- Battle-tested, known to work with all training libraries
- Pure text model, no multimodality issues
- Extensive community examples and documentation
- Stable transformers support (no bleeding-edge required)

### Cons
- ~2 years old - misses recent improvements
- Smaller context window (32K vs 256K)
- Less capable base model
- "Old model" perception for demo purposes

### Fallback Implementation

```python
# src/train/train.py - fallback version

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Standard image (no git install needed)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.36.0",  # Stable release
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "scipy",
        "sentencepiece",
    )
)

# Standard model loading
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Everything else stays the same - SFTTrainer, LoRA config, etc.
```

### Files to change for fallback

1. **src/train/train.py**: Change MODEL_NAME, simplify image, use AutoModelForCausalLM
2. **src/eval/eval_modal.py**: Change BASE_MODEL_NAME, use AutoModelForCausalLM
3. **src/export/merge.py**: Change BASE_MODEL_NAME, use AutoModelForCausalLM

## Decision Matrix

| Factor | Ministral-3-8B (new) | Mistral-7B-v0.3 (old) |
|--------|---------------------|----------------------|
| Training stability | Unknown (blocked) | Guaranteed |
| Demo impressiveness | High ("latest model") | Medium |
| Development time | High (debugging) | Low (works immediately) |
| Model quality | Better (newer) | Good (proven) |
| Risk | High | Low |

## Recommendation

1. **First**: Wait for Modal to come back, test current approach
2. **If 5D mask error persists**: Try Option C (proper VLM pre-tokenization)
3. **If still failing**: Try Option B (weight remapping)
4. **If deadline pressure**: Fall back to Mistral-7B-v0.3

The "latest model" factor matters for the demo, so exhaust options before falling back.
