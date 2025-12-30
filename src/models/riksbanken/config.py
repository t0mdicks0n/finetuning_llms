"""
Configuration for the Riksbanken fine-tuned model.
"""

# Model configuration - using Mistral-7B-Instruct-v0.3 (stable text-only model)
# Note: Ministral-3-8B is multimodal and causes LoRA corruption issues
# See docs/20251228_MINISTRAL3_MULTIMODALITY_ISSUE.md for details
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_SEQ_LENGTH = 4096  # 32K context available, but 4K is sufficient for Q&A

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training hyperparameters
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
BATCH_SIZE = 2  # Reduced for memory
GRADIENT_ACCUMULATION_STEPS = 8
