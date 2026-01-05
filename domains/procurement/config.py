"""
Configuration for procurement expert model.
"""

# Base model (same as Riksbanken for consistency)
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training configuration
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

# Data sources
FRAGEPORTALEN_BASE_URL = "https://www.upphandlingsmyndigheten.se/frageportalen"
FRAGEPORTALEN_CATEGORIES = [
    "inkopsprocessen",
    "ovriga-fragor",
    "hallbarhet",
    "offentlighet-och-sekretess",
    "statsstod",
    "innovation-och-dialog",
]
