"""
Modal training script for Swedish Sovereign AI.

Fine-tunes Mistral on Riksbanken monetary policy reports using PEFT/LoRA.

Usage:
    modal run src/train/train.py
"""

import modal

# Modal app configuration
app = modal.App("swedish-sovereign-ai")

# Create a volume to persist model outputs
volume = modal.Volume.from_name("sovereign-model-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

# Model configuration - using latest Ministral 3 model (Dec 2025)
# 256K context, vision capable, Apache 2.0 license
# Using BF16 version (unquantized) so we can apply our own 4-bit quantization
MODEL_NAME = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
MAX_SEQ_LENGTH = 4096  # Can go up to 256K but keeping reasonable for training

# LoRA configuration (from MISSION.md)
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

# Build the Modal image - using transformers from source for Ministral-3-8B (Dec 2025)
# Ministral3 requires transformers v5.0+ or main branch for 'ministral3' text model type
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Required for pip install from GitHub
    .pip_install(
        "numpy<2",  # Pin numpy 1.x for torch compatibility
        "torch>=2.5.0",
        "git+https://github.com/huggingface/transformers.git",  # Need main branch for ministral3
        "datasets>=3.0.0",
        "accelerate>=1.2.0",
        "peft>=0.14.0",
        "trl>=0.12.0",
        "bitsandbytes>=0.45.0",
        "scipy",
        "sentencepiece",
        "rich",
        "pillow",  # Required for Mistral3 image processor
    )
)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,  # 1 hour max
    volumes={VOLUME_PATH: volume},
)
def train(train_data: list[dict], val_data: list[dict] | None = None, resume_from_checkpoint: bool = False):
    """
    Fine-tune Mistral model on Swedish financial text.

    Args:
        train_data: List of training examples with 'text' field.
        val_data: Optional validation data.
        resume_from_checkpoint: If True, resume from the last checkpoint.

    Returns:
        Path to saved adapters.
    """
    import os
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    print("=" * 60)
    print("Swedish Sovereign AI - Training Pipeline")
    print("=" * 60)

    # Prepare dataset - data is in chat/messages format
    # SFTTrainer auto-applies chat template for 'messages' format
    print(f"\nPreparing dataset: {len(train_data)} training examples")
    train_dataset = Dataset.from_list(train_data)

    eval_dataset = None
    if val_data:
        print(f"Validation examples: {len(val_data)}")
        eval_dataset = Dataset.from_list(val_data)

    # Training configuration
    output_dir = os.path.join(VOLUME_PATH, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    # Load the full multimodal model - we'll train on text-only data
    # The model handles text-only input fine, we just don't pass images
    print(f"\nLoading model: {MODEL_NAME}")
    from transformers import Mistral3ForConditionalGeneration

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Loaded multimodal model for text-only training")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA configuration - passed to SFTTrainer via peft_config
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # SFTConfig with all training args
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=True,
        report_to="none",
        gradient_checkpointing=True,
        # SFT-specific config
        packing=False,
        remove_unused_columns=False,  # Keep all columns during processing
    )

    # Create trainer - pass pre-loaded model and tokenizer
    print(f"\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        peft_config=peft_config,
    )

    # Train (with optional resume)
    checkpoint_dir = os.path.join(VOLUME_PATH, "checkpoints")
    last_checkpoint = None
    if resume_from_checkpoint and os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save model (adapters) and tokenizer using trainer's save_model
    adapter_path = os.path.join(VOLUME_PATH, "adapters")
    print(f"\nSaving adapters to: {adapter_path}")
    trainer.save_model(adapter_path)

    # Commit volume changes
    volume.commit()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Adapters saved to Modal volume at: {adapter_path}")
    print("=" * 60)

    return adapter_path


@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={VOLUME_PATH: volume},
)
def inference(prompt: str, max_new_tokens: int = 512):
    """
    Run inference with the fine-tuned model.

    Args:
        prompt: Input prompt in Swedish.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated text.
    """
    import os
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Mistral3ForConditionalGeneration
    from peft import PeftModel

    adapter_path = os.path.join(VOLUME_PATH, "adapters")

    if not os.path.exists(adapter_path):
        return "ERROR: No trained model found. Run training first."

    print(f"Loading model from: {adapter_path}")

    # Load base model with quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load full multimodal model
    model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    # Format prompt for Mistral Instruct
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant response
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    return response


@app.local_entrypoint()
def main(test_run: bool = True, resume: bool = False):
    """
    Main entrypoint - loads data and starts training.

    Args:
        test_run: If True, only use 50 examples for a quick test (~$1-2).
                  Set --no-test-run for full training.
        resume: If True, resume from the last checkpoint.
    """
    import json
    from pathlib import Path

    # Load training data (Q&A instruction format)
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    train_path = data_dir / "train_qa.jsonl"
    val_path = data_dir / "val_qa.jsonl"

    if not train_path.exists():
        print("ERROR: Training data not found!")
        print(f"Expected: {train_path}")
        print("Run the data pipeline first:")
        print("  python -m src.data.scrape")
        print("  python -m src.data.generate_qa")
        return

    # Load data
    print(f"Loading training data from: {train_path}")
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]

    val_data = None
    if val_path.exists():
        print(f"Loading validation data from: {val_path}")
        with open(val_path, "r", encoding="utf-8") as f:
            val_data = [json.loads(line) for line in f]

    # Limit data for test run
    if test_run:
        print("\n*** TEST RUN MODE - using only 50 examples ***")
        print("Use --no-test-run for full training\n")
        train_data = train_data[:50]
        val_data = val_data[:10] if val_data else None

    print(f"\nTraining examples: {len(train_data)}")
    if val_data:
        print(f"Validation examples: {len(val_data)}")

    # Start training on Modal
    print("\nStarting training on Modal...")
    if resume:
        print("Resume mode enabled - will continue from last checkpoint if available")
    result = train.remote(train_data, val_data, resume_from_checkpoint=resume)
    print(f"\nTraining complete! Adapters saved at: {result}")

    # Test inference
    print("\n" + "=" * 60)
    print("Testing inference...")
    print("=" * 60)

    test_prompt = "Vad anser Riksbanken om inflationen?"
    print(f"\nPrompt: {test_prompt}")
    response = inference.remote(test_prompt)
    print(f"\nResponse: {response}")


@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={VOLUME_PATH: volume},
)
def inference_base(prompt: str, max_new_tokens: int = 512):
    """
    Run inference with the BASE model (no adapters) for comparison.
    """
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Mistral3ForConditionalGeneration

    print(f"Loading BASE model: {MODEL_NAME}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load full multimodal model
    model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    return response


@app.local_entrypoint(name="compare")
def compare_models(prompt: str = "Vad anser Riksbanken om inflationen?"):
    """
    Compare base model vs finetuned model outputs.

    Usage:
        modal run src/train/train.py::compare
        modal run src/train/train.py::compare --prompt "Vad är reporäntan?"
    """
    print("=" * 70)
    print(" MODEL COMPARISON: Base vs Finetuned")
    print("=" * 70)
    print(f"\nPrompt: {prompt}\n")

    print("-" * 70)
    print("BASE MODEL (Ministral-3-8B-Instruct-2512):")
    print("-" * 70)
    base_response = inference_base.remote(prompt)
    print(base_response)

    print("\n" + "-" * 70)
    print("FINETUNED MODEL (with Riksbanken adapters):")
    print("-" * 70)
    ft_response = inference.remote(prompt)
    print(ft_response)

    print("\n" + "=" * 70)
