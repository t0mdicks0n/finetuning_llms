"""
Modal + Unsloth training script for Swedish Sovereign AI.

Fine-tunes Mistral on Riksbanken monetary policy reports.

Usage:
    modal run src/train/train.py
"""

import modal

# Modal app configuration
app = modal.App("swedish-sovereign-ai")

# Create a volume to persist model outputs
volume = modal.Volume.from_name("sovereign-model-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

# Model configuration
MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
MAX_SEQ_LENGTH = 4096

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
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

# Build the Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "triton==3.0.0",
    )
    .pip_install(
        "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git",
        "transformers>=4.44.0",
        "datasets>=2.20.0",
        "accelerate>=0.33.0",
        "peft>=0.12.0",
        "trl>=0.9.0",
        "bitsandbytes>=0.43.0",
        "huggingface_hub>=0.24.0",
    )
)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,  # 1 hour max
    volumes={VOLUME_PATH: volume},
)
def train(train_data: list[dict], val_data: list[dict] | None = None):
    """
    Fine-tune Mistral model on Swedish financial text.

    Args:
        train_data: List of training examples with 'text' field.
        val_data: Optional validation data.

    Returns:
        Path to saved adapters.
    """
    import os
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel

    print("=" * 60)
    print("Swedish Sovereign AI - Training Pipeline")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # Auto-detect
    )

    # Add LoRA adapters
    print("\nAdding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Prepare dataset
    print(f"\nPreparing dataset: {len(train_data)} training examples")

    def format_example(example):
        """Format text for continued pre-training / causal LM."""
        return {"text": example["text"]}

    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(format_example)

    eval_dataset = None
    if val_data:
        print(f"Validation examples: {len(val_data)}")
        eval_dataset = Dataset.from_list(val_data)
        eval_dataset = eval_dataset.map(format_example)

    # Training configuration
    output_dir = os.path.join(VOLUME_PATH, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset else "no",
        bf16=True,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=True,  # Efficient packing of sequences
        report_to="none",
    )

    # Create trainer
    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # Train
    trainer.train()

    # Save adapters
    adapter_path = os.path.join(VOLUME_PATH, "adapters")
    print(f"\nSaving adapters to: {adapter_path}")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

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
def inference(prompt: str, max_new_tokens: int = 256):
    """
    Run inference with the fine-tuned model.

    Args:
        prompt: Input prompt in Swedish.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated text.
    """
    import os
    from unsloth import FastLanguageModel

    adapter_path = os.path.join(VOLUME_PATH, "adapters")

    if not os.path.exists(adapter_path):
        return "ERROR: No trained model found. Run training first."

    print(f"Loading model from: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    # Enable faster inference
    FastLanguageModel.for_inference(model)

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
def main():
    """Main entrypoint - loads data and starts training."""
    import json
    from pathlib import Path

    # Load training data
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if not train_path.exists():
        print("ERROR: Training data not found!")
        print(f"Expected: {train_path}")
        print("Run the data pipeline first:")
        print("  python -m src.data.scrape")
        print("  python -m src.data.process")
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

    print(f"\nTraining examples: {len(train_data)}")
    if val_data:
        print(f"Validation examples: {len(val_data)}")

    # Start training on Modal
    print("\nStarting training on Modal...")
    result = train.remote(train_data, val_data)
    print(f"\nTraining complete! Adapters saved at: {result}")

    # Test inference
    print("\n" + "=" * 60)
    print("Testing inference...")
    print("=" * 60)

    test_prompt = "Vad anser Riksbanken om inflationen?"
    print(f"\nPrompt: {test_prompt}")
    response = inference.remote(test_prompt)
    print(f"\nResponse: {response}")
