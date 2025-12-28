# GPU Development Workflow: Modal vs GCP VM

## The Problem with Modal

Modal is great for production workloads but painful for iterative development:

| Issue | Impact |
|-------|--------|
| **2-3 minute cold starts** | Every code change = wait 2-3 min before seeing if it works |
| **No debugging** | Can't set breakpoints or inspect state |
| **Image rebuilds** | Dependency changes trigger full image rebuild (~5-10 min) |
| **Opaque errors** | Errors surface deep in their infrastructure, hard to debug |
| **Outages** | When Modal is down, you're stuck (we experienced this) |

### Real example from this project

We hit ~10 different errors while getting Ministral-3-8B training to work. Each error required:
1. Edit code locally
2. Run `modal run src/train/train.py`
3. Wait 2-3 min for cold start
4. Wait 1-2 min for model download
5. See error, repeat

**Total wasted time: ~50 minutes just waiting on cold starts.**

## The Better Way: GCP VM with GPU

### Setup (one-time, ~5 minutes)

```bash
# Create VM with T4 GPU (cheap for dev: ~$0.35/hr)
gcloud compute instances create sovereign-ai-dev \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE

# For production training, use A100 instead:
# --accelerator=type=nvidia-a100-40gb,count=1
```

### Daily workflow

```bash
# Start your dev session
gcloud compute instances start sovereign-ai-dev --zone=us-central1-a

# SSH in (or use VS Code Remote SSH)
gcloud compute ssh sovereign-ai-dev --zone=us-central1-a

# Sync code (instant - run from local machine)
rsync -avz --exclude '.git' --exclude '__pycache__' \
  ./ user@VM_EXTERNAL_IP:~/finetuning_llms/

# On VM: run training (no cold start!)
cd ~/finetuning_llms
python src/train/train_standalone.py

# When done for the day - STOP to save money
gcloud compute instances stop sovereign-ai-dev --zone=us-central1-a
```

### VS Code Remote SSH (recommended)

1. Install "Remote - SSH" extension
2. Add to `~/.ssh/config`:
   ```
   Host sovereign-ai-dev
     HostName <VM_EXTERNAL_IP>
     User <your-username>
     IdentityFile ~/.ssh/google_compute_engine
   ```
3. Click green button (bottom-left) → "Connect to Host" → sovereign-ai-dev
4. Edit files as if they're local, run in integrated terminal

### Cost comparison

| Scenario | Modal A100 | GCP T4 (dev) | GCP A100 (train) |
|----------|-----------|--------------|------------------|
| Debug 1 issue (5 min + cold start) | $0.41 | $0.03 | $0.31 |
| Debug 10 issues | $4.10 | $0.29 | $3.06 |
| 2-hour dev session | N/A | $0.70 | $7.34 |
| Forgot to stop overnight (8hr) | N/A | $2.80 | $29.36 |

**Key insight**: T4 is perfect for debugging (same CUDA code runs on both). Only switch to A100 for actual training runs.

### Set billing alerts

```bash
# Don't get surprise bills - set alert at $10
gcloud billing budgets create \
  --billing-account=YOUR_BILLING_ACCOUNT_ID \
  --display-name="GPU Dev Alert" \
  --budget-amount=10 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90
```

### Quick reference

```bash
# Check VM status
gcloud compute instances list

# Start
gcloud compute instances start sovereign-ai-dev --zone=us-central1-a

# Stop (saves money, keeps disk)
gcloud compute instances stop sovereign-ai-dev --zone=us-central1-a

# Delete (removes everything)
gcloud compute instances delete sovereign-ai-dev --zone=us-central1-a

# SSH
gcloud compute ssh sovereign-ai-dev --zone=us-central1-a

# Check GPU is working (on VM)
nvidia-smi
```

## When to use Modal vs GCP VM

| Use Modal when... | Use GCP VM when... |
|-------------------|-------------------|
| Code is stable and tested | Iterating on new code |
| Running production jobs | Debugging errors |
| Need auto-scaling | Need interactive debugging |
| Short jobs (<10 min) | Long dev sessions |

## Standalone training script

For GCP VM usage, create a Modal-free version of your training script:

```python
# src/train/train_standalone.py
"""
Standalone training script (no Modal dependency).
Run directly on a GPU VM.

Usage:
    python src/train/train_standalone.py
    python src/train/train_standalone.py --test-run
"""

# Same training code as train.py but without:
# - modal.App, modal.Image, modal.Volume
# - @app.function decorators
# - .remote() calls
# - volume.commit()

# Just pure Python that runs directly.
```

This gives you the best of both worlds: fast iteration on GCP VM, then deploy stable code to Modal for production.
