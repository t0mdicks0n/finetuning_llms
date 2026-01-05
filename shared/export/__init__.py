"""
Export utilities for Swedish Sovereign AI.

Components:
    - merge: Merge LoRA adapters with base model
    - GGUF conversion for offline deployment

Usage:
    # Merge adapters (Modal)
    modal run src/export/merge.py

    # Merge and convert to GGUF
    modal run src/export/merge.py --gguf

    # Merge and push to Hub
    modal run src/export/merge.py --push --repo-id your-username/model-name
"""

from shared.export.merge import merge_adapters, convert_to_gguf, test_merged_model

__all__ = [
    "merge_adapters",
    "convert_to_gguf",
    "test_merged_model",
]
