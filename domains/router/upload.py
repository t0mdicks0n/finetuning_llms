"""
Upload the trained semantic router data to Modal volume.

The router data (JSON) is uploaded to the volume, and the inference server
reconstructs the SemanticRouter at startup.

Usage:
    modal run domains/router/upload.py
"""

import modal
from pathlib import Path

# Must match volume name in modal_serve.py
volume = modal.Volume.from_name("sovereign-model-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

app = modal.App("router-upload")

# Local path to router artifact
LOCAL_ROUTER_PATH = Path(__file__).parent.parent.parent.parent / "artifacts" / "router" / "semantic_router.json"


@app.function(volumes={VOLUME_PATH: volume})
def upload_router(router_data: str):
    """Upload router JSON data to Modal volume."""
    import os

    router_dir = f"{VOLUME_PATH}/router"
    router_path = f"{router_dir}/semantic_router.json"

    os.makedirs(router_dir, exist_ok=True)

    with open(router_path, "w", encoding="utf-8") as f:
        f.write(router_data)

    print(f"Uploaded router data to {router_path}")
    print(f"Size: {len(router_data) / 1024:.1f} KB")

    # Commit changes to volume
    volume.commit()
    print("Volume committed successfully")


@app.local_entrypoint()
def main():
    """Upload local router data to Modal volume."""
    print("=" * 60)
    print("Uploading Semantic Router Data to Modal Volume")
    print("=" * 60)

    if not LOCAL_ROUTER_PATH.exists():
        print(f"\nERROR: Router data not found at {LOCAL_ROUTER_PATH}")
        print("\nTo create the router data, run:")
        print("  pipenv run python -m src.models.router.train --save")
        return

    print(f"\nReading router data from: {LOCAL_ROUTER_PATH}")
    router_data = LOCAL_ROUTER_PATH.read_text(encoding="utf-8")
    print(f"Router data size: {len(router_data) / 1024:.1f} KB")

    print("\nUploading to Modal volume...")
    upload_router.remote(router_data)

    print("\n" + "=" * 60)
    print("Done! Router data is now available at /vol/router/semantic_router.json")
    print("=" * 60)
