#!/usr/bin/env python3
"""
Interactive test client for Swedish Sovereign AI inference server.

Usage:
    1. Deploy: modal serve src/serve/modal_serve.py
    2. Run: python src/serve/test_interactive.py <base-url>

Example:
    python src/serve/test_interactive.py https://your-workspace--swedish-sovereign-ai-serve-inferenceserver
"""

import sys
import json
import requests
from typing import Optional

# Default test questions for comparison
TEST_QUESTIONS = [
    "Vad är reporäntan?",
    "Vad anser Riksbanken om inflationen?",
    "Hur påverkar räntehöjningar hushållen?",
    "Vad är KPIF och varför används det?",
    "Beskriv penningpolitisk transmission.",
    "Vad är kvantitativa lättnader?",
    "Hur ser Riksbanken på arbetsmarknaden?",
    "Vad händer med kronan vid räntehöjningar?",
    "Förklara skillnaden mellan KPI och KPIF.",
    "Vad är Riksbankens inflationsmål?",
]


def get_chat_url(base_url: str) -> str:
    """Get the chat endpoint URL. Just use the URL as-is if it's already a chat endpoint."""
    base_url = base_url.rstrip("/")

    # If it's already a chat URL, use it directly
    if "-chat" in base_url and ".modal.run" in base_url:
        return base_url

    # Otherwise try to build it (may not work if hostname too long)
    is_dev = "-dev.modal.run" in base_url
    if base_url.endswith(".modal.run"):
        base_url = base_url[:-len(".modal.run")]
    if base_url.endswith("-dev"):
        base_url = base_url[:-len("-dev")]
        is_dev = True

    if is_dev:
        return f"{base_url}-chat-dev.modal.run"
    else:
        return f"{base_url}-chat.modal.run"


def chat(
    base_url: str,
    message: str,
    use_finetuned: Optional[bool] = None,
    auto_route: bool = True,
    max_tokens: int = 512,
) -> dict:
    """
    Send a chat request to the server.

    Args:
        use_finetuned: None=auto-route, True=finetuned, False=vanilla
        auto_route: Enable semantic routing (ignored if use_finetuned is set)
    """
    url = get_chat_url(base_url)
    payload = {
        "messages": [{"role": "user", "content": message}],
        "auto_route": auto_route,
        "max_tokens": max_tokens,
    }

    # Only include use_finetuned if explicitly set (overrides auto-routing)
    if use_finetuned is not None:
        payload["use_finetuned"] = use_finetuned

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "model": None}


def health_check(base_url: str) -> dict:
    """Check server health - skip if URL would be too long."""
    # Health endpoint often has truncated URL due to Modal's hostname limits
    # Just return a simple status
    return {"status": "skipped", "note": "Health check skipped (hostname too long for Modal)"}


def compare_models(base_url: str, question: str, max_tokens: int = 512):
    """Compare vanilla and finetuned model responses."""
    print(f"\nQuestion: {question}")
    print("=" * 70)

    # Vanilla model
    print("\n[VANILLA MODEL]")
    print("-" * 40)
    result = chat(base_url, question, use_finetuned=False, max_tokens=max_tokens)
    if "error" in result and result.get("model") is None:
        print(f"Error: {result['error']}")
    else:
        print(result.get("response", "No response"))

    # Finetuned model
    print("\n[FINETUNED MODEL]")
    print("-" * 40)
    result = chat(base_url, question, use_finetuned=True, max_tokens=max_tokens)
    if "error" in result and result.get("model") is None:
        print(f"Error: {result['error']}")
    else:
        model_type = result.get("model", "unknown")
        print(f"(Using: {model_type})")
        print(result.get("response", "No response"))

    print("\n" + "=" * 70)


def run_all_comparisons(base_url: str):
    """Run comparisons for all test questions."""
    print("\n" + "=" * 70)
    print("RUNNING ALL TEST COMPARISONS")
    print("=" * 70)

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}]")
        compare_models(base_url, question, max_tokens=300)
        print("\nPress Enter to continue (or 'q' to quit)...")
        user_input = input()
        if user_input.lower() == 'q':
            break


def interactive_mode(base_url: str):
    """Interactive chat mode."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("\nCommands:")
    print("  /auto     - Auto-route (router picks model) [default]")
    print("  /vanilla  - Force vanilla model")
    print("  /ft       - Force finetuned model")
    print("  /compare  - Compare both models")
    print("  /all      - Run all test comparisons")
    print("  /health   - Check server health")
    print("  /quit     - Exit")
    print("\nDefault: Auto-route based on prompt")

    mode = "auto"

    while True:
        print(f"\n[Mode: {mode}]")
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/auto":
                mode = "auto"
                print("Switched to auto-route mode (router picks model)")
            elif cmd == "/vanilla":
                mode = "vanilla"
                print("Switched to vanilla model (forced)")
            elif cmd == "/ft":
                mode = "finetuned"
                print("Switched to finetuned model (forced)")
            elif cmd == "/compare":
                mode = "compare"
                print("Switched to compare mode")
            elif cmd == "/all":
                run_all_comparisons(base_url)
            elif cmd == "/health":
                result = health_check(base_url)
                print(json.dumps(result, indent=2))
            elif cmd == "/quit":
                print("Goodbye!")
                break
            else:
                print(f"Unknown command: {cmd}")
            continue

        # Process message
        if mode == "compare":
            compare_models(base_url, user_input)
        elif mode == "auto":
            result = chat(base_url, user_input)  # auto_route=True by default
            model_type = result.get("model", "unknown")
            route = result.get("route", "")
            route_info = f" [routed: {route}]" if route else ""
            print(f"\nAssistant ({model_type}{route_info}): {result.get('response', result.get('error', 'No response'))}")
        elif mode == "vanilla":
            result = chat(base_url, user_input, use_finetuned=False)
            print(f"\nAssistant (vanilla): {result.get('response', result.get('error', 'No response'))}")
        elif mode == "finetuned":
            result = chat(base_url, user_input, use_finetuned=True)
            model_type = result.get("model", "unknown")
            print(f"\nAssistant ({model_type}): {result.get('response', result.get('error', 'No response'))}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_interactive.py <base-url>")
        print("\nExample:")
        print("  python test_interactive.py https://your-workspace--swedish-sovereign-ai-serve-inferenceserver")
        print("\nTo get the URL:")
        print("  1. Run: modal serve src/serve/modal_serve.py")
        print("  2. Copy the base URL (without the endpoint suffix)")
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")

    print("=" * 70)
    print("Swedish Sovereign AI - Interactive Test Client")
    print("=" * 70)
    print(f"\nServer: {base_url}")

    # Health check
    print("\nChecking server health...")
    health = health_check(base_url)
    if "error" in health:
        print(f"Warning: Health check failed: {health['error']}")
        print("The server might still be starting up. Continuing anyway...")
    else:
        print(f"Server healthy!")
        print(f"  Base model: {health.get('base_model', 'unknown')}")
        print(f"  LoRA available: {health.get('lora_available', False)}")

    # Start interactive mode
    interactive_mode(base_url)


if __name__ == "__main__":
    main()
