#!/bin/bash
# Test script for Swedish Sovereign AI inference server
#
# Usage:
#   1. Deploy the server: modal serve src/serve/modal_serve.py
#   2. Copy the URL from the output
#   3. Run: ./src/serve/test_requests.sh <base-url>
#
# Example:
#   ./src/serve/test_requests.sh https://your-workspace--swedish-sovereign-ai-serve-inferenceserver

set -e

BASE_URL="${1:-http://localhost:8000}"

echo "=========================================="
echo "Testing Swedish Sovereign AI Server"
echo "Base URL: $BASE_URL"
echo "=========================================="

# Health check
echo -e "\n[1] Health Check"
echo "----------------"
curl -s "$BASE_URL-health.modal.run" | python3 -m json.tool 2>/dev/null || curl -s "$BASE_URL-health.modal.run"

# Test vanilla model
echo -e "\n\n[2] Vanilla Model - Simple Question"
echo "------------------------------------"
curl -s -X POST "$BASE_URL-chat.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Vad är reporäntan?"}],
    "use_finetuned": false,
    "max_tokens": 256
  }' | python3 -m json.tool 2>/dev/null || echo "Request failed"

# Test finetuned model
echo -e "\n\n[3] Finetuned Model - Same Question"
echo "------------------------------------"
curl -s -X POST "$BASE_URL-chat.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Vad är reporäntan?"}],
    "use_finetuned": true,
    "max_tokens": 256
  }' | python3 -m json.tool 2>/dev/null || echo "Request failed"

# Riksbanken inflation question
echo -e "\n\n[4] Finetuned Model - Riksbanken Inflation"
echo "-------------------------------------------"
curl -s -X POST "$BASE_URL-chat.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Vad anser Riksbanken om inflationen?"}],
    "use_finetuned": true,
    "max_tokens": 512
  }' | python3 -m json.tool 2>/dev/null || echo "Request failed"

# Multi-turn conversation
echo -e "\n\n[5] Multi-turn Conversation"
echo "---------------------------"
curl -s -X POST "$BASE_URL-chat.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Vad är KPIF?"},
      {"role": "assistant", "content": "KPIF står för konsumentprisindex med fast ränta."},
      {"role": "user", "content": "Varför använder Riksbanken det istället för vanlig KPI?"}
    ],
    "use_finetuned": true,
    "max_tokens": 512
  }' | python3 -m json.tool 2>/dev/null || echo "Request failed"

echo -e "\n\n=========================================="
echo "Tests complete!"
echo "=========================================="
