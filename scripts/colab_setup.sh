#!/usr/bin/env bash
# Install and start Ollama (GPU-accelerated on Colab), then pull the model
# used by selection/llm_selector.py. Paste into a Colab cell with a leading
# "!", or run directly on any Linux box with a GPU.
set -euo pipefail

# The Ollama installer needs zstd, which minimal images (e.g. Colab) lack.
if ! command -v zstd >/dev/null 2>&1; then
    (sudo -n true 2>/dev/null && sudo apt-get update -qq && sudo apt-get install -y -qq zstd) \
        || (apt-get update -qq && apt-get install -y -qq zstd)
fi

curl -fsSL https://ollama.com/install.sh | sh

# Start the server in the background -- a Colab cell can't block forever.
nohup ollama serve > /tmp/ollama_serve.log 2>&1 &
sleep 5

ollama pull qwen2.5-coder:7b

echo "Ollama is up. Check with: curl -s http://localhost:11434/api/tags"
echo "Loaded models:           ollama ps"
