#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-micro-fab-llm}"

mkdir -p "${PROJECT_ROOT}/context"
mkdir -p "${PROJECT_ROOT}/dataset/images"
mkdir -p "${PROJECT_ROOT}/dataset/labels"
mkdir -p "${PROJECT_ROOT}/micro_fab_llm"
mkdir -p "${PROJECT_ROOT}/tests"

touch "${PROJECT_ROOT}/micro_fab_llm/__init__.py"
touch "${PROJECT_ROOT}/micro_fab_llm/synthetic_defects.py"
touch "${PROJECT_ROOT}/micro_fab_llm/train_grpo.py"
touch "${PROJECT_ROOT}/micro_fab_llm/merge_lora_gemma4.py"
touch "${PROJECT_ROOT}/micro_fab_llm/quantize_vlm.py"
touch "${PROJECT_ROOT}/micro_fab_llm/serve_llm.py"
touch "${PROJECT_ROOT}/tests/__init__.py"
touch "${PROJECT_ROOT}/context/micro-fab.md"

if [[ ! -f "${PROJECT_ROOT}/requirements.txt" ]]; then
  cat > "${PROJECT_ROOT}/requirements.txt" <<'REQS'
# --- Core Deep Learning & Fine-Tuning ---
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
unsloth==2026.5.2
unsloth_zoo==2026.5.1
trl==0.24.0
peft==0.19.1
accelerate==1.13.0
bitsandbytes==0.49.2
transformers==5.5.0
datasets==4.3.0
wandb==0.26.1
xformers==0.0.32.post1

# --- Vision & Data Processing ---
opencv-python==4.13.0.92
Pillow==12.2.0
numpy==2.2.6

# --- High-Throughput Deployment ---
vllm==0.11.0
fastapi==0.136.1
uvicorn==0.46.0
pydantic==2.13.3

# --- Testing & Validation ---
pytest==9.0.3
aiohttp==3.13.5
REQS
fi

if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
  cat > "${PROJECT_ROOT}/pyproject.toml" <<'PYPROJECT'
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "micro-fab-llm"
version = "0.1.0"
description = "GPU-native Micro-Fab VLM (Gemma-4-E4B)"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = [
  "torch==2.8.0",
  "torchvision==0.23.0",
  "torchaudio==2.8.0",
  "unsloth==2026.5.2",
  "unsloth_zoo==2026.5.1",
  "trl==0.24.0",
  "peft==0.19.1",
  "accelerate==1.13.0",
  "bitsandbytes==0.49.2",
  "transformers==5.5.0",
  "datasets==4.3.0",
  "wandb==0.26.1",
  "xformers==0.0.32.post1",
  "opencv-python==4.13.0.92",
  "Pillow==12.2.0",
  "numpy==2.2.6",
  "vllm==0.11.0",
  "fastapi==0.136.1",
  "uvicorn==0.46.0",
  "pydantic==2.13.3",
  "aiohttp==3.13.5",
]

[project.optional-dependencies]
dev = ["pytest==9.0.3"]
quantization = ["llmcompressor==0.10.0.2"]

[tool.setuptools.packages.find]
where = ["."]
include = ["micro_fab_llm*"]
PYPROJECT
fi

if [[ ! -f "${PROJECT_ROOT}/README.md" ]]; then
  cat > "${PROJECT_ROOT}/README.md" <<'README'
# Micro-Fab VLM

GPU-native Gemma-4 vision-language pipeline for semiconductor optical defect inspection.

## Quick Start

```bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements.txt
python -m micro_fab_llm.synthetic_defects
```

## Train

```bash
python -m micro_fab_llm.train_grpo --dataset-dir dataset \
  --max-seq-length 4096 \
  --max-prompt-length 2048 \
  --lora-r 32 \
  --lora-alpha 64 \
  --num-generations 8 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --optim adamw_8bit \
  --use-wandb \
  --wandb-run-name "grpo_run_$(date +%Y%m%d_%H%M%S)" \
  --wandb-tags "grpo,synthetic_defects" \
  --wandb-mode "online"
```

## Merge

```bash
python -m micro_fab_llm.merge_lora_gemma4
```
README
fi

echo "Project structure created at ${PROJECT_ROOT}"