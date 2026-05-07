# Micro-Fab VLM

Micro-Fab VLM is a one-week sprint project for turning a former RL training
pipeline into a ground-truth vision pipeline for semiconductor optical defect
inspection. It generates synthetic wafer images, fine-tunes a multimodal
Gemma-4 model with GRPO, merges the LoRA adapter into a 16-bit master
checkpoint, and prepares the model for high-throughput edge deployment.

## Current Focus

The current focus is Phase 3: merging the trained Micro-Fab vision LoRA adapter
and preparing a quantized deployment checkpoint. Phase 1 and Phase 2 remain in
the repo as reproducible data generation and GRPO training steps.

Each label is JSON with a defect class and normalized YOLO bounding box:

```json
{
  "defect": "scratch",
  "bbox_yolo": [0.512, 0.428, 0.214, 0.037]
}
```

## Project Layout

```text
micro-fab-llm/
├── context/
│   └── micro-fab.md
├── micro_fab_llm/
│   ├── synthetic_defects.py
│   ├── train_grpo.py
│   ├── merge_lora_gemma4.py
│   ├── quantize_vlm.py
│   └── serve_llm.py
├── tests/
├── requirements.txt
├── train_requirements.txt
├── pyproject.toml
└── README.md
```

Generated data is written to:

```text
dataset/
├── images/
│   └── wafer_0000.png
└── labels/
    └── wafer_0000.json
```

## Setup

Create and activate a local virtual environment:

```bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
```

Install project dependencies:

```bash
uv pip install -r requirements.txt
```

`requirements.txt` is pinned to the versions currently installed in `.venv`.
`train_requirements.txt` is kept as the smaller training-only lock file.

If you prefer to install the package metadata directly:

```bash
uv pip install -e ".[dev]"
```

## Generate Synthetic Data

Generate the default 500-image dataset:

```bash
.venv/bin/python -m micro_fab_llm.synthetic_defects
```

Generate a smaller smoke-test dataset:

```bash
.venv/bin/python -m micro_fab_llm.synthetic_defects --count 10 --output-dir /tmp/micro_fab_smoke --seed 123
```

Useful options:

```bash
python -m micro_fab_llm.synthetic_defects \
  --count 4000 \
  --output-dir dataset \
  --image-size 512 \
  --seed 42
```

## Synthetic Generator Details

The generator currently supports three mock defect classes:

- `scratch`: jagged dark polyline with randomized thickness and inflection points.
- `particle`: bright blob made from overlapping ellipses and circles.
- `bridge_short`: connected pad-like defect with randomized pad dimensions, spacing, and bridge thickness.

Before blending, each selected defect is randomly scaled between `0.5x` and
`1.5x`, rotated between `0` and `360` degrees on an expanded canvas, and then
trimmed to its non-transparent bounds. The JSON bounding box is calculated from
the final alpha mask used for injection.

## Train The LoRA Adapter

The GRPO trainer reads the generated `dataset/images` and `dataset/labels`
pairs, loads images through PIL, formats Gemma-4 VLM prompts, and saves the
LoRA adapter to `micro_fab_vision_lora` by default. The command is for A100 GPU training.

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

For a quick local smoke test, reduce `--num-generations`, batch size, and
`--max-train-samples`.

## Merge The 16-bit Master

After training, merge the LoRA adapter into the base multimodal Gemma-4 model.
The merge script uses Unsloth `FastVisionModel` so the vision projection layers
are handled with the same path used during fine-tuning.

```bash
python -m micro_fab_llm.merge_lora_gemma4
```

Equivalent explicit command:

```bash
python -m micro_fab_llm.merge_lora_gemma4 \
  --lora-dir micro_fab_vision_lora \
  --base-model unsloth/gemma-4-E4B-it \
  --output-dir micro_fab_vlm_merged_16bit \
  --torch-dtype bfloat16 \
  --max-seq-length 4096
```

The output directory `micro_fab_vlm_merged_16bit` is the 16-bit master
checkpoint used for post-training quantization and deployment testing.

## Quantize For Deployment

The PTQ script calibrates on real synthetic wafer images and exports a
vLLM-ready 8-bit checkpoint:

```bash
python -m micro_fab_llm.quantize_vlm \
  --model-dir micro_fab_vlm_merged_16bit \
  --dataset-dir dataset \
  --num-calibration-samples 32 \
  --output-dir micro_fab_vlm_deployed_8bit
```

`llmcompressor==0.10.0.2` is listed as the `quantization` extra in
`pyproject.toml`. Its current resolver requirements differ from the active
Unsloth training stack, so use a separate deployment/quantization environment
if pip attempts to move `torch` or `transformers`.

## Roadmap

The remaining sprint phases focus on serving and latency validation:

- Refactor the FastAPI gateway to accept base64 images and return JSON defect diagnostics.
- Validate vLLM latency on a single 24GB RTX 3090 against the 50ms target.
- Add regression tests for strict JSON output parsing and calibration sample loading.
