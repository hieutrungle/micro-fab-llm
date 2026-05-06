# Micro-Fab VLM

Micro-Fab VLM is a one-week sprint project for turning a former RL training
pipeline into a ground-truth vision pipeline for semiconductor optical defect
inspection. The current phase generates synthetic wafer images and strict JSON
labels that can be used to fine-tune a multimodal Gemma model.

## Current Focus

Phase 1 builds a synthetic defect dataset with OpenCV. The generator creates
brushed-metal wafer backgrounds, produces transparent mock defect assets, adds
random geometric and color variation, blends defects with `cv2.seamlessClone`,
and writes paired image/label files.

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
│   └── serve_pricing_llm.py
├── tests/
├── requirements.txt
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
python -m venv .venv
source .venv/bin/activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

If you prefer to avoid activating the environment, run commands through the
local interpreter:

```bash
.venv/bin/python -m pip install -r requirements.txt
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
.venv/bin/python -m micro_fab_llm.synthetic_defects \
  --count 500 \
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

## Roadmap

The remaining sprint phases refactor training and serving around image inputs:

- Replace the old RL reward path with ground-truth JSON verification.
- Load wafer images through PIL for multimodal GRPO fine-tuning.
- Merge and quantize LoRA adapters for a 24GB GPU target.
- Refactor the FastAPI gateway to accept base64 images and return JSON defect diagnostics.
