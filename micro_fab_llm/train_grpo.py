"""Phase 2 multimodal GRPO training for semiconductor defect inspection.

The trainer consumes the synthetic dataset produced by ``synthetic_defects.py``:

```
dataset/
|-- images/wafer_0000.png
`-- labels/wafer_0000.json
```

Each training sample contains a Gemma-4 vision prompt, a PIL image, and the
ground-truth defect class. Rewards are computed by strict JSON verification.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import inspect
import json
import logging
from pathlib import Path
import random
from typing import Any, cast

import torch
from datasets import Dataset
from PIL import Image as PILImage

import unsloth  # Required before importing trl so Unsloth can patch the trainer.
import trl
from unsloth import FastVisionModel, get_chat_template

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


LOGGER = logging.getLogger(__name__)

GRPOConfig = getattr(trl, "GRPOConfig")
GRPOTrainer = getattr(trl, "GRPOTrainer")

SEED = 3407
DEFAULT_DATASET_DIR = Path("dataset")
DEFAULT_SAVE_PATH = "micro_fab_vision_lora"
DEFECT_TYPES = ("scratch", "particle", "bridge_short")


@dataclass
class TrainConfig:
    """Configuration for Gemma-4 vision GRPO training."""

    model_name: str = "unsloth/gemma-4-E4B-it"
    dataset_dir: Path = DEFAULT_DATASET_DIR
    output_dir: str = "grpo_micro_fab_outputs"
    save_path: str = DEFAULT_SAVE_PATH
    cache_dir: str = "./tmp_model_cache"
    logging_dir: str | None = None
    report_to: str = "none"

    max_train_samples: int | None = None
    num_train_epochs: float = 2.0
    max_steps: int = -1

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    max_grad_norm: float = 0.1
    temperature: float = 0.9

    max_seq_length: int = 4096
    max_prompt_length: int = 2048
    max_completion_length: int = 96
    num_generations: int = 2

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True

    load_in_4bit: bool = True
    fast_inference: bool = False
    gpu_memory_utilization: float = 0.8

    seed: int = SEED
    logging_steps: int = 10
    save_steps: int = 100
    log_completions: bool = False
    mask_truncated_completions: bool = False
    use_gspo: bool = False
    debug: bool = False

    use_wandb: bool = False
    wandb_project: str = "micro-fab-vlm"
    wandb_entity: str | None = "hieult"
    wandb_run_name: str | None = None
    wandb_mode: str = "offline"
    wandb_tags: str | None = None


def _parse_args() -> TrainConfig:
    """Parse CLI args into a training configuration."""

    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train Gemma-4-E4B vision GRPO on synthetic wafer defects.")

    parser.add_argument("--model-name", type=str, default=defaults.model_name)
    parser.add_argument("--dataset-dir", type=Path, default=defaults.dataset_dir)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means use the full dataset.")
    parser.add_argument("--num-train-epochs", type=float, default=defaults.num_train_epochs)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)

    parser.add_argument("--per-device-train-batch-size", type=int, default=defaults.per_device_train_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults.gradient_accumulation_steps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--lr-scheduler-type", type=str, default=defaults.lr_scheduler_type)
    parser.add_argument("--optim", type=str, default=defaults.optim)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)

    parser.add_argument("--max-seq-length", type=int, default=defaults.max_seq_length)
    parser.add_argument("--max-prompt-length", type=int, default=defaults.max_prompt_length)
    parser.add_argument("--max-completion-length", type=int, default=defaults.max_completion_length)
    parser.add_argument("--num-generations", type=int, default=defaults.num_generations)

    parser.add_argument("--lora-r", type=int, default=defaults.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--finetune-vision-layers", action="store_true", default=defaults.finetune_vision_layers)
    parser.add_argument("--freeze-language-layers", action="store_true")
    parser.add_argument("--freeze-attention-modules", action="store_true")
    parser.add_argument("--freeze-mlp-modules", action="store_true")

    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=defaults.load_in_4bit)
    parser.add_argument("--fast-inference", action="store_true", default=defaults.fast_inference)
    parser.add_argument("--gpu-memory-utilization", type=float, default=defaults.gpu_memory_utilization)

    parser.add_argument("--output-dir", type=str, default=defaults.output_dir)
    parser.add_argument("--save-path", type=str, default=defaults.save_path)
    parser.add_argument("--cache-dir", type=str, default=defaults.cache_dir)
    parser.add_argument("--report-to", type=str, default=defaults.report_to)
    parser.add_argument("--logging-dir", type=str, default=defaults.logging_dir)
    parser.add_argument("--logging-steps", type=int, default=defaults.logging_steps)
    parser.add_argument("--save-steps", type=int, default=defaults.save_steps)
    parser.add_argument("--log-completions", action="store_true", default=defaults.log_completions)
    parser.add_argument("--mask-truncated-completions", action="store_true", default=defaults.mask_truncated_completions)
    parser.add_argument("--use-gspo", action="store_true", default=defaults.use_gspo)

    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=defaults.wandb_project)
    parser.add_argument("--wandb-entity", type=str, default=defaults.wandb_entity)
    parser.add_argument("--wandb-run-name", type=str, default=defaults.wandb_run_name)
    parser.add_argument("--wandb-mode", type=str, default=defaults.wandb_mode)
    parser.add_argument("--wandb-tags", type=str, default=defaults.wandb_tags)

    args = parser.parse_args()
    return TrainConfig(
        model_name=args.model_name,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        save_path=args.save_path,
        cache_dir=args.cache_dir,
        logging_dir=args.logging_dir,
        report_to=args.report_to,
        max_train_samples=args.max_train_samples if args.max_train_samples > 0 else None,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        temperature=args.temperature,
        max_seq_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=not args.freeze_language_layers,
        finetune_attention_modules=not args.freeze_attention_modules,
        finetune_mlp_modules=not args.freeze_mlp_modules,
        load_in_4bit=args.load_in_4bit,
        fast_inference=args.fast_inference,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        log_completions=args.log_completions,
        mask_truncated_completions=args.mask_truncated_completions,
        use_gspo=args.use_gspo,
        debug=args.debug,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        wandb_tags=args.wandb_tags,
    )


def _setup_logging(debug: bool) -> None:
    """Configure process-wide logging."""

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", force=True)


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy if available, and Torch."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        LOGGER.debug("NumPy is not installed; skipping NumPy seeding.")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_defect_prompt() -> list[dict[str, Any]]:
    """Return a Gemma-4 vision prompt in TRL's conversational VLM format."""

    defect_options = ", ".join(DEFECT_TYPES)
    user_prompt = (
        "You are a semiconductor optical inspection VLM. "
        "Classify exactly one visible wafer defect from the allowed classes. "
        "Inspect the wafer image and classify the injected defect. "
        f"The only allowed defect values are: {defect_options}. "
        "Return exactly one JSON object with this schema: "
        "{\"defect\": \"scratch|particle|bridge_short\"}. "
        "Do not include markdown, prose, coordinates, or extra keys. "
        "After the JSON object, immediately output STOP."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def load_defect_dataset(dataset_dir: Path) -> Dataset:
    """Load synthetic wafer images and JSON labels into a multimodal GRPO dataset."""

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    records: list[dict[str, Any]] = []
    for label_path in sorted(labels_dir.glob("*.json")):
        image_path = images_dir / f"{label_path.stem}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image for label {label_path.name}: {image_path}")

        label = json.loads(label_path.read_text(encoding="utf-8"))
        defect = label.get("defect")
        if not isinstance(defect, str) or not defect.strip():
            raise ValueError(f"Label {label_path} must contain a non-empty string 'defect' field.")

        with PILImage.open(image_path) as image:
            rgb_image = image.convert("RGB")

        records.append(
            {
                "prompt": _format_defect_prompt(),
                "image": rgb_image,
                "ground_truth": _normalize_defect(defect),
            }
        )

    if not records:
        raise ValueError(f"No JSON labels found in {labels_dir}")

    return Dataset.from_list(records)


def _extract_completion_text(completion_sample: Any) -> str:
    """Extract generated text across TRL completion payload variants."""

    if completion_sample is None:
        return ""

    if isinstance(completion_sample, str):
        return completion_sample

    if isinstance(completion_sample, dict):
        content = completion_sample.get("content")
        if isinstance(content, list):
            text_parts = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") in {None, "text"}
            ]
            return "\n".join(part for part in text_parts if part)

        for key in ("content", "text", "generated_text", "completion", "response", "output_text"):
            if key in completion_sample:
                return _extract_completion_text(completion_sample[key])
        return ""

    if isinstance(completion_sample, (list, tuple)):
        if completion_sample and all(isinstance(item, dict) for item in completion_sample):
            for item in reversed(completion_sample):
                text = _extract_completion_text(item)
                if text.strip():
                    return text
        parts = [_extract_completion_text(item) for item in completion_sample]
        return "\n".join(part for part in parts if part)

    return str(completion_sample)


def _parse_defect_from_completion(completion_sample: Any, *, debug: bool = False) -> str | None:
    """Parse the ``defect`` field from the first valid JSON object in a completion."""

    text = _extract_completion_text(completion_sample)
    if not text:
        return None
    if debug:
        LOGGER.debug("Raw completion text: %s", text[:1000])

    decoder = json.JSONDecoder()
    parse_text = text.split("STOP", 1)[0]
    for start_index, character in enumerate(parse_text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(parse_text[start_index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "defect" not in payload:
            continue
        return _normalize_defect(payload["defect"])

    return None


def _normalize_defect(value: Any) -> str:
    """Normalize defect class text for reward comparison."""

    return str(value).strip().lower()


def accuracy_reward_func(
    prompts: list[Any],
    completions: list[Any],
    ground_truth: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward strict JSON defect classification against ground truth labels."""

    del prompts
    rewards: list[float] = []
    parsed_count = 0
    correct_count = 0

    for completion, truth in zip(completions, ground_truth):
        parsed_defect = _parse_defect_from_completion(completion)
        if parsed_defect is None:
            rewards.append(-2.0)
            continue

        parsed_count += 1
        if parsed_defect == _normalize_defect(truth):
            correct_count += 1
            rewards.append(2.0)
        else:
            rewards.append(-1.0)

    total = len(rewards)
    parse_success = float(parsed_count / total) if total else 0.0
    classification_accuracy = float(correct_count / total) if total else 0.0

    log_metric = kwargs.get("log_metric")
    if callable(log_metric):
        log_metric("parse_success", parse_success)
        log_metric("classification_accuracy", classification_accuracy)

    if total:
        LOGGER.info(
            "accuracy_reward parse_success=%.3f classification_accuracy=%.3f correct=%d/%d",
            parse_success,
            classification_accuracy,
            correct_count,
            total,
        )

    return rewards


def _build_grpo_config(config: TrainConfig) -> Any:
    """Build GRPOConfig while filtering unavailable keys across TRL versions."""

    bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    report_to = "wandb" if config.use_wandb else config.report_to
    grpo_config_kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "optim": "adamw_8bit",
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "lr_scheduler_type": config.lr_scheduler_type,
        "optim": config.optim,
        "max_grad_norm": config.max_grad_norm,
        "temperature": config.temperature,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": config.max_completion_length,
        "num_generations": config.num_generations,
        "log_completions": config.log_completions,
        "mask_truncated_completions": config.mask_truncated_completions,
        "remove_unused_columns": False,
        "bf16": bf16_enabled,
        "fp16": not bf16_enabled,
        "report_to": report_to,
        "seed": config.seed,
    }
    if config.use_gspo:
        grpo_config_kwargs["importance_sampling_level"] = "sequence"
        grpo_config_kwargs["loss_type"] = "dr_grpo"
    if config.max_steps != -1:
        grpo_config_kwargs["max_steps"] = config.max_steps
    if config.logging_dir:
        grpo_config_kwargs["logging_dir"] = config.logging_dir

    grpo_config_params = inspect.signature(GRPOConfig.__init__).parameters
    filtered_kwargs = {key: value for key, value in grpo_config_kwargs.items() if key in grpo_config_params}
    dropped_keys = sorted(set(grpo_config_kwargs) - set(filtered_kwargs))
    if dropped_keys:
        LOGGER.debug("Ignoring unsupported GRPOConfig keys for this TRL version: %s", dropped_keys)

    return GRPOConfig(**filtered_kwargs)


def _build_trainer(model: Any, processor: Any, train_dataset: Dataset, config: Any) -> Any:
    """Build GRPOTrainer with compatibility fallback for processor argument names."""

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": config,
        "train_dataset": train_dataset,
        "reward_funcs": [accuracy_reward_func],
    }

    trainer_init_params = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = processor
    else:
        raise RuntimeError(
            "GRPOTrainer.__init__ does not expose processing_class or tokenizer. "
            "Please update TRL/Unsloth."
        )

    return GRPOTrainer(**trainer_kwargs)


def _maybe_init_wandb(config: TrainConfig, train_dataset: Dataset) -> Any | None:
    """Initialize W&B if requested and available."""

    if not config.use_wandb:
        return None
    if wandb is None:
        LOGGER.warning("wandb is not installed; continuing without W&B tracking.")
        return None

    tags = [tag.strip() for tag in config.wandb_tags.split(",") if tag.strip()] if config.wandb_tags else None
    effective_run_name = config.wandb_run_name or f"{config.wandb_project}_{datetime.now().strftime('%y%m%d')}"
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=effective_run_name,
        mode=cast(Any, config.wandb_mode),
        tags=tags,
        config={**asdict(config), "dataset_dir": str(config.dataset_dir)},
    )
    if run is not None:
        run.summary["train_dataset_size"] = len(train_dataset)
    return run


def _load_model_and_tokenizer(config: TrainConfig) -> tuple[Any, Any]:
    """Load Gemma-4 vision model/processor and attach LoRA adapters."""

    model, processor = FastVisionModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
        fast_inference=config.fast_inference,
        gpu_memory_utilization=config.gpu_memory_utilization,
        cache_dir=config.cache_dir,
    )
    processor = get_chat_template(processor, "gemma-4")
    _ensure_padding_token(processor)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=config.finetune_vision_layers,
        finetune_language_layers=config.finetune_language_layers,
        finetune_attention_modules=config.finetune_attention_modules,
        finetune_mlp_modules=config.finetune_mlp_modules,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    return model, processor


def _ensure_padding_token(processor: Any) -> None:
    """Make sure the processor or nested tokenizer can pad GRPO batches."""

    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"


def main() -> None:
    """Entry point for multimodal GRPO training."""

    config = _parse_args()
    _setup_logging(config.debug)
    _set_seed(config.seed)

    serializable_config = {**asdict(config), "dataset_dir": str(config.dataset_dir)}
    LOGGER.info("Starting Micro-Fab VLM GRPO run on cuda=%s", torch.cuda.is_available())
    LOGGER.info("Training config: %s", json.dumps(serializable_config, sort_keys=True, indent=2))

    train_dataset = load_defect_dataset(config.dataset_dir)
    if config.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(config.max_train_samples, len(train_dataset))))
    LOGGER.info("Loaded %d synthetic defect samples from %s", len(train_dataset), config.dataset_dir)
    if config.debug:
        LOGGER.debug("Sample prompt: %s", train_dataset[0]["prompt"])
        LOGGER.debug("Sample ground_truth: %s", train_dataset[0]["ground_truth"])

    model, processor = _load_model_and_tokenizer(config)
    wandb_run = _maybe_init_wandb(config, train_dataset)

    grpo_config = _build_grpo_config(config)
    trainer = _build_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        config=grpo_config,
    )

    LOGGER.info("Starting training...")
    trainer.train()
    LOGGER.info("Training complete. Saving LoRA artifacts to %s", config.save_path)

    model.save_pretrained(config.save_path)
    processor.save_pretrained(config.save_path)

    if wandb_run is not None:
        wandb_run.summary["save_path"] = config.save_path
        wandb_run.finish()


if __name__ == "__main__":
    main()
