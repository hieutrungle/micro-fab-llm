"""Merge a vision LoRA adapter fine-tuned with Unsloth/GRPO into Gemma-4.

This script folds the Micro-Fab VLM LoRA weights into the base multimodal
Gemma-4 checkpoint and saves a 16-bit Hugging Face checkpoint suitable for
post-training quantization and vLLM deployment.
"""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - depends on training environment
    torch = None  # type: ignore[assignment]

try:
    from unsloth import FastVisionModel
except ImportError:  # pragma: no cover - optional runtime dependency
    FastVisionModel = None


LOGGER = logging.getLogger(__name__)

DEFAULT_LORA_DIR = Path("micro_fab_vision_lora")
DEFAULT_BASE_MODEL = "unsloth/gemma-4-E4B-it"
DEFAULT_OUTPUT_DIR = Path("micro_fab_vlm_merged_16bit")
DEFAULT_MAX_SEQ_LENGTH = 4096

REQUIRED_OUTPUT_FILES = ("config.json", "tokenizer_config.json")
MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "spiece.model")
PROCESSOR_FILES = ("preprocessor_config.json", "processor_config.json", "image_processor_config.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a Gemma-4 vision LoRA adapter into the base model.")
    parser.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA_DIR, help="Path to the trained LoRA adapter.")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="Base model ID or local path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the merged model.")
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=("float16", "bfloat16", "float32"),
        help="Weight dtype to use while loading and saving.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Sequence length used when loading with Unsloth.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code when loading the tokenizer or model.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    return parser


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if torch is None:
        raise RuntimeError("Missing dependency: torch. Install the training dependencies before merging.")
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def _load_unsloth_model_and_processor(
    lora_dir: Path,
    base_model: str,
    max_seq_length: int,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    if FastVisionModel is None:
        raise RuntimeError(
            "unsloth is not installed; install the training dependencies before merging the vision adapter."
        )

    kwargs: dict[str, Any] = {
        "model_name": str(lora_dir),
        "max_seq_length": max_seq_length,
        "dtype": torch_dtype,
        "load_in_4bit": False,
        "fast_inference": False,
    }
    signature = inspect.signature(FastVisionModel.from_pretrained)
    if "trust_remote_code" in signature.parameters:
        kwargs["trust_remote_code"] = trust_remote_code
    if "device_map" in signature.parameters:
        kwargs["device_map"] = {"": "cpu"}

    LOGGER.info("Loading vision LoRA adapter via Unsloth from %s", lora_dir)
    LOGGER.info("Adapter should reference base model %s", base_model)
    model, processor = FastVisionModel.from_pretrained(**kwargs)
    _ensure_padding_token(processor)
    return model, processor


def _save_merged_with_unsloth(
    model: Any,
    processor: Any,
    output_dir: Path,
    torch_dtype: torch.dtype,
) -> None:
    if hasattr(model, "save_pretrained_merged"):
        LOGGER.info("Saving merged model with Unsloth save_pretrained_merged")
        try:
            model.save_pretrained_merged(str(output_dir), processor, save_method="merged_16bit")
            return
        except TypeError:
            model.save_pretrained_merged(str(output_dir), processor)
            return

    if not hasattr(model, "merge_and_unload"):
        raise TypeError("Loaded Unsloth model does not support merge_and_unload().")

    LOGGER.info("Merging LoRA adapter into base model weights")
    merged_model = model.merge_and_unload()
    merged_model = merged_model.to(dtype=torch_dtype)
    merged_model.config.torch_dtype = str(torch_dtype).replace("torch.", "")
    LOGGER.info("Saving merged model to %s", output_dir)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)


def _ensure_padding_token(processor: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token


def _validate_saved_artifacts(output_dir: Path) -> None:
    missing: list[str] = []

    for required_file in REQUIRED_OUTPUT_FILES:
        if not (output_dir / required_file).exists():
            missing.append(required_file)

    if not any((output_dir / filename).exists() for filename in MODEL_WEIGHT_FILES):
        missing.append("model weights (.safetensors or .bin)")

    if not any((output_dir / filename).exists() for filename in TOKENIZER_FILES):
        missing.append("tokenizer model file")

    if not any((output_dir / filename).exists() for filename in PROCESSOR_FILES):
        missing.append("vision processor config")

    if missing:
        raise RuntimeError(f"Merged export is incomplete. Missing artifacts: {', '.join(missing)}")


def merge_lora_adapter(
    lora_dir: Path,
    base_model: str,
    output_dir: Path,
    max_seq_length: int,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
) -> None:
    """Load the adapter, merge it into the base weights, and save a standard checkpoint."""
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA directory does not exist: {lora_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = _load_unsloth_model_and_processor(
        lora_dir=lora_dir,
        base_model=base_model,
        max_seq_length=max_seq_length,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    _save_merged_with_unsloth(
        model=model,
        processor=processor,
        output_dir=output_dir,
        torch_dtype=torch_dtype,
    )

    _validate_saved_artifacts(output_dir)

    LOGGER.info("Merge complete. Output directory is ready for deployment: %s", output_dir)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    try:
        torch_dtype = _resolve_dtype(args.torch_dtype)
        lora_dir = args.lora_dir.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        LOGGER.info("Starting vision LoRA merge with dtype=%s", torch_dtype)
        merge_lora_adapter(
            lora_dir=lora_dir,
            base_model=args.base_model,
            output_dir=output_dir,
            max_seq_length=args.max_seq_length,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception:
        LOGGER.exception("Vision LoRA merge failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()