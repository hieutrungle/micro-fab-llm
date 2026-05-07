"""Post-training quantization for the merged Micro-Fab Gemma-4 VLM.

The script calibrates the merged 16-bit checkpoint with real synthetic wafer
images, applies an 8-bit vLLM-compatible llm-compressor recipe, and exports a
compressed deployment checkpoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import inspect
import json
import logging
from pathlib import Path
import random
from typing import Any

from PIL import Image as PILImage

try:
    import torch
except ImportError:  # pragma: no cover - depends on deployment environment
    torch = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("micro_fab_vlm_merged_16bit")
DEFAULT_DATASET_DIR = Path("dataset")
DEFAULT_OUTPUT_DIR = Path("micro_fab_vlm_deployed_8bit")
DEFAULT_NUM_CALIBRATION_SAMPLES = 32
DEFAULT_MAX_SEQ_LENGTH = 4096
DEFAULT_SEED = 3407
DEFAULT_SCHEME = "W8A8"
DEFECT_TYPES = ("scratch", "particle", "bridge_short")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "spiece.model")
PROCESSOR_FILES = ("preprocessor_config.json", "processor_config.json", "image_processor_config.json")
MULTIMODAL_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "image_sizes",
    "image_size",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
)
FLOAT_INPUT_KEYS = ("pixel_values", "image_features", "image_embeds")
VISION_IGNORE_PATTERNS = (
    "re:.*vision_model.*",
    "re:.*vision_tower.*",
    "re:.*visual.*",
    "re:.*multi_modal_projector.*",
    "re:.*mm_projector.*",
    "re:.*vision_projector.*",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantize the merged Micro-Fab Gemma-4 VLM with multimodal PTQ calibration."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Merged 16-bit model directory.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Synthetic dataset root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Quantized model output directory.")
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=DEFAULT_NUM_CALIBRATION_SAMPLES,
        help="Number of image/prompt samples to use for PTQ calibration.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Maximum token length during calibration preprocessing.",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default=DEFAULT_SCHEME,
        help="llm-compressor GPTQ scheme. Use W8A8 for RTX 3090 INT8 deployment.",
    )
    parser.add_argument(
        "--smoothquant-strength",
        type=float,
        default=0.8,
        help="SmoothQuant smoothing strength. Set 0 to disable SmoothQuant.",
    )
    parser.add_argument(
        "--keep-vision-bfloat16",
        action="store_true",
        help="Skip quantizing vision tower/projector modules if accuracy or backend compatibility requires it.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Additional llm-compressor ignore pattern. Can be passed multiple times.",
    )
    parser.add_argument(
        "--sequential-target",
        action="append",
        default=[],
        help="Optional transformer block class for sequential calibration. Can be passed multiple times.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Dtype used to load the merged checkpoint before quantization.",
    )
    parser.add_argument("--device-map", type=str, default="auto", help='Transformers device_map value, or "none".')
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code while loading.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Calibration sample shuffle seed.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    return parser


def _format_defect_messages(image: PILImage.Image | None = None) -> list[dict[str, Any]]:
    image_content: dict[str, Any] = {"type": "image"}
    if image is not None:
        image_content["image"] = image
    return [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": _defect_prompt_text()},
            ],
        },
    ]


def _defect_prompt_text() -> str:
    defect_options = ", ".join(DEFECT_TYPES)
    return (
        "You are a semiconductor optical inspection VLM. "
        "Classify exactly one visible wafer defect from the allowed classes. "
        "Inspect the wafer image and classify the injected defect. "
        f"The only allowed defect values are: {defect_options}. "
        "Return exactly one JSON object with this schema: "
        "{\"defect\": \"scratch|particle|bridge_short\"}. "
        "Do not include markdown, prose, coordinates, or extra keys. "
        "After the JSON object, immediately output STOP."
    )


def _resolve_dtype(dtype_name: str) -> torch.dtype | str:
    if torch is None:
        raise RuntimeError("Missing dependency: torch. Install the training/deployment requirements before quantizing.")
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def _load_quantization_dependencies() -> tuple[Any, type[Any], Any, type[Any], type[Any]]:
    if torch is None:
        raise RuntimeError("Missing dependency: torch. Install the training/deployment requirements before quantizing.")
    try:
        import transformers
        from datasets import Dataset
        from llmcompressor import oneshot
        from llmcompressor.modifiers.gptq import GPTQModifier
        from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
    except ImportError as error:  # pragma: no cover - depends on deployment environment
        raise RuntimeError(
            "Missing quantization dependency. Install llm-compressor and the training dependencies, "
            "for example: pip install llmcompressor"
        ) from error

    return transformers, Dataset, oneshot, GPTQModifier, SmoothQuantModifier


def _from_pretrained_kwargs(loader: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(loader.from_pretrained)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _load_model_and_processor(
    model_dir: Path,
    torch_dtype: torch.dtype | str,
    device_map: str,
    trust_remote_code: bool,
    transformers: Any,
) -> tuple[Any, Any]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Merged 16-bit model directory does not exist: {model_dir}")

    LOGGER.info("Loading processor from %s", model_dir)
    processor = transformers.AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    _ensure_padding_token(processor)

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if device_map.lower() != "none":
        load_kwargs["device_map"] = device_map

    load_kwargs["dtype"] = torch_dtype

    loader_names = ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM")
    load_errors: list[str] = []
    for loader_name in loader_names:
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        try:
            kwargs = _from_pretrained_kwargs(loader, load_kwargs)
            LOGGER.info("Loading model with transformers.%s", loader_name)
            model = loader.from_pretrained(str(model_dir), **kwargs)
            model.eval()
            return model, processor
        except Exception as error:  # pragma: no cover - loader support is environment-specific
            load_errors.append(f"{loader_name}: {error}")
            LOGGER.debug("Model loader %s failed", loader_name, exc_info=True)

    raise RuntimeError("Unable to load merged VLM checkpoint. Tried: " + " | ".join(load_errors))


def _ensure_padding_token(processor: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"


def _collect_calibration_pairs(dataset_dir: Path, sample_count: int, seed: int) -> list[tuple[Path, str]]:
    if sample_count <= 0:
        raise ValueError("--num-calibration-samples must be greater than zero.")

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing calibration images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing calibration labels directory: {labels_dir}")

    pairs: list[tuple[Path, str]] = []
    for label_path in sorted(labels_dir.glob("*.json")):
        image_path = _find_image_for_label(images_dir, label_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"Missing image for calibration label {label_path.name}")

        label = json.loads(label_path.read_text(encoding="utf-8"))
        defect = label.get("defect", "unknown")
        pairs.append((image_path, str(defect)))

    if len(pairs) < sample_count:
        raise ValueError(
            f"Requested {sample_count} calibration samples, but only found {len(pairs)} labeled images in {dataset_dir}."
        )

    random.Random(seed).shuffle(pairs)
    selected = pairs[:sample_count]
    LOGGER.info("Selected %d calibration samples from %s", len(selected), dataset_dir)
    return selected


def _find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        image_path = images_dir / f"{stem}{extension}"
        if image_path.is_file():
            return image_path
    return None


def _encode_calibration_sample(
    processor: Any,
    image: PILImage.Image,
    max_seq_length: int,
) -> Mapping[str, Any]:
    attempts: list[Callable[[], Any]] = [
        lambda: processor.apply_chat_template(
            _format_defect_messages(image),
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_generation_prompt=False,
            add_special_tokens=False,
        ),
        lambda: processor.apply_chat_template(
            _format_defect_messages(image),
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_generation_prompt=False,
        ),
        lambda: processor(
            text=processor.apply_chat_template(
                _format_defect_messages(),
                tokenize=False,
                add_generation_prompt=False,
            ),
            images=image,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_seq_length,
        ),
        lambda: processor(
            text=processor.apply_chat_template(
                _format_defect_messages(),
                tokenize=False,
                add_generation_prompt=False,
            ),
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_seq_length,
        ),
        lambda: processor(
            text=_defect_prompt_text(),
            images=image,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_seq_length,
        ),
        lambda: processor(
            text=_defect_prompt_text(),
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_seq_length,
        ),
    ]

    errors: list[str] = []
    for attempt in attempts:
        try:
            encoded = attempt()
            if isinstance(encoded, Mapping):
                return encoded
            if hasattr(encoded, "data") and isinstance(encoded.data, Mapping):
                return encoded.data
            raise TypeError(f"Processor returned unsupported type: {type(encoded).__name__}")
        except Exception as error:  # pragma: no cover - processor APIs vary by transformers release
            errors.append(str(error))

    raise RuntimeError("Failed to encode a multimodal calibration sample. Attempts failed with: " + " | ".join(errors))


def _build_calibration_dataset(
    processor: Any,
    dataset_dir: Path,
    sample_count: int,
    max_seq_length: int,
    seed: int,
    dataset_cls: type[Any],
) -> Any:
    samples: list[dict[str, Any]] = []
    for index, (image_path, defect) in enumerate(_collect_calibration_pairs(dataset_dir, sample_count, seed), start=1):
        LOGGER.debug("Encoding calibration sample %d/%d: %s (%s)", index, sample_count, image_path, defect)
        with PILImage.open(image_path) as image_file:
            image = image_file.convert("RGB")
            encoded = _encode_calibration_sample(processor, image, max_seq_length)

        sample = _serialize_processor_outputs(encoded)
        if not any(key in sample for key in MULTIMODAL_KEYS):
            raise RuntimeError(
                "Processor did not return image tensors for calibration. "
                "Refusing to run text-only PTQ for a vision deployment model."
            )
        samples.append(sample)

    return dataset_cls.from_list(samples)


def _serialize_processor_outputs(encoded: Mapping[str, Any]) -> dict[str, Any]:
    sample: dict[str, Any] = {}
    for key, value in encoded.items():
        if value is None or isinstance(value, str):
            continue
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
        else:
            try:
                tensor = torch.as_tensor(value)
            except (TypeError, ValueError):
                LOGGER.debug("Skipping non-tensor processor output %s=%r", key, type(value).__name__)
                continue

        if tensor.numel() == 0:
            continue
        sample[key] = tensor.tolist()

    if "input_ids" not in sample:
        raise RuntimeError("Processor output did not include input_ids.")
    return sample


def _data_collator(batch: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError("This PTQ script expects llm-compressor calibration batches of size 1.")

    collated: dict[str, torch.Tensor] = {}
    for key, value in batch[0].items():
        tensor = torch.as_tensor(value)
        if key in FLOAT_INPUT_KEYS or torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=torch.bfloat16)
        else:
            tensor = tensor.to(dtype=torch.long)
        collated[key] = tensor
    return collated


def _infer_sequential_targets(model: Any) -> list[str]:
    candidates = sorted(
        {
            type(module).__name__
            for module in model.modules()
            if type(module).__name__.endswith(("DecoderLayer", "TransformerLayer"))
            and "Vision" not in type(module).__name__
        }
    )
    if candidates:
        LOGGER.info("Inferred sequential calibration target: %s", candidates[0])
        return [candidates[0]]
    LOGGER.info("Could not infer a sequential target; llm-compressor will choose its default calibration flow.")
    return []


def _build_recipe(
    scheme: str,
    smoothquant_strength: float,
    ignore_patterns: list[str],
    sequential_targets: list[str],
    gptq_modifier_cls: type[Any],
    smoothquant_modifier_cls: type[Any],
) -> list[Any]:
    recipe: list[Any] = []
    if smoothquant_strength > 0:
        recipe.append(smoothquant_modifier_cls(smoothing_strength=smoothquant_strength))

    gptq_kwargs: dict[str, Any] = {
        "targets": "Linear",
        "scheme": scheme,
        "ignore": ignore_patterns,
    }
    signature = inspect.signature(gptq_modifier_cls)
    if sequential_targets and "sequential_targets" in signature.parameters:
        gptq_kwargs["sequential_targets"] = sequential_targets

    recipe.append(gptq_modifier_cls(**gptq_kwargs))
    return recipe


def _run_oneshot(
    model: Any,
    calibration_dataset: Any,
    recipe: list[Any],
    max_seq_length: int,
    sample_count: int,
    sequential_targets: list[str],
    oneshot: Callable[..., Any],
) -> None:
    oneshot_kwargs: dict[str, Any] = {
        "model": model,
        "dataset": calibration_dataset,
        "recipe": recipe,
        "max_seq_length": max_seq_length,
        "num_calibration_samples": sample_count,
        "data_collator": _data_collator,
    }

    signature = inspect.signature(oneshot)
    if sequential_targets and "sequential_targets" in signature.parameters:
        oneshot_kwargs["sequential_targets"] = sequential_targets

    LOGGER.info("Starting llm-compressor PTQ with %d multimodal calibration samples", sample_count)
    oneshot(**oneshot_kwargs)


def _save_quantized_model(model: Any, processor: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving vLLM-ready quantized checkpoint to %s", output_dir)
    try:
        model.save_pretrained(str(output_dir), save_compressed=True, safe_serialization=True)
    except TypeError:
        model.save_pretrained(str(output_dir), save_compressed=True)
    processor.save_pretrained(str(output_dir))
    _validate_saved_artifacts(output_dir)


def _validate_saved_artifacts(output_dir: Path) -> None:
    missing: list[str] = []
    for filename in ("config.json", "tokenizer_config.json"):
        if not (output_dir / filename).exists():
            missing.append(filename)
    if not any((output_dir / filename).exists() for filename in MODEL_WEIGHT_FILES):
        missing.append("model weights (.safetensors or .bin)")
    if not any((output_dir / filename).exists() for filename in TOKENIZER_FILES):
        missing.append("tokenizer model file")
    if not any((output_dir / filename).exists() for filename in PROCESSOR_FILES):
        missing.append("vision processor config")
    if missing:
        raise RuntimeError(f"Quantized export is incomplete. Missing artifacts: {', '.join(missing)}")


def quantize_vlm(args: argparse.Namespace) -> None:
    transformers, dataset_cls, oneshot, gptq_modifier_cls, smoothquant_modifier_cls = _load_quantization_dependencies()
    torch_dtype = _resolve_dtype(args.torch_dtype)

    model_dir = args.model_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    model, processor = _load_model_and_processor(
        model_dir=model_dir,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        transformers=transformers,
    )
    calibration_dataset = _build_calibration_dataset(
        processor=processor,
        dataset_dir=dataset_dir,
        sample_count=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        dataset_cls=dataset_cls,
    )

    ignore_patterns = ["re:.*lm_head", *args.ignore_pattern]
    if args.keep_vision_bfloat16:
        ignore_patterns.extend(VISION_IGNORE_PATTERNS)
        LOGGER.info("Keeping vision tower/projector modules in 16-bit.")
    else:
        LOGGER.info("Quantizing text and vision Linear modules; pass --keep-vision-bfloat16 to skip the vision path.")

    sequential_targets = args.sequential_target or _infer_sequential_targets(model)
    recipe = _build_recipe(
        scheme=args.scheme,
        smoothquant_strength=args.smoothquant_strength,
        ignore_patterns=ignore_patterns,
        sequential_targets=sequential_targets,
        gptq_modifier_cls=gptq_modifier_cls,
        smoothquant_modifier_cls=smoothquant_modifier_cls,
    )
    LOGGER.info("Quantization recipe: scheme=%s ignore=%s", args.scheme, ignore_patterns)

    _run_oneshot(
        model=model,
        calibration_dataset=calibration_dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        sample_count=args.num_calibration_samples,
        sequential_targets=sequential_targets,
        oneshot=oneshot,
    )
    _save_quantized_model(model, processor, output_dir)
    LOGGER.info("Quantization complete. Deployable model is ready at %s", output_dir)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    try:
        quantize_vlm(args)
    except Exception:
        LOGGER.exception("VLM quantization failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
