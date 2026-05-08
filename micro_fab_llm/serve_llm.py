"""Async FastAPI server for multimodal wafer defect inspection."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, CONFIG_MAPPING


LOGGER = logging.getLogger(__name__)

ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]


@dataclass(frozen=True)
class ServerConfig:
    model_path: str = "micro_fab_vlm_merged_16bit"
    host: str = "0.0.0.0"
    port: int = 8000
    dtype: ModelDType = "bfloat16"
    max_model_len: int = 4096
    trust_remote_code: bool = False
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0


CONFIG = ServerConfig()

_CODE_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)

_GLOBAL_MODEL: Any | None = None
_GLOBAL_PROCESSOR: Any | None = None


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


_setup_logging()


def _read_model_type_from_config(model_path: str) -> str | None:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.warning("Failed to parse model config at %s", config_path)
        return None

    model_type = payload.get("model_type")
    return str(model_type) if isinstance(model_type, str) else None


def _ensure_transformers_supports_model_type(model_path: str) -> None:
    model_type = _read_model_type_from_config(model_path)
    if model_type is None or model_type in CONFIG_MAPPING:
        return

    raise RuntimeError(
        "Incompatible transformers version for this checkpoint. "
        f"model_type={model_type!r} is not registered in transformers CONFIG_MAPPING. "
        "Upgrade transformers in this environment, for example: "
        "pip install --upgrade transformers"
    )


def _resolve_torch_dtype(dtype_name: ModelDType) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if dtype_name in {"half", "float16"}:
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _get_device_map() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_engine() -> Any:
    """Initialize the native Transformers Gemma-4 VLM."""
    global _GLOBAL_MODEL, _GLOBAL_PROCESSOR
    if _GLOBAL_MODEL is not None:
        return _GLOBAL_MODEL

    _ensure_transformers_supports_model_type(CONFIG.model_path)
    LOGGER.info("Initializing Transformers VLM from %s", CONFIG.model_path)
    _GLOBAL_PROCESSOR = AutoProcessor.from_pretrained(
        CONFIG.model_path,
        trust_remote_code=CONFIG.trust_remote_code,
    )
    _GLOBAL_MODEL = AutoModelForImageTextToText.from_pretrained(
        CONFIG.model_path,
        dtype=_resolve_torch_dtype(CONFIG.dtype),
        device_map=_get_device_map(),
        low_cpu_mem_usage=True,
        trust_remote_code=CONFIG.trust_remote_code,
    )
    _GLOBAL_MODEL.eval()
    return _GLOBAL_MODEL


def _get_engine() -> Any:
    return init_engine()


def _get_processor() -> Any:
    init_engine()
    if _GLOBAL_PROCESSOR is None:
        raise RuntimeError("Processor was not initialized")
    return _GLOBAL_PROCESSOR


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(_get_device_map())


class InspectionRequest(BaseModel):
    image_base64: str


class InspectionResponse(BaseModel):
    defect: str
    confidence: float | None = None


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Initialize the model after process bootstrap is complete."""
    try:
        init_engine()
    except Exception:
        LOGGER.exception("Model initialization failed during FastAPI lifespan startup")
        raise
    yield


app = FastAPI(title="Micro-Fab VLM Inspection API", lifespan=_lifespan)


def format_inspection_prompt() -> str:
    """Format the strict JSON multimodal inspection prompt."""
    system_prompt = (
        "You are a semiconductor wafer optical inspection VLM. "
        "Return ONLY a valid JSON object with schema "
        "{\"defect\": string, \"confidence\": number | null}. "
        "The defect value must be the best defect class visible in the image. "
        "Do not output markdown, prose, comments, or extra keys. "
        "When you are finished, immediately output the word 'STOP'."
    )
    user_prompt = "Classify the defect in this wafer."
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    processor = _get_processor()
    if hasattr(processor, "apply_chat_template"):
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            formatted = processor.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            formatted = processor.apply_chat_template(messages, tokenize=False)

        if isinstance(formatted, str):
            return formatted
        return str(formatted)

    return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"


def _decode_base64_image(image_base64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("image_base64 is not valid base64") from error

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as error:
        raise ValueError("image_base64 does not decode to a supported image") from error


def _strip_markdown_json_fence(text: str) -> str:
    cleaned = text.split("STOP", 1)[0].strip()
    match = _CODE_FENCE_PATTERN.match(cleaned)
    if match is not None:
        return match.group("body").strip()
    return cleaned


def _extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract a JSON object after removing optional markdown code fences."""
    cleaned = _strip_markdown_json_fence(text)
    decoder = json.JSONDecoder()

    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise ValueError("Could not parse a JSON object from model output")


def _move_processor_inputs_to_device(inputs: Any, device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def _parse_inspection_response(text: str) -> InspectionResponse:
    payload = _extract_json_from_text(text)
    defect = payload.get("defect")
    if not isinstance(defect, str) or not defect.strip():
        raise ValueError("Model JSON did not include a non-empty 'defect' string")

    confidence: float | None = None
    raw_confidence = payload.get("confidence")
    if raw_confidence is not None:
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = None

    return InspectionResponse(defect=defect.strip(), confidence=confidence)


def _generate_inspection(pil_image: Image.Image) -> str:
    model = _get_engine()
    processor = _get_processor()
    formatted_text = format_inspection_prompt()
    inputs = processor(
        text=formatted_text,
        images=pil_image,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=CONFIG.max_model_len,
    )
    inputs = _move_processor_inputs_to_device(inputs, _model_device(model))

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": CONFIG.max_tokens,
        "do_sample": CONFIG.temperature > 0,
    }
    if CONFIG.temperature > 0:
        generation_kwargs["temperature"] = CONFIG.temperature
        generation_kwargs["top_p"] = CONFIG.top_p

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0, prompt_length:]
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


@app.post("/inspect", response_model=InspectionResponse)
async def inspect_wafer(request: InspectionRequest) -> InspectionResponse:
    try:
        pil_image = await asyncio.to_thread(_decode_base64_image, request.image_base64)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    text = ""
    try:
        text = await asyncio.to_thread(_generate_inspection, pil_image)
        return _parse_inspection_response(text)
    except HTTPException:
        raise
    except RuntimeError as error:
        LOGGER.exception("Inspection request failed")
        raise HTTPException(status_code=500, detail="Inspection failed") from error
    except ValueError as error:
        LOGGER.warning(
            "Failed to parse completion into inspection response: %s | raw=%r",
            error,
            text[:500] if isinstance(text, str) else text,
        )
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("Inspection request failed")
        raise HTTPException(status_code=500, detail="Inspection failed") from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=CONFIG.host,
        port=CONFIG.port,
        reload=False,
    )
