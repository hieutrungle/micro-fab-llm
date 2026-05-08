"""Async FastAPI server for multimodal wafer defect inspection via vLLM."""

from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import io
import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from transformers import AutoTokenizer, CONFIG_MAPPING, PreTrainedTokenizerBase
import vllm
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


LOGGER = logging.getLogger(__name__)

ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]


@dataclass(frozen=True)
class ServerConfig:
    model_path: str = "micro_fab_vlm_deployed_4bit"
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    dtype: ModelDType = "bfloat16"
    quantization: str = "bitsandbytes"
    load_format: str = "bitsandbytes"
    gpu_memory_utilization: float = 0.80
    max_model_len: int = 4096
    enforce_eager: bool = True
    trust_remote_code: bool = False
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0


CONFIG = ServerConfig()

_CODE_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)

_GLOBAL_ENGINE: AsyncLLMEngine | None = None
_GLOBAL_TOKENIZER: PreTrainedTokenizerBase | None = None


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


def init_engine() -> AsyncLLMEngine:
    """Initialize the async vLLM engine for the quantized VLM checkpoint."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is not None:
        return _GLOBAL_ENGINE

    _ensure_transformers_supports_model_type(CONFIG.model_path)
    LOGGER.info("Initializing async vLLM engine from %s", CONFIG.model_path)
    engine_args = AsyncEngineArgs(
        model=CONFIG.model_path,
        quantization=CONFIG.quantization,
        load_format=CONFIG.load_format,
        tensor_parallel_size=CONFIG.tensor_parallel_size,
        dtype=CONFIG.dtype,
        gpu_memory_utilization=CONFIG.gpu_memory_utilization,
        max_model_len=CONFIG.max_model_len,
        enforce_eager=CONFIG.enforce_eager,
        trust_remote_code=CONFIG.trust_remote_code,
    )
    _GLOBAL_ENGINE = AsyncLLMEngine.from_engine_args(engine_args)
    return _GLOBAL_ENGINE


def _get_engine() -> AsyncLLMEngine:
    return init_engine()


def _get_prompt_tokenizer() -> PreTrainedTokenizerBase:
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is not None:
        return _GLOBAL_TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG.model_path,
        trust_remote_code=CONFIG.trust_remote_code,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("Loaded tokenizer is not a PreTrainedTokenizerBase instance")
    _GLOBAL_TOKENIZER = tokenizer
    return _GLOBAL_TOKENIZER


class InspectionRequest(BaseModel):
    image_base64: str


class InspectionResponse(BaseModel):
    defect: str
    confidence: float | None = None


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Initialize vLLM after process bootstrap is complete."""
    try:
        init_engine()
        _get_prompt_tokenizer()
    except Exception:
        LOGGER.exception("vLLM initialization failed during FastAPI lifespan startup")
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
    user_prompt = "<image>\nClassify the defect in this wafer."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tokenizer = _get_prompt_tokenizer()
    if hasattr(tokenizer, "apply_chat_template"):
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            formatted = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False)

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


def _build_vllm_input(formatted_text: str, pil_image: Image.Image) -> Any:
    inputs_cls = getattr(vllm, "Inputs", None)
    if inputs_cls is not None:
        return inputs_cls(prompt=formatted_text, multi_modal_data={"image": pil_image})
    return {"prompt": formatted_text, "multi_modal_data": {"image": pil_image}}


async def _await_final_output(generation: Any) -> Any:
    if inspect.isawaitable(generation):
        return await generation

    if isinstance(generation, AsyncIterator) or hasattr(generation, "__aiter__"):
        final_output = None
        async for output in generation:
            final_output = output
        return final_output

    return generation


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


@app.post("/inspect", response_model=InspectionResponse)
async def inspect_wafer(request: InspectionRequest) -> InspectionResponse:
    try:
        pil_image = await asyncio.to_thread(_decode_base64_image, request.image_base64)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    formatted_text = format_inspection_prompt()
    vllm_input = _build_vllm_input(formatted_text, pil_image)
    sampling_params = SamplingParams(
        max_tokens=CONFIG.max_tokens,
        temperature=CONFIG.temperature,
        top_p=CONFIG.top_p,
        stop=["STOP"],
    )
    request_id = f"inspect-{uuid4()}"

    text = ""
    try:
        results = await _await_final_output(
            _get_engine().generate(vllm_input, sampling_params, request_id)
        )
        if results is None or not getattr(results, "outputs", None):
            raise HTTPException(status_code=502, detail="vLLM returned an empty completion")
        text = results.outputs[0].text
        return _parse_inspection_response(text)
    except HTTPException:
        raise
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
