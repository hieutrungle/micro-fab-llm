"""Async load tester for the Micro-Fab VLM /inspect endpoint.

Run this script while the API server is already running:

    python -m micro_fab_llm.serve_llm
    python tests/test_vlm_throughput.py

It sends 100 concurrent POST requests using 5 randomly selected synthetic
wafer image/label pairs and prints throughput plus classification accuracy.
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp


DATASET_IMAGES_DIR = Path("dataset/images")
DATASET_LABELS_DIR = Path("dataset/labels")
DEFAULT_URL = "http://127.0.0.1:8000/inspect"
IMAGE_SAMPLE_COUNT = 20
TOTAL_REQUESTS = 400
REQUEST_TIMEOUT_SECONDS = 600
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

_START_EVENT: asyncio.Event | None = None


@dataclass(frozen=True)
class RequestResult:
    latency_seconds: float
    status_code: int
    is_correct: bool


@dataclass(frozen=True)
class InspectionSample:
    payload: dict[str, str]
    ground_truth: str


def load_random_image_payloads(
    images_dir: Path = DATASET_IMAGES_DIR,
    labels_dir: Path = DATASET_LABELS_DIR,
    sample_count: int = IMAGE_SAMPLE_COUNT,
) -> list[InspectionSample]:
    """Read random image/label pairs and return API-ready inspection samples."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing synthetic image directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing synthetic label directory: {labels_dir}")

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and (labels_dir / f"{path.stem}.json").is_file()
    )
    if len(image_paths) < sample_count:
        raise ValueError(
            f"Need at least {sample_count} labeled images in {images_dir}, found {len(image_paths)}."
        )

    selected_paths = random.sample(image_paths, sample_count)
    samples: list[InspectionSample] = []
    for image_path in selected_paths:
        label_path = labels_dir / f"{image_path.stem}.json"
        label = json.loads(label_path.read_text(encoding="utf-8"))
        defect = label.get("defect")
        if not isinstance(defect, str) or not defect.strip():
            raise ValueError(f"Label file {label_path} does not contain a valid defect string.")

        encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        samples.append(
            InspectionSample(
                payload={"image_base64": encoded_image},
                ground_truth=defect.strip(),
            )
        )
    return samples


async def fire_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, str],
    ground_truth: str,
) -> RequestResult:
    """Send one inspection request and return latency, status, and correctness."""
    if _START_EVENT is not None:
        await _START_EVENT.wait()

    start_time = time.perf_counter()
    status_code = 0
    is_correct = False
    try:
        async with session.post(url, json=payload) as response:
            status_code = response.status
            try:
                response_payload = await response.json()
            except (aiohttp.ContentTypeError, json.JSONDecodeError):
                response_payload = {}

            predicted_defect = (
                response_payload.get("defect") if isinstance(response_payload, dict) else None
            )
            is_correct = (
                status_code == 200
                and isinstance(predicted_defect, str)
                and predicted_defect == ground_truth
            )
    except (aiohttp.ClientError, asyncio.TimeoutError):
        is_correct = False
    latency_seconds = time.perf_counter() - start_time
    return RequestResult(
        latency_seconds=latency_seconds,
        status_code=status_code,
        is_correct=is_correct,
    )


def print_report(results: list[RequestResult], total_execution_time: float) -> None:
    """Print a clean terminal throughput report."""
    total_requests = len(results)
    successful_requests = sum(result.status_code == 200 for result in results)
    correct_predictions = sum(
        result.is_correct for result in results if result.status_code == 200
    )
    average_latency = (
        sum(result.latency_seconds for result in results) / total_requests
        if total_requests
        else 0.0
    )
    requests_per_second = (
        total_requests / total_execution_time if total_execution_time > 0 else 0.0
    )
    success_rate = (
        (successful_requests / total_requests) * 100.0 if total_requests else 0.0
    )
    classification_accuracy = (
        (correct_predictions / successful_requests) * 100.0
        if successful_requests
        else 0.0
    )

    print("\nMicro-Fab VLM Throughput Report")
    print("=" * 34)
    print(f"Total Concurrent Requests: {total_requests}")
    print(f"Total Execution Time:      {total_execution_time:.3f} seconds")
    print(f"Requests Per Second (RPS): {requests_per_second:.3f}")
    print(f"Average Latency/Request:   {average_latency:.3f} seconds")
    print(f"Classification Accuracy:   {classification_accuracy:.1f}%")
    print(f"Success Rate:              {success_rate:.1f}%")


async def main() -> None:
    """Fire 100 concurrent /inspect requests and report throughput metrics."""
    global _START_EVENT

    samples = load_random_image_payloads()
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    connector = aiohttp.TCPConnector(limit=TOTAL_REQUESTS)
    _START_EVENT = asyncio.Event()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            asyncio.create_task(
                fire_request(
                    session,
                    DEFAULT_URL,
                    (sample := random.choice(samples)).payload,
                    sample.ground_truth,
                )
            )
            for _ in range(TOTAL_REQUESTS)
        ]

        # Let every task reach the start barrier before releasing the batch.
        await asyncio.sleep(0)
        start_time = time.perf_counter()
        _START_EVENT.set()
        results = await asyncio.gather(*tasks)
        total_execution_time = time.perf_counter() - start_time

    print_report(results, total_execution_time)


if __name__ == "__main__":
    asyncio.run(main())
