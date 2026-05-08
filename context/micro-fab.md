# Project Context: Micro-Fab VLM (Vision-Language Model Pipeline)

## 1. Project Objective

We are transitioning an existing text-based GRPO Reinforcement Learning pipeline (previously used for dynamic pricing) into a high-throughput **Vision-Language Model (VLM)** pipeline for semiconductor optical defect inspection.

The goal is to fine-tune and deploy a quantized model on a single 24GB GPU (RTX 3090) capable of ingesting microscopic wafer images and outputting strict JSON diagnostic reports at millisecond latencies.

## 2. The Tech Stack & Architecture

- **Base Model:** `unsloth/gemma-4-E4B-it` (Leveraging its native multimodal/image capabilities).
- **Training Engine:** Unsloth + `trl` (GRPO). We are replacing TorchRL continuous simulation with Ground-Truth Verification.
- **Serving Engine:** `vLLM` (using PagedAttention and Continuous Batching) wrapped in an asynchronous `FastAPI` gateway.
- **Data Processing:** OpenCV (`cv2.seamlessClone`) for synthetic defect generation, and `Pillow` for VLM image formatting.

## 3. The 1-Week Roadmap (Tasks for the Agent)

### Phase 1: Synthetic Vision Data Generation

- **Goal:** Create a script (`synthetic_defects.py`) to solve extreme class imbalance.
- **Mechanism:** Use OpenCV Poisson blending (`cv2.seamlessClone`) to mathematically blend transparent defect PNGs (scratches, particles) onto clean brushed-metal wafer backgrounds.
- **Output:** Generate a dataset of 500 labeled pairs (wafer image + ground truth JSON).

### Phase 2: Multimodal GRPO Fine-Tuning

- **Target File:** `train_grpo.py`
- **Modifications:** 1. Remove `torchrl` and `tensordict` dependencies.
  2. Update the data loader to load PIL Images.
  3. Modify the prompt formatter to use the Gemma `<image>` tag.
  4. Rewrite the `RewardBridge` to perform text-matching against ground-truth JSON (e.g., `+2.0` reward for matching the defect class, `-2.0` for invalid JSON) instead of calculating economic elasticity.

### Phase 3: Merging & Quantization

- **Target File:** `merge_lora_gemma4.py`
- **Modifications:** Run the existing script to fold the 16-bit adapters into the base model. Afterwards, prepare the model for Post-Training Quantization (PTQ) to W8A8 (FP8) or INT8 for deployment.

### Phase 4: High-Throughput Factory API

- **Status:** Implemented in `micro_fab_llm/serve_llm.py`; needs GPU smoke testing and latency validation.
- **Target File:** `serve_llm.py`
- **Modifications:** 1. Refactor the Pydantic schema to accept `image_base64` strings instead of numerical rider/driver states.
  2. Decode the base64 string into a PIL image.
  3. Pass both the `<image>` prompt and the PIL image into the `vLLM` engine using `vllm.Inputs(prompt, multi_modal_data={"image": pil_image})`.
  4. Extract and return the JSON defect classification.

## 3.1 Current Task Status

### Done in code

- **Phase 1:** Synthetic wafer defect generation and JSON label output.
- **Phase 2:** Multimodal GRPO training path with PIL images, Gemma `<image>` prompting, and ground-truth JSON reward matching.
- **Phase 3:** LoRA merge and deployment quantization preparation scripts.
- **Phase 4:** Async FastAPI `/inspect` endpoint that accepts base64 images, routes PIL images through vLLM multimodal input, and returns defect JSON.

### Still to complete or validate

- Produce or verify the final `micro_fab_vlm_deployed_4bit` checkpoint in the deployment environment.
- Run an end-to-end `/inspect` smoke test with the 4-bit VLM loaded by vLLM.
- Validate throughput and latency on the target 24GB RTX 3090.
- Add regression tests for strict JSON parsing, base64 decoding failures, and malformed model completions.
- Document deployment/runtime settings after GPU validation is complete.

## 4. Strict Constraints & Coding Guidelines

1. **24GB VRAM Limit:** The entire training and serving stack must fit on a single RTX 3090. Rely heavily on Unsloth's memory optimizations and vLLM's KV cache quantization.
2. **Asynchronous Architecture:** Do not use blocking synchronous calls in the FastAPI server. The physical KLA machinery must not stall while waiting for inference.
3. **No RL Environments:** Do not import or use `torchrl`. The objective is ground-truth perception, not sequential decision-making.
4. **Strict JSON:** The LLM output MUST be constrained to output valid JSON using the STOP token logic currently present in the codebase.
