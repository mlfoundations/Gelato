#!/usr/bin/env python3
"""
Grounding evaluation script supporting Qwen3 and Qwen2.5 models.

This script evaluates vision-language models on unified grounding datasets.

Features:
- Model types: Qwen3 and Qwen2.5 vision-language models
- Distribution modes:
  - 'sharded': Single process with device_map='auto' (automatic tensor parallelism)
  - 'dp': Multi-process data parallelism across GPUs
- Prompt modes: 'gta1' (with resolution in system prompt) and 'gelato' (resolution-free)
- Coordinate formats: Pixel space or normalized (0-1000) space
- Automatic image resizing with smart_resize
- Greedy decoding for deterministic results
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor
)
import multiprocessing as mp
import torch.multiprocessing as torch_mp
from datetime import datetime
from PIL import Image

from qwen_vl_utils import process_vision_info, smart_resize


def parse_click_from_response(response: str) -> Optional[Tuple[float, float]]:
    """
    Parse click coordinates from model response using simple (x, y) format.

    Args:
        response: Model output string

    Returns:
        Tuple of (x, y) coordinates or None if not found
    """
    # Simple pattern to match (x, y) coordinates
    pattern = r'\((\d+),\s*(\d+)\)'

    match = re.search(pattern, response)
    if match:
        x_coord = int(match.group(1))
        y_coord = int(match.group(2))
        return (x_coord, y_coord)

    return None


def check_click_in_bbox(click_pos: Tuple[float, float], bbox: List[float]) -> bool:
    """Check if a click position is within a bounding box."""
    x, y = click_pos
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max


def check_click_in_polygon(click_pos: Tuple[float, float], polygon_xy: List[float]) -> bool:
    """
    Ray casting algorithm to determine if a point is inside a polygon.
    polygon_xy is a flat list [x1, y1, x2, y2, ..., xn, yn].
    """
    if not polygon_xy or len(polygon_xy) < 6:
        return False

    x, y = click_pos
    points: List[Tuple[float, float]] = [
        (float(polygon_xy[i]), float(polygon_xy[i + 1])) for i in range(0, len(polygon_xy), 2)
    ]

    inside = False
    j = len(points) - 1
    for i in range(len(points)):
        xi, yi = points[i]
        xj, yj = points[j]
        intersects_y = (yi > y) != (yj > y)
        if intersects_y:
            denom = (yj - yi) if (yj - yi) != 0 else 1e-12
            x_intersect = (xj - xi) * (y - yi) / denom + xi
            if x < x_intersect:
                inside = not inside
        j = i

    return inside


def check_click_in_region(
    click_pos: Tuple[float, float],
    bbox: List[float],
    polygon_xy: Optional[List[float]] = None
) -> bool:
    """Check if click is within a region described by polygon if present, else bbox."""
    if polygon_xy:
        return check_click_in_polygon(click_pos, polygon_xy)
    return check_click_in_bbox(click_pos, bbox)


def evaluate_batch(
    batch_annotations: List[Dict],
    images_dir: Path,
    model,
    processor,
    verbose: bool,
    pixel_space_output: bool,
    max_pixels: int,
    prompt_mode: str,
    image_factor: int,
    include_refusal: bool = False,
) -> List[Tuple[bool, Optional[str], Optional[Tuple[float, float]], str, str, Tuple[int, int], Tuple[int, int]]]:
    """
    Evaluate a batch of samples from the dataset.

    Returns:
        List of (success, error_message, predicted_click, prompt, response, original_dims, resized_dims) tuples
    """
    batch_size = len(batch_annotations)
    results = []

    # Prepare all images and prompts
    batch_images = []
    batch_prompts = []
    batch_img_paths = []
    batch_dimensions = []
    batch_processed_dimensions = []
    valid_indices = []

    for i, annotation in enumerate(batch_annotations):
        img_path = images_dir / annotation['image_path']

        if not img_path.exists():
            results.append((False, f"Image not found: {img_path}", None, "", "", (0, 0), (0, 0)))
            continue

        # Load original image
        pil_image = Image.open(img_path)
        original_width, original_height = pil_image.size

        # Always calculate and apply resize using smart_resize
        resized_height, resized_width = smart_resize(
            original_height,
            original_width,
            factor=image_factor,
            max_pixels=max_pixels
        )
        pil_image = pil_image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        batch_processed_dimensions.append((resized_width, resized_height))

        prompt = annotation['instruction']

        batch_images.append(pil_image)
        batch_prompts.append(prompt)
        batch_img_paths.append(img_path)
        batch_dimensions.append((original_width, original_height))
        valid_indices.append(i)

    if not batch_images:
        return results

    # Prepare messages for batch processing
    batch_messages = []
    batch_texts = []

    for i, (image, prompt) in enumerate(zip(batch_images, batch_prompts)):
        messages = []

        # Get the resized dimensions for this specific image
        width, height = batch_processed_dimensions[i]

        if prompt_mode == 'gta1':
            # GTA1 system prompt with resolution
            gta1_system_prompt = f'''You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)'''

            messages.append({
                "role": "system",
                "content": gta1_system_prompt
            })
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            })
        elif prompt_mode == 'gelato' or prompt_mode == 'gelato-refusal':
            # Gelato - GTA1-style prompt in user message
            gelato_prompt = "You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. For elements with area, return the center point.\n\nOutput the coordinate pair exactly:\n(x,y)"

            if prompt_mode == 'gelato-refusal':
                gelato_prompt += "\nIf you cannot find the element, please respond with 'refusal'."

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": gelato_prompt + "\n\n"},
                    {"type": "image", "image": image},
                    {"type": "text", "text": "\n" + prompt}
                ]
            })

        batch_messages.append(messages)
        base_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch_texts.append(base_text)

    batch_image_inputs, _ = process_vision_info(batch_messages, image_patch_size=image_factor // 2)

    # Prepare inputs for batch
    inputs = processor(
        text=batch_texts,
        images=batch_image_inputs if batch_image_inputs else None,
        padding=True,
        do_resize=False,
        padding_side="left",
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate responses for batch with greedy decoding
    with torch.no_grad():
        if verbose:
            print(f"Generating responses for batch of {batch_size} samples...")

        generation_kwargs = {
            'max_new_tokens': 32,
            'use_cache': True,
            'do_sample': False,
            'num_beams': 1,
        }

        generated_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    # Process results
    result_idx = 0
    for i, annotation in enumerate(batch_annotations):
        if i not in valid_indices:
            continue

        # Use the instruction text as the prompt
        prompt_text = annotation['instruction']
        response = output_texts[result_idx] if result_idx < len(output_texts) else ""

        if include_refusal:
            if "refusal" in response.lower():
                is_correct = annotation["box_type"] == "refusal"
                result_symbol = "✓" if is_correct else "✗"
                if verbose:
                    print(f"\nInstruction: {prompt_text}")
                    print(f"Response: {response}")
                    print(f"Result: {result_symbol}")
                original_dims = batch_dimensions[result_idx] if result_idx < len(batch_dimensions) else (0, 0)
                resized_dims = batch_processed_dimensions[result_idx] if result_idx < len(batch_processed_dimensions) else (0, 0)
                results.append((is_correct, None, None, prompt_text, response, original_dims, resized_dims))
                result_idx += 1
                continue


        # Parse click coordinates
        click_pos = parse_click_from_response(response)

        if click_pos is None:
            if verbose:
                print(f"\nInstruction: {prompt_text}")
                print(f"Response: {response}")
                print(f"Result: ✗ (No valid coordinates found)")
            original_dims = batch_dimensions[result_idx] if result_idx < len(batch_dimensions) else (0, 0)
            resized_dims = batch_processed_dimensions[result_idx] if result_idx < len(batch_processed_dimensions) else (0, 0)
            results.append((False, "No valid click action found in response", None, prompt_text, response, original_dims, resized_dims))
            result_idx += 1
            continue

        # Handle coordinate conversion
        original_width, original_height = batch_dimensions[result_idx]
        processed_width, processed_height = batch_processed_dimensions[result_idx]

        if pixel_space_output:
            # Model outputs pixel coordinates in resized image space
            x_pixel, y_pixel = click_pos

            # Cap pixel values to be within the image dimensions
            x_pixel = min(max(0, x_pixel), processed_width)
            y_pixel = min(max(0, y_pixel), processed_height)

            # Normalize to 0-1000 range
            x_norm = (x_pixel / processed_width) * 1000
            y_norm = (y_pixel / processed_height) * 1000
        else:
            # Coordinates are already in normalized 0-1000 space
            x_norm, y_norm = click_pos

        # Convert normalized coordinates to original image space
        click_x = (x_norm / 1000.0) * original_width
        click_y = (y_norm / 1000.0) * original_height
        scaled_click = (click_x, click_y)

        # Get ground truth region
        bbox_xyxy = annotation['bbox_xyxy']
        polygon_xy = annotation.get('polygon_xy') if isinstance(annotation, dict) else None

        # Check if click is within region
        is_correct = check_click_in_region(scaled_click, bbox_xyxy, polygon_xy)

        if verbose:
            result_symbol = "✓" if is_correct else "✗"
            print(f"\nInstruction: {prompt_text}")
            print(f"Response: {response}")
            if pixel_space_output:
                print(f"Predicted: ({int(click_pos[0])}, {int(click_pos[1])}) in {processed_width}×{processed_height}px → ({click_x:.1f}, {click_y:.1f}) in {original_width}×{original_height}px")
            else:
                print(f"Predicted: ({int(x_norm)}, {int(y_norm)}) normalized → ({click_x:.1f}, {click_y:.1f}) in {original_width}×{original_height}px")
            print(f"Ground truth bbox: {bbox_xyxy}")
            print(f"Result: {result_symbol}")

        results.append((is_correct, None, scaled_click, prompt_text, response,
                       (original_width, original_height),
                       (processed_width, processed_height)))
        result_idx += 1

    return results


def gpu_worker(
    gpu_id: int,
    annotations_subset: List[Dict],
    model_path: str,
    model_type: str,
    images_dir: Path,
    batch_size: int,
    verbose: bool,
    max_pixels: int,
    pixel_space_output: bool,
    prompt_mode: str,
    processor_path: str,
    return_dict: Dict,
    image_factor: int,
    include_refusal: bool = False,
):
    """Worker function that runs on a single GPU for data parallel mode."""
    try:
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"

        print(f"GPU {gpu_id}: Loading {model_type} model on {device}")

        # Load appropriate model based on model type
        if model_type == 'qwen3':
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16
            )
        else:  # qwen2.5
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16
            )
        model = model.to(device)

        processor = AutoProcessor.from_pretrained(processor_path, max_pixels=max_pixels)
        model.eval()

        results = []
        errors = []
        correct_count = 0
        total_count = 0

        print(f"GPU {gpu_id}: Processing {len(annotations_subset)} samples in batches of {batch_size}")

        total_batches = (len(annotations_subset) + batch_size - 1) // batch_size
        pbar = tqdm(
            total=total_batches,
            desc=f"GPU {gpu_id}",
            position=gpu_id,
            leave=True,
            ncols=120
        )

        for i in range(0, len(annotations_subset), batch_size):
            batch_annotations = annotations_subset[i:i+batch_size]

            batch_results = evaluate_batch(
                batch_annotations,
                images_dir,
                model,
                processor,
                verbose,
                pixel_space_output,
                max_pixels,
                prompt_mode,
                image_factor,
                include_refusal,
            )

            for j, (annotation, (is_correct, error_msg, predicted_click, prompt, response, original_dims, resized_dims)) in enumerate(zip(batch_annotations, batch_results)):
                if error_msg:
                    errors.append({
                        'annotation': annotation,
                        'error': error_msg
                    })

                results.append({
                    'annotation': annotation,
                    'is_correct': is_correct,
                    'error_msg': error_msg,
                    'predicted_click': predicted_click,
                    'response': response,
                    'original_dims': original_dims,
                    'resized_dims': resized_dims
                })

                total_count += 1
                if is_correct:
                    correct_count += 1

            pbar.update(1)
            if total_count > 0:
                pbar.set_postfix({'samples': total_count})

        pbar.close()

        return_dict[gpu_id] = {
            'results': results,
            'errors': errors
        }

        print(f"GPU {gpu_id}: Completed ({correct_count}/{total_count} samples)")

    except Exception as e:
        print(f"GPU {gpu_id}: Fatal error: {str(e)}")
        return_dict[gpu_id] = {
            'results': [],
            'errors': [{'error': f"GPU {gpu_id} fatal error: {str(e)}"}]
        }


def update_stats(stat_bucket: Dict[str, Dict[str, int]], key: str, is_correct: bool) -> None:
    """Helper to update statistics buckets."""
    entry = stat_bucket.setdefault(key, {"correct": 0, "total": 0})
    entry["total"] += 1
    if is_correct:
        entry["correct"] += 1


def evaluate_single_process_sharded(
    annotations: List[Dict[str, Any]],
    images_dir: Path,
    model_path: str,
    model_type: str,
    processor_path: str,
    batch_size: int,
    pixel_space_output: bool,
    max_pixels: int,
    prompt_mode: str,
    verbose: bool,
    image_factor: int,
    include_refusal: bool = False,
) -> Dict[str, Any]:
    """Evaluate using a single process with device_map='auto' for sharding across GPUs."""
    processor_kwargs: Dict[str, Any] = {"max_pixels": max_pixels}
    processor = AutoProcessor.from_pretrained(processor_path, **processor_kwargs)

    model_kwargs: Dict[str, Any] = {"device_map": "auto", "torch_dtype": torch.bfloat16}

    print(f"\nLoading {model_type} model '{model_path}' with device_map='auto' (sharded mode)")
    print(f"Model kwargs: {model_kwargs}")

    # Load appropriate model based on model type
    if model_type == 'qwen3':
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    else:  # qwen2.5
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    model.eval()

    total_batches = math.ceil(len(annotations) / batch_size)
    progress = tqdm(total=total_batches, desc=f"{model_type.upper()} (sharded)")

    correct = 0
    total = 0
    errors: List[Dict[str, Any]] = []
    sample_results: List[Dict[str, Any]] = []
    results_by_file: Dict[str, Any] = {}
    results_by_application: Dict[str, Dict[str, int]] = {}
    results_by_ui_type: Dict[str, Dict[str, int]] = {}
    results_by_data_source: Dict[str, Dict[str, int]] = {}
    results_by_dataset_type: Dict[str, Any] = {}

    with torch.inference_mode():
        for start in range(0, len(annotations), batch_size):
            end = start + batch_size
            batch_annotations = annotations[start:end]

            batch_results = evaluate_batch(
                batch_annotations=batch_annotations,
                images_dir=images_dir,
                model=model,
                processor=processor,
                verbose=verbose,
                pixel_space_output=pixel_space_output,
                max_pixels=max_pixels,
                prompt_mode=prompt_mode,
                image_factor=image_factor,
                include_refusal=include_refusal,
            )

            for annotation, result in zip(batch_annotations, batch_results):
                (
                    is_correct,
                    error_msg,
                    predicted_click,
                    prompt,
                    response,
                    original_dims,
                    resized_dims,
                ) = result

                if error_msg:
                    errors.append({"annotation": annotation, "error": error_msg})

                sample_results.append(
                    {
                        "img_filename": annotation.get("image_path", annotation.get("img_filename", "")),
                        "instruction": annotation.get("instruction", ""),
                        "response": response,
                        "is_correct": bool(is_correct),
                        "predicted_click": predicted_click,
                        "error_msg": error_msg,
                        "bbox": annotation.get("bbox_xyxy", annotation.get("bbox", [])),
                        "original_resolution": original_dims,
                        "resized_resolution": resized_dims,
                    }
                )

                total += 1
                if is_correct:
                    correct += 1

                ui_type = annotation.get("metadata", {}).get("ui_type", "unknown")
                application = annotation.get("metadata", {}).get("application", "unknown")
                data_source = annotation.get("metadata", {}).get("data_source", "unknown")
                dataset_type = annotation.get("dataset_type", "unknown")
                source_file = annotation.get("source_file", "unknown")

                file_entry = results_by_file.setdefault(source_file, {"correct": 0, "total": 0, "by_ui_type": {}})
                file_entry["total"] += 1
                if is_correct:
                    file_entry["correct"] += 1
                update_stats(file_entry["by_ui_type"], ui_type, bool(is_correct))

                update_stats(results_by_application, application, bool(is_correct))
                update_stats(results_by_ui_type, ui_type, bool(is_correct))
                update_stats(results_by_data_source, data_source, bool(is_correct))

                dset_entry = results_by_dataset_type.setdefault(
                    dataset_type,
                    {"correct": 0, "total": 0, "by_ui_type": {}},
                )
                dset_entry["total"] += 1
                if is_correct:
                    dset_entry["correct"] += 1
                update_stats(dset_entry["by_ui_type"], ui_type, bool(is_correct))

            progress.update(1)

    progress.close()
    accuracy = correct / total if total else 0.0

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Overall accuracy: {accuracy:.2%}")

    if results_by_dataset_type:
        print("\nAccuracy by dataset and UI type:")
        for dset, stats in sorted(results_by_dataset_type.items()):
            dset_acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
            print(f"  {dset}: {dset_acc:.2%} ({stats['correct']}/{stats['total']})")
            for ui, ui_stats in sorted(stats.get("by_ui_type", {}).items()):
                ui_acc = ui_stats["correct"] / ui_stats["total"] if ui_stats["total"] else 0.0
                print(f"    - {ui}: {ui_acc:.2%} ({ui_stats['correct']}/{ui_stats['total']})")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": len(errors),
        "results_by_dataset_type": results_by_dataset_type,
        "results_by_file": results_by_file,
        "results_by_application": results_by_application,
        "results_by_ui_type": results_by_ui_type,
        "results_by_data_source": results_by_data_source,
        "sample_results": sample_results,
    }


def evaluate_grounding(
    unified_json: Path,
    images_dir: Path,
    model_path: str,
    model_type: str,
    distributed_mode: str,
    processor_path: Optional[str],
    max_samples: Optional[int],
    verbose: bool,
    batch_size: int,
    num_gpus: Optional[int],
    max_pixels: int,
    pixel_space_output: bool,
    prompt_mode: str,
    include_refusal: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a vision-language model on grounding tasks.

    Args:
        unified_json: Path to the unified JSON annotation file
        images_dir: Path to the directory containing images
        model_path: Path to the model checkpoint
        model_type: Model architecture type ('qwen3' or 'qwen2.5')
        distributed_mode: Distribution strategy - 'sharded' (device_map='auto', single process)
                         or 'dp' (data parallel, multi-process)
        processor_path: Path to the processor (defaults to model_path)
        max_samples: Maximum number of samples to evaluate (None for all)
        verbose: Print detailed evaluation logs
        batch_size: Batch size for inference
        num_gpus: Number of GPUs to use (only for 'dp' mode, None uses all available)
        max_pixels: Maximum number of pixels for image resizing
        pixel_space_output: Whether model outputs pixel coordinates (vs normalized 0-1000)
        prompt_mode: Prompt format - 'gta1' (with resolution) or 'gelato' (resolution-free)
        include_refusal: Whether to include refusal samples in the evaluation (default: False)

    Returns:
        Dictionary containing evaluation results including accuracy, sample-level results,
        and breakdowns by dataset type, UI type, application, etc.
    """
    # Set image_factor based on model type
    image_factor = 32 if model_type == 'qwen3' else 28

    print(f"Model path: {model_path}")
    print(f"Model type: {model_type}")
    print(f"Distributed mode: {distributed_mode}")
    print(f"Image factor: {image_factor}")
    print(f"Unified JSON: {unified_json}")
    print(f"Images directory: {images_dir}")

    # Load unified dataset
    if not unified_json.exists():
        print(f"Error: Unified JSON file not found: {unified_json}")
        return {}

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return {}

    print(f"\nLoading unified dataset from: {unified_json}")
    with open(unified_json, 'r') as f:
        all_annotations = json.load(f)

    if not include_refusal:
        # Filter out refusal samples
        all_annotations = [anno for anno in all_annotations if anno["box_type"] != "refusal"]

    print(f"Total annotations: {len(all_annotations)}")

    # Validate annotations
    if not all_annotations:
        print("Error: No annotations found in dataset")
        return {}

    if max_samples and max_samples < len(all_annotations):
        all_annotations = all_annotations[:max_samples]
        print(f"Limiting to {max_samples} samples")

    # Set processor path
    if processor_path is None:
        processor_path = model_path

    # Route to appropriate evaluator based on distributed mode
    if distributed_mode == 'sharded':
        return evaluate_single_process_sharded(
            annotations=all_annotations,
            images_dir=images_dir,
            model_path=model_path,
            model_type=model_type,
            processor_path=processor_path,
            batch_size=batch_size,
            pixel_space_output=pixel_space_output,
            max_pixels=max_pixels,
            prompt_mode=prompt_mode,
            verbose=verbose,
            image_factor=image_factor,
            include_refusal=include_refusal,
        )
    else:  # 'dp' - data parallel mode
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, will use CPU mode via single process")
            num_gpus = 1
        else:
            available_gpus = torch.cuda.device_count()
            if num_gpus is None:
                num_gpus = available_gpus
            else:
                num_gpus = min(num_gpus, available_gpus)
            print(f"Using {num_gpus} GPU(s) out of {available_gpus} available")

        # Split annotations across GPUs
        annotations_per_gpu = len(all_annotations) // num_gpus
        remainder = len(all_annotations) % num_gpus

        gpu_annotation_subsets = []
        start_idx = 0
        for i in range(num_gpus):
            end_idx = start_idx + annotations_per_gpu + (1 if i < remainder else 0)
            gpu_annotation_subsets.append(all_annotations[start_idx:end_idx])
            start_idx = end_idx

        # Set up multiprocessing
        if num_gpus > 1:
            torch_mp.set_start_method('spawn', force=True)
        manager = mp.Manager()
        return_dict = manager.dict()

        # Launch processes
        processes = []
        for gpu_id in range(num_gpus):
            if num_gpus == 1:
                gpu_worker(
                    gpu_id,
                    gpu_annotation_subsets[gpu_id],
                    model_path,
                    model_type,
                    images_dir,
                    batch_size,
                    verbose,
                    max_pixels,
                    pixel_space_output,
                    prompt_mode,
                    processor_path,
                    return_dict,
                    image_factor,
                    include_refusal,
                )
            else:
                p = torch_mp.Process(
                    target=gpu_worker,
                    args=(
                        gpu_id,
                        gpu_annotation_subsets[gpu_id],
                        model_path,
                        model_type,
                        images_dir,
                        batch_size,
                        verbose,
                        max_pixels,
                        pixel_space_output,
                        prompt_mode,
                        processor_path,
                        return_dict,
                        image_factor,
                        include_refusal,
                    )
                )
                p.start()
                processes.append(p)

        if processes:
            print("\nWaiting for all GPUs to complete...")
            for p in processes:
                p.join()

        print("\nAll GPUs completed. Collecting results...")

        # Collect results
        all_results = []
        errors = []
        sample_results = []

        for gpu_id in range(num_gpus):
            if gpu_id in return_dict:
                gpu_data = return_dict[gpu_id]
                all_results.extend(gpu_data['results'])
                errors.extend(gpu_data['errors'])

        # Process results
        correct = 0
        total = 0
        results_by_file = {}
        results_by_application = {}
        results_by_ui_type = {}
        results_by_data_source = {}
        results_by_dataset_type = {}

        for result in all_results:
            annotation = result['annotation']
            is_correct = result['is_correct']

            sample_results.append({
                'img_filename': annotation.get('image_path', annotation.get('img_filename', '')),
                'instruction': annotation['instruction'],
                'response': result.get('response', ''),
                'is_correct': is_correct,
                'predicted_click': result.get('predicted_click'),
                'error_msg': result.get('error_msg'),
                'bbox': annotation.get('bbox_xyxy', annotation.get('bbox', [])),
                'original_resolution': result.get('original_dims', (0, 0)),
                'resized_resolution': result.get('resized_dims', (0, 0))
            })

            if is_correct:
                correct += 1
            total += 1

            source_file = annotation['source_file']
            ui_type = annotation.get('metadata', {}).get('ui_type', 'unknown')
            application = annotation.get('metadata', {}).get('application', 'unknown')
            data_source = annotation.get('metadata', {}).get('data_source', 'unknown')
            dataset_type = annotation.get('dataset_type', 'unknown')

            # Update statistics
            if source_file not in results_by_file:
                results_by_file[source_file] = {'correct': 0, 'total': 0, 'by_ui_type': {}}
            results_by_file[source_file]['total'] += 1
            if is_correct:
                results_by_file[source_file]['correct'] += 1

            if ui_type not in results_by_file[source_file]['by_ui_type']:
                results_by_file[source_file]['by_ui_type'][ui_type] = {'correct': 0, 'total': 0}
            results_by_file[source_file]['by_ui_type'][ui_type]['total'] += 1
            if is_correct:
                results_by_file[source_file]['by_ui_type'][ui_type]['correct'] += 1

            update_stats(results_by_application, application, is_correct)
            update_stats(results_by_ui_type, ui_type, is_correct)
            update_stats(results_by_data_source, data_source, is_correct)

            if dataset_type not in results_by_dataset_type:
                results_by_dataset_type[dataset_type] = {'correct': 0, 'total': 0, 'by_ui_type': {}}
            results_by_dataset_type[dataset_type]['total'] += 1
            if is_correct:
                results_by_dataset_type[dataset_type]['correct'] += 1

            if ui_type not in results_by_dataset_type[dataset_type]['by_ui_type']:
                results_by_dataset_type[dataset_type]['by_ui_type'][ui_type] = {'correct': 0, 'total': 0}
            results_by_dataset_type[dataset_type]['by_ui_type'][ui_type]['total'] += 1
            if is_correct:
                results_by_dataset_type[dataset_type]['by_ui_type'][ui_type]['correct'] += 1

        accuracy = correct / total if total > 0 else 0

        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Total samples: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Overall accuracy: {accuracy:.2%}")

        if results_by_dataset_type:
            print("\nAccuracy by dataset and UI type:")
            for dset, stats in sorted(results_by_dataset_type.items()):
                dset_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                print(f"  {dset}: {dset_acc:.2%} ({stats['correct']}/{stats['total']})")
                if stats.get('by_ui_type'):
                    for ui_type, ui_stats in sorted(stats['by_ui_type'].items()):
                        ui_acc = ui_stats['correct'] / ui_stats['total'] if ui_stats['total'] > 0 else 0
                        print(f"    - {ui_type}: {ui_acc:.2%} ({ui_stats['correct']}/{ui_stats['total']})")

        if errors:
            print(f"\nTotal errors: {len(errors)}")

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'errors': len(errors),
            'results_by_dataset_type': results_by_dataset_type,
            'results_by_file': results_by_file,
            'results_by_application': results_by_application,
            'results_by_ui_type': results_by_ui_type,
            'results_by_data_source': results_by_data_source,
            'sample_results': sample_results
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 and Qwen2.5 models on grounding")

    parser.add_argument('--json-file', '-j', type=str, required=True, help='Path to unified JSON file')
    parser.add_argument('--images-dir', '-i', type=str, required=True, help='Path to images directory')
    parser.add_argument('--model', '-m', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, required=True, choices=['qwen3', 'qwen2.5'], help='Model type')
    parser.add_argument('--processor', type=str, help='Path to processor (defaults to model path)')
    parser.add_argument('--max-samples', '-n', type=int, help='Maximum number of samples')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--batch-size', '-bs', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--max-pixels', '-mpx', type=int, default=4000000, help='Max pixels (default: 4,000,000)')
    parser.add_argument('--pixel-space-output', action='store_true', help='Model outputs pixel coordinates')
    parser.add_argument('--prompt-mode', type=str, choices=['gta1', 'gelato', 'gelato-refusal'], default='gta1', help='Prompt mode')
    parser.add_argument('--include-refusal', action='store_true', help='Include refusal samples')
    parser.add_argument('--distributed-mode', '-dm', type=str, choices=['sharded', 'dp'], default='sharded',
                        help='Distributed mode: "sharded" (device_map=auto, single process) or "dp" (data parallel, multi-process)')
    parser.add_argument('--num-gpus', '-ng', type=int, help='Number of GPUs (only for dp mode)')
    parser.add_argument('--output-dir', '-o', type=str, default='.', help='Output directory')

    args = parser.parse_args()

    json_file = Path(args.json_file)
    images_dir = Path(args.images_dir)

    if not json_file.exists():
        print(f"Error: JSON file not found: {json_file}")
        return

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return

    # Run evaluation
    results = evaluate_grounding(
        unified_json=json_file,
        images_dir=images_dir,
        model_path=args.model,
        model_type=args.model_type,
        distributed_mode=args.distributed_mode,
        processor_path=args.processor,
        max_samples=args.max_samples,
        verbose=args.verbose,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        max_pixels=args.max_pixels,
        pixel_space_output=args.pixel_space_output,
        prompt_mode=args.prompt_mode,
        include_refusal=args.include_refusal,
    )

    if results:
        # Generate output filename
        model_name = Path(args.model).name or Path(args.model).parent.name
        dataset_name = json_file.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_filename = f"grounding_eval_{dataset_name}_{model_name}_{timestamp}.json"
        output_filename = re.sub(r'[<>:"/\\|?*]', '_', output_filename)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / output_filename

        # Add metadata
        image_factor = 32 if args.model_type == 'qwen3' else 28
        results['metadata'] = {
            'model_path': args.model,
            'model_type': args.model_type,
            'distributed_mode': args.distributed_mode,
            'dataset': dataset_name,
            'timestamp': timestamp,
            'prompt_mode': args.prompt_mode,
            'pixel_space_output': args.pixel_space_output,
            'batch_size': args.batch_size,
            'max_pixels': args.max_pixels,
            'image_factor': image_factor,
            'num_gpus': args.num_gpus if args.distributed_mode == 'dp' else None,
        }

        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        print(f"\nKey metrics:")
        print(f"  Dataset: {dataset_name}")
        print(f"  Model: {model_name}")
        print(f"  Accuracy: {results.get('accuracy', 0):.2%}")
        print(f"  Total samples: {results.get('total', 0)}")


if __name__ == "__main__":
    main()