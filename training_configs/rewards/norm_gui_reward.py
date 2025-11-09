import json
import math
import re
from typing import Any, Optional, Sequence

PREDICTION_SCALE = 1000.0  # model outputs 0–1000 per coordinate


def format_reward(response: Any) -> float:
    if not isinstance(response, str):
        response = str(response)
    return 1.0 if re.fullmatch(r"\(\s*\d+\s*,\s*\d+\s*\)", response.strip()) else 0.0


def _parse_prediction(response: Any) -> Optional[tuple[float, float]]:
    if not isinstance(response, str):
        response = str(response)
    match = re.fullmatch(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", response.strip())
    if not match:
        return None
    x = int(match.group(1))
    y = int(match.group(2))
    return x / PREDICTION_SCALE, y / PREDICTION_SCALE


def _load_ground_truth(ground_truth: Any) -> Optional[list[float]]:
    if hasattr(ground_truth, "__array__"):
        ground_truth = ground_truth.tolist()
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except Exception:
            return None
    if not isinstance(ground_truth, Sequence):
        return None
    try:
        return [float(v) for v in ground_truth]
    except Exception:
        return None


def is_inside_bbox(predicted: tuple[float, float], bbox: Sequence[float]) -> bool:
    px, py = predicted
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def is_inside_polygon(predicted: tuple[float, float], coords: Sequence[float]) -> bool:
    if len(coords) < 6 or len(coords) % 2 != 0:
        return False
    px, py = predicted
    inside = False
    j = len(coords) - 2
    for i in range(0, len(coords), 2):
        xi, yi = coords[i], coords[i + 1]
        xj, yj = coords[j], coords[j + 1]
        denom = (yj - yi) if abs(yj - yi) > 1e-12 else 1e-12
        intersects = ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / denom + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def normalized_distance_to_center(predicted: tuple[float, float], bbox: Sequence[float]) -> float:
    px, py = predicted
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 0.0)
    height = max(y2 - y1, 0.0)
    diag = math.hypot(width, height)
    if diag <= 0:
        return 1.0
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    dist = math.hypot(px - center_x, py - center_y)
    return min(dist / diag, 1.0)


def normalized_distance_to_polygon_centroid(predicted: tuple[float, float], coords: Sequence[float]) -> float:
    if len(coords) < 6 or len(coords) % 2 != 0:
        return 1.0
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    px, py = predicted
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    if diag <= 0:
        return 1.0
    dist = math.hypot(px - cx, py - cy)
    return min(dist / diag, 1.0)


def accuracy_reward(response: Any, ground_truth: Any) -> float:
    predicted = _parse_prediction(response)
    if predicted is None:
        return 0.0
    arr = _load_ground_truth(ground_truth)
    if arr is None:
        return 0.0

    if len(arr) == 4:
        if is_inside_bbox(predicted, arr):
            penalty = normalized_distance_to_center(predicted, arr)
            return max(0.0, 1.0 - penalty)
        return 0.0

    if len(arr) >= 6:
        if is_inside_polygon(predicted, arr):
            penalty = normalized_distance_to_polygon_centroid(predicted, arr)
            return max(0.0, 1.0 - penalty)
    return 0.0


def compute_score(reward_input: dict[str, Any], format_weight: float = 0.0) -> dict[str, float]:
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential` for norm gui reward function.")

    format_score = format_reward(reward_input["response"])
    accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    correct = 1.0 if accuracy_score > 0.0 else 0.0

    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
        "correct": correct,
    }
