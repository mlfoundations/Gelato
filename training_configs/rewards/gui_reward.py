import re
import json
from typing import Any

def format_reward(response: Any) -> float:
    # Check if the response is in the format of (x, y) where x and y are integers
    # Convert to string if it's not already
    if not isinstance(response, str):
        response = str(response)
    pattern = re.compile(r"^\(\d+,\s*\d+\)$", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0

def is_inside_bbox(predicted_coords: str, ground_truth_bbox: str) -> bool:
    # Check if the predicted coordinates are inside the ground truth bounding box
    x, y = predicted_coords.strip().split(",")
    x, y = int(x.strip()), int(y.strip())
    
    # Parse JSON array format [x1, y1, x2, y2]
    bbox = json.loads(ground_truth_bbox)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    return x1 <= x <= x2 and y1 <= y <= y2

def is_inside_polygon(predicted_coords: str, ground_truth_polygon: str) -> bool:
    # Ray casting algorithm to determine if a point is inside a polygon.
    # ground_truth_polygon is a JSON array [x1,y1,x2,y2,...,xn,yn]
    try:
        x_str, y_str = predicted_coords.strip().split(",")
        px, py = int(x_str.strip()), int(y_str.strip())
        coords = json.loads(ground_truth_polygon)
        if not isinstance(coords, list) or len(coords) < 6:
            return False
        points = []
        for i in range(0, len(coords), 2):
            points.append((float(coords[i]), float(coords[i+1])))
        inside = False
        j = len(points) - 1
        for i in range(len(points)):
            xi, yi = points[i]
            xj, yj = points[j]
            intersects = ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) if (yj - yi) != 0 else 1e-12) + xi)
            if intersects:
                inside = not inside
            j = i
        return inside
    except Exception:
        return False

def normalized_distance_to_center(predicted_coords: str, ground_truth_bbox: str) -> float:
    # Calculate the normalized distance from the predicted coordinates to the center of the ground truth bounding box (normalized by the width and height of the bounding box)
    x, y = predicted_coords.strip().split(",")
    x, y = int(x.strip()), int(y.strip())
    
    # Parse JSON array format [x1, y1, x2, y2]
    bbox = json.loads(ground_truth_bbox)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2
    center_y = y1 + height / 2
    return ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 / ((width ** 2 + height ** 2) ** 0.5)

def normalized_distance_to_polygon_centroid(predicted_coords: str, ground_truth_polygon: str) -> float:
    # Distance from point to polygon centroid normalized by polygon bbox diagonal
    x_str, y_str = predicted_coords.strip().split(",")
    x, y = int(x_str.strip()), int(y_str.strip())
    coords = json.loads(ground_truth_polygon)
    if not isinstance(coords, list) or len(coords) < 6:
        return 1.0
    points = []
    for i in range(0, len(coords), 2):
        points.append((float(coords[i]), float(coords[i+1])))
    # centroid as average of vertices
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    # bbox diagonal for normalization
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if diag <= 0:
        return 1.0
    dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    return dist / diag

def accuracy_reward(response: Any, ground_truth: Any) -> float:
    # Check if the predicted coordinates are inside the ground truth region (bbox or polygon)
    # and if so penalize it based on distance to the region center/centroid
    # Convert response to string if it's not already
    if not isinstance(response, str):
        response = str(response)
    
    content_match = re.search(r"^\((\d+,\s*\d+)\)$", response.strip())
    try:
        given_answer = content_match.group(1).strip() if content_match else None
    except Exception:
        given_answer = None

    if given_answer is None:
        return 0.0
    
    # Support passing in list/array or JSON string for both bbox [x1,y1,x2,y2]
    # or polygon [x1,y1,...,xn,yn]
    if hasattr(ground_truth, '__array__'):
        ground_truth = ground_truth.tolist()
    if isinstance(ground_truth, list):
        ground_truth = json.dumps(ground_truth)

    try:
        arr = json.loads(ground_truth)
    except Exception:
        return 0.0

    # BBox case
    if isinstance(arr, list) and len(arr) == 4:
        if is_inside_bbox(given_answer, json.dumps(arr)):
            return 1.0 - normalized_distance_to_center(given_answer, json.dumps(arr))
        return 0.0

    # Polygon case
    if isinstance(arr, list) and len(arr) >= 6:
        if is_inside_polygon(given_answer, json.dumps(arr)):
            # Use distance to centroid normalized by bbox diagonal
            return 1.0 - normalized_distance_to_polygon_centroid(given_answer, json.dumps(arr))
        return 0.0

    return 0.0

def compute_score(reward_input: dict[str, Any], format_weight: float = 0.0) -> dict[str, float]: 
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential` for gui reward function.")

    format_score = format_reward(reward_input["response"]) # Note: format_reward is not used in our experiments
    accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])

    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
        "correct": float(accuracy_score > 0.0)
    }
