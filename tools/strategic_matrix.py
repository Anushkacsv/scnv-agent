import json
import os
from typing import Dict, Any

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic")

def load_json(filename: str) -> list:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        return json.load(f)

# Load strategic matrix into memory
_matrix_data = load_json("strategic_matrix.json")
_matrix = {(item["source"], item["destination"]): item for item in _matrix_data}

def check_strategic_matrix(source_id: str, dest_id: str) -> Dict[str, Any]:
    """
    Checks if a lane between source and destination is considered strategic,
    and returns its capacity utilization.
    """
    lane_info = _matrix.get((source_id, dest_id))
    
    if lane_info:
        return {
            "source": source_id,
            "destination": dest_id,
            "is_strategic_lane": lane_info["is_strategic_lane"],
            "capacity_utilization": lane_info["capacity_utilization"],
            "lane_exists": True
        }
    else:
        # Default fallback for unmapped lanes in the synthetic data
        return {
            "source": source_id,
            "destination": dest_id,
            "is_strategic_lane": False,
            "capacity_utilization": 0.8, # Mock default capacity
            "lane_exists": False
        }
