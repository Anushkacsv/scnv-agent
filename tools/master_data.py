import json
import os
from typing import Dict, Any

# Path to the synthetic data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic")

def load_json(filename: str) -> list:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Warning: Data file {filepath} not found.")
        return []
    with open(filepath, "r") as f:
        return json.load(f)

# Load data into memory for tool usage
_skus = {s["sku_id"]: s for s in load_json("sku_master.json")}
_plants = {p["plant_id"]: p for p in load_json("plant_master.json")}

def check_master_data(sku_id: str, plant_id: str) -> Dict[str, Any]:
    """
    Checks master data for an SKU and a given destination plant.
    Returns the source model (SINGLE/DUAL), valid sourcing plants, and shelf life.
    """
    sku_info = _skus.get(sku_id)
    if not sku_info:
        return {"error": f"SKU {sku_id} not found in master data."}
    
    plant_info = _plants.get(plant_id)
    
    return {
        "sku_id": sku_id,
        "source_model": sku_info.get("source_model", "UNKNOWN"),
        "sourcing_plants": sku_info.get("sourcing_plants", []),
        "shelf_life_days": sku_info.get("shelf_life_days", 90),
        "destination_plant_exists": bool(plant_info),
        "is_sourcing_plant": plant_id in sku_info.get("sourcing_plants", [])
    }
