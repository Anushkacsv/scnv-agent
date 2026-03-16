import random
from typing import Dict, Any

def calculate_logistics_cost(source_id: str, dest_id: str, volume: float) -> Dict[str, Any]:
    """
    Calculates transport costs and distances between two locations.
    In Phase 1, returns simulated cost algorithms based on a randomized distance.
    """
    
    # Distance in km (simulated based on random distance between plants/DCs)
    distance_km = random.uniform(50.0, 1500.0)
    
    # Volume is likely in KG or HL. We assume a cost of ~$0.05 per km per unit volume
    base_rate_per_km_per_unit = 0.05
    
    # Apply some random regional modifier
    regional_modifier = random.uniform(0.8, 1.3)
    
    total_cost = distance_km * base_rate_per_km_per_unit * volume * regional_modifier
    
    return {
        "source": source_id,
        "destination": dest_id,
        "volume": volume,
        "distance_km": round(distance_km, 2),
        "total_cost": round(total_cost, 2),
        "cost_per_unit": round(total_cost / max(volume, 1.0), 2)
    }
