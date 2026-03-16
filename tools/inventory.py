import random
from typing import Dict, Any

def get_inventory_levels(location_id: str, sku_id: str) -> Dict[str, Any]:
    """
    Mock function to simulate checking real-time inventory levels.
    In Phase 1, returns randomized but plausible stock values.
    """
    
    # In a real scenario, this would query SAP or our MCHB stock tables.
    # For synthetic testing, we return randomized stock levels.
    current_stock = random.uniform(0.0, 50000.0)
    
    return {
        "location_id": location_id,
        "sku_id": sku_id,
        "stock_hl": current_stock,
        "incoming_hl": random.uniform(0.0, 10000.0) if random.random() > 0.5 else 0.0,
        "outgoing_hl": random.uniform(0.0, current_stock * 0.8), # Can't ship more than we have
        "is_stockout_risk": current_stock < 1000.0
    }
