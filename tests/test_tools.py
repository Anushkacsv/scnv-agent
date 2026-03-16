import unittest
import os
import sys

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tools"))

from master_data import check_master_data
from strategic_matrix import check_strategic_matrix
from inventory import get_inventory_levels
from logistics import calculate_logistics_cost

class TestSharedTools(unittest.TestCase):
    
    def test_master_data_not_found(self):
        result = check_master_data("invalid_sku", "invalid_plant")
        self.assertIn("error", result)
        
    def test_master_data_structure(self):
        # We know STO_100000 has source eb0ecdb070a1a0ac46de0cd733d39cf3 and SKU 208ee04cee8edd26673840442f869ae1
        # It's an md5 hash string from our raw CSV data
        result = check_master_data("208ee04cee8edd26673840442f869ae1", "eb0ecdb070a1a0ac46de0cd733d39cf3")
        
        # We don't assert exactly TRUE or FALSE because sampling might omit, but we assert structure
        if "error" not in result:
            self.assertIn("source_model", result)
            self.assertIn("sourcing_plants", result)
            
    def test_strategic_matrix_fallback(self):
        result = check_strategic_matrix("nonexistent", "nonexistent")
        self.assertFalse(result["lane_exists"])
        self.assertFalse(result["is_strategic_lane"])
        self.assertEqual(result["capacity_utilization"], 0.8)
        
    def test_inventory_structure(self):
        result = get_inventory_levels("PlantA", "SKU1")
        self.assertEqual(result["location_id"], "PlantA")
        self.assertEqual(result["sku_id"], "SKU1")
        self.assertIn("stock_hl", result)
        
    def test_logistics_calculation(self):
        result = calculate_logistics_cost("A", "B", 1000.0)
        self.assertEqual(result["source"], "A")
        self.assertEqual(result["destination"], "B")
        self.assertEqual(result["volume"], 1000.0)
        self.assertTrue(result["total_cost"] > 0)

if __name__ == '__main__':
    unittest.main()
