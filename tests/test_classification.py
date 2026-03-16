import unittest
import os
import sys

# Add tools and agents directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tools"))

from scm_analyst import STOClassifier

class TestSTOClassifier(unittest.TestCase):
    
    def setUp(self):
        self.classifier = STOClassifier()
        
    def test_dc_to_dc_lateral(self):
        sto = {"source_location": "NonExistentPlant1", "destination_location": "NonExistentPlant2", "sku_id": "AnySKU"}
        result = self.classifier.classify_sto(sto)
        self.assertEqual(result["classification"], "UNPRODUCTIVE")
        self.assertEqual(result["rule_applied"], 2)
        
    def test_dc_to_plant_reverse(self):
        # We need to mock or use a known plant for destination
        # For our test, if we assume 'Plant_A' is a plant (we'd have to mock master_data response, 
        # but since it returns False for unknowns, we can just test the logic path if we mock check_master_data)
        pass 
        
    # We will test the engine on the full synthetic dataset next
        
if __name__ == '__main__':
    unittest.main()
