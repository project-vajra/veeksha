import unittest
from veeksha.config.slo import ConstantSloConfig
from veeksha.slo.runner import _get_slo_metric_keys

class TestSloRunner(unittest.TestCase):
    def test_get_slo_metric_keys_mapping(self):
        configs = [
            ConstantSloConfig(metric="e2e", value=1.0),
            ConstantSloConfig(metric="tpot", value=0.1),
            ConstantSloConfig(metric="ttfc", value=0.5),
        ]
        keys = _get_slo_metric_keys(configs)
        
        # e2e should map to end_to_end_latency
        self.assertIn("end_to_end_latency", keys)
        # tpot should be present
        self.assertIn("tpot", keys)
        # ttfc should be present
        self.assertIn("ttfc", keys)

if __name__ == "__main__":
    unittest.main()
