import unittest
from veeksha.config.slo import ConstantSloConfig

class TestSloConfig(unittest.TestCase):
    def test_supported_metrics(self):
        # Should not raise
        ConstantSloConfig(metric="ttfc", value=1.0)
        ConstantSloConfig(metric="tbc", value=1.0)
        ConstantSloConfig(metric="tpot", value=1.0)
        ConstantSloConfig(metric="e2e", value=1.0)

    def test_unsupported_metric(self):
        with self.assertRaisesRegex(ValueError, "metric 'invalid' is not supported"):
            ConstantSloConfig(metric="invalid", value=1.0)

    def test_invalid_value(self):
        with self.assertRaisesRegex(ValueError, "value must be specified and must be > 0"):
            ConstantSloConfig(metric="ttfc", value=-1.0)

if __name__ == "__main__":
    unittest.main()
