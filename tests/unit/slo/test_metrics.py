import unittest

from veeksha.slo.metrics import extract_metric_values, lower_is_better


class TestMetrics(unittest.TestCase):
    def test_duplex_success_metrics_are_higher_is_better(self):
        for metric in (
            "audio_before_commit_ratio",
            "duplex_overlap_observed",
            "user_audio_fluidity_index",
            "tts_service_fluidity_index",
            "tts_service_fluidity_eligible",
        ):
            self.assertFalse(lower_is_better(metric))
        self.assertTrue(lower_is_better("ttfc"))

    def test_extract_metric_values(self):
        metrics = {
            "ttfc": [0.1, 0.2, 0.3],
            "tbc": [[0.01, 0.02], [0.03, 0.04]],
            "end_to_end_latency": [1.0, 2.0, 3.0],
            "tpot": [0.001, 0.002, 0.003],
            "other": [10, 20],
        }

        # Test ttfc
        self.assertEqual(extract_metric_values("ttfc", metrics), [0.1, 0.2, 0.3])

        # Test tbc flattening
        self.assertEqual(
            extract_metric_values("tbc", metrics), [0.01, 0.02, 0.03, 0.04]
        )

        # Test e2e mapping
        self.assertEqual(extract_metric_values("e2e", metrics), [1.0, 2.0, 3.0])

        # Test tpot mapping
        self.assertEqual(extract_metric_values("tpot", metrics), [0.001, 0.002, 0.003])

        # Test unknown
        self.assertEqual(extract_metric_values("unknown", metrics), [])

        # Test empty
        self.assertEqual(extract_metric_values("ttfc", {}), [])


if __name__ == "__main__":
    unittest.main()
