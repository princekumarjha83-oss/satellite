import unittest

from ecosystem_pipeline import agreement_score, anomaly_score, build_evidence_object, parse_historical_scores, weighted_fusion


class EcosystemPipelineTests(unittest.TestCase):
    def test_missing_modality_is_renormalized_not_zero(self):
        result = weighted_fusion({"camera": {"normalized_score": 80}, "audio": {"normalized_score": None}, "satellite": {"normalized_score": 60}})
        self.assertEqual(result["fusion_score"], 68.58)
        self.assertEqual(result["missing_modalities"], ["audio"])

    def test_custom_weights_are_retained_in_fusion_output(self):
        result = weighted_fusion({"camera": {"normalized_score": 80}, "audio": {"normalized_score": 20}, "satellite": {"normalized_score": None}},
                                 {"camera": .8, "audio": .2, "satellite": 0})
        self.assertEqual(result["fusion_score"], 68.0)
        self.assertEqual(result["weights"], {"camera": .8, "audio": .2})

    def test_agreement_distinguishes_consistent_evidence(self):
        high = agreement_score({"camera": {"normalized_score": 82}, "audio": {"normalized_score": 79}, "satellite": {"normalized_score": 84}})
        low = agreement_score({"camera": {"normalized_score": 30}, "audio": {"normalized_score": 86}, "satellite": {"normalized_score": 84}})
        self.assertEqual(high["agreement_level"], "high")
        self.assertEqual(low["agreement_level"], "low")

    def test_anomaly_uses_historical_baseline(self):
        result = anomaly_score(20, [79, 81, 80, 82, 78])
        self.assertTrue(result["is_anomaly"])

    def test_historical_score_parser_requires_valid_baseline(self):
        self.assertEqual(parse_historical_scores("70, 72, 71"), [70.0, 72.0, 71.0])
        with self.assertRaises(ValueError):
            parse_historical_scores("70, invalid, 71")

    def test_rgb_satellite_is_not_converted_to_ndvi(self):
        evidence = build_evidence_object([{"kind": "satellite", "measurements": {"width_pixels": 100}}], "P", "Z")
        self.assertIsNone(evidence["normalized_scores"]["satellite"])
        self.assertIsNone(evidence["mehs"])

    def test_calibrated_satellite_measurement_is_available_for_fusion(self):
        evidence = build_evidence_object([{"kind": "satellite", "measurements": {"ndvi": .72, "previous_ndvi": .80}}], "P", "Z")
        self.assertEqual(evidence["normalized_scores"]["satellite"], 72.0)
        self.assertEqual(evidence["fusion"]["available_modalities"], ["satellite"])
