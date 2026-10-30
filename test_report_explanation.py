import unittest

from report_explanation import generate_explanation


class ExplanationTests(unittest.TestCase):
    def test_does_not_claim_missing_satellite_measurements(self):
        evidence = {"features": {"camera": {"raw": {"species_count": 1, "wildlife_observations": 2}},
                     "audio": {"raw": {}}, "satellite": {"raw": {}}},
                    "fusion": {"available_modalities": ["camera"], "confidence": "incomplete evidence"},
                    "agreement": {"explanation": "Agreement unavailable."}, "anomaly": {"explanation": "Anomaly unavailable."},
                    "limitations": ["Satellite evidence is excluded."], "missing_data_flags": ["audio", "satellite"]}
        result = generate_explanation(evidence, [])
        self.assertIn("Camera analysis", result["observed_data"][0])
        self.assertNotIn("Calibrated satellite", " ".join(result["observed_data"]))
        self.assertTrue(result["human_review_required"])
