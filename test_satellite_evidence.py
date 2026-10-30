import io
import unittest
from unittest.mock import patch

from PIL import Image
import requests
from streamlit.testing.v1 import AppTest

from satellite_evidence import inspect_satellite
from wildlife_reporting import build_report, save_evidence
from satellite_evidence import calibrated_spatial_measurements


class SatelliteEvidenceTests(unittest.TestCase):
    def test_calibrated_measurement_retains_raw_temporal_values(self):
        result = calibrated_spatial_measurements(.72, .81, "2026-09-21", "11.1,76.4", 12)
        self.assertEqual(result["ndvi"], .72)
        self.assertEqual(result["previous_ndvi"], .81)
        self.assertAlmostEqual(result["ndvi_change"], -.09)
        self.assertEqual(result["vegetation_change"], "decline")
        self.assertIn("User-supplied", result["measurement_provenance"])

    def image_file(self):
        image = io.BytesIO()
        Image.new("RGB", (48, 32), "green").save(image, format="PNG")
        image.name = "satellite.png"
        return image

    def test_inspection_is_based_on_image_and_rejects_invalid_files(self):
        details, preview = inspect_satellite(self.image_file().getvalue())
        self.assertEqual((details["width_pixels"], details["height_pixels"]), (48, 32))
        self.assertEqual(details["format"], "PNG")
        self.assertTrue(preview.startswith(b"\x89PNG"))
        with self.assertRaises(Exception):
            inspect_satellite(b"invalid")

    def test_satellite_is_not_counted_as_animal_analysis(self):
        state = {}
        details, _ = inspect_satellite(self.image_file().getvalue())
        save_evidence(state, "satellite", details["sha256"], "satellite.png", measurements=details)
        report = build_report(list(state["wildlife_evidence"].values()), "", "", "today", "")
        self.assertEqual(report["satellite_count"], 1)
        self.assertEqual(report["analyzed_count"], 0)
        self.assertEqual(report["findings"], [])
        self.assertIn("not been assessed", report["summary"])

    def test_report_upload_includes_preview_and_satellite_card(self):
        with patch("requests.get", side_effect=requests.ConnectionError("offline test")):
            app = AppTest.from_file("app.py", default_timeout=60).run()
            with patch("satellite_evidence.render_satellite_upload"):
                app.switch_page("wildlife_reports.py").run()
            with patch("streamlit.file_uploader", return_value=self.image_file()):
                app.run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(len(app.session_state["wildlife_evidence"]), 1)
            self.assertTrue(any(m.label == "Images included" and m.value == "1" for m in app.metric))

    def test_older_report_without_satellite_count_renders_and_filters(self):
        def legacy_builder(*args, **kwargs):
            report = build_report(*args, **kwargs)
            report.pop("satellite_count")
            return report

        state = {}
        details, _ = inspect_satellite(self.image_file().getvalue())
        save_evidence(state, "satellite", details["sha256"], "satellite.png", measurements=details)
        with patch("requests.get", side_effect=requests.ConnectionError("offline test")), \
             patch("wildlife_reporting.build_report", side_effect=legacy_builder):
            app = AppTest.from_file("app.py", default_timeout=60).run()
            app.session_state["wildlife_evidence"] = state["wildlife_evidence"]
            app.switch_page("wildlife_reports.py").run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(next(m.value for m in app.metric if m.label == "Images included"), "1")
            app.multiselect[0].set_value([]).run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(next(m.value for m in app.metric if m.label == "Images included"), "0")


if __name__ == "__main__":
    unittest.main()
