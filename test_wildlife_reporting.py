import unittest
import io
import sys
import types
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest
from PIL import Image

from wildlife_reporting import build_report, import_session_evidence, markdown_report, save_evidence


def audio_result():
    return {"source": "calls.wav", "model": "test-audio-model", "duration_seconds": 10,
            "animals": [{"animal": "Lion", "max_similarity": .4, "windows_matched": 1,
                         "intervals": [[0, 5]], "call_description": "roaring"}],
            "windows": [{"status": "possible_animal", "animal": "Lion"},
                        {"status": "uncertain", "closest_match": "Tiger"}]}


def visual_result():
    return {"source": "clip.mp4", "model": "test-visual-model", "media_type": "video", "media_sha256": "video-hash",
            "duration_seconds": 8, "frames_analyzed": 2, "coverage": "Sampled 2 frames; other frames were not analyzed.",
            "animals": [{"animal": "lion", "max_score": .8, "observations": 2,
                         "first_seen_seconds": 0, "last_seen_seconds": 5}]}


class WildlifeReportingTests(unittest.TestCase):
    def test_reports_loads_with_old_wildlife_ui_module_and_keeps_fields(self):
        # Reproduce a running server with an older imported UI module that has
        # no saved_control attribute. Reports must not depend on that module.
        old_ui = types.ModuleType("wildlife_ui")
        with patch.dict(sys.modules, {"wildlife_ui": old_ui}), \
             patch("requests.get", side_effect=requests.ConnectionError("offline test")):
            app = AppTest.from_file("app.py", default_timeout=60).run()
            app.switch_page("wildlife_reports.py").run()
            self.assertFalse(app.exception, str(app.exception))
            app.text_input[0].set_value("My monitoring project").run()
            app.switch_page("forest_dashboard.py").run()
            app.switch_page("wildlife_reports.py").run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(app.text_input[0].value, "My monitoring project")

    def test_registry_keeps_multiple_sources_and_does_not_downgrade_analysis(self):
        state = {}
        result = audio_result()
        save_evidence(state, "audio", "audio-hash", "calls.wav", analysis=result)
        result["animals"].clear()
        save_evidence(state, "audio", "audio-hash", "calls.wav", measurements={"rms": .1})
        save_evidence(state, "video", "video-hash", "clip.mp4", analysis=visual_result())
        self.assertEqual(len(state["wildlife_evidence"]), 2)
        record = state["wildlife_evidence"]["audio:audio-hash"]
        self.assertEqual(record["analysis"]["animals"][0]["animal"], "Lion")
        self.assertEqual(record["measurements"]["rms"], .1)

    def test_shared_labels_are_not_claimed_as_confirmed_events(self):
        state = {}
        save_evidence(state, "audio", "a", "calls.wav", analysis=audio_result())
        save_evidence(state, "video", "v", "clip.mp4", analysis=visual_result())
        report = build_report(list(state["wildlife_evidence"].values()), "Project", "B3", "2026-09-21", "My notes")
        self.assertEqual(report["shared_labels"], ["lion"])
        self.assertEqual(report["analyzed_count"], 2)
        self.assertIn("does not confirm", report["cross_source_analysis"])
        self.assertEqual(report["findings"][0]["score_type"], "Audio cosine similarity")
        self.assertEqual(report["findings"][1]["score_type"], "Visual matching score")
        self.assertTrue(any("uncertain" in event for event in report["events"]))
        self.assertEqual(report["user_notes"], "My notes")
        exported = markdown_report(report)
        self.assertIn("calls.wav", exported)
        self.assertIn("Sampled 2 frames", exported)
        self.assertNotIn("NDVI", exported)
        self.assertNotIn("82 / 100", exported)

    def test_pending_and_no_match_are_distinct(self):
        state = {}
        save_evidence(state, "image", "pending", "pending.jpg")
        save_evidence(state, "image", "empty", "empty.jpg", analysis={"animals": [], "frames_analyzed": 1})
        report = build_report(list(state["wildlife_evidence"].values()), "", "", "today", "")
        self.assertEqual(report["analyzed_count"], 1)
        self.assertTrue(any("not been completed" in e for e in report["events"]))
        self.assertTrue(any("no animal match" in e for e in report["events"]))

    def test_current_session_results_import_without_duplicates(self):
        state = {"audio_animal_result": ("audio-hash", audio_result()),
                 "wildlife_result": ("settings-hash", visual_result(), None)}
        import_session_evidence(state)
        import_session_evidence(state)
        self.assertEqual(len(state["wildlife_evidence"]), 2)
        self.assertIn("video:video-hash", state["wildlife_evidence"])

    def test_empty_report_does_not_invent_findings(self):
        report = build_report([], "", "", "today", "lion confirmed by operator")
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["possible_animal_labels"], [])
        self.assertIn("No user evidence", report["summary"])

    def test_human_review_gate_is_exported(self):
        report = build_report([], "", "", "today", "", human_review_confirmed=True)
        self.assertTrue(report["human_review"]["confirmed"])
        self.assertIn("Human review confirmed", markdown_report(report))

    def test_report_page_uses_only_selected_evidence_without_inference(self):
        with patch("requests.get", side_effect=requests.ConnectionError("offline test")) as network:
            app = AppTest.from_file("app.py", default_timeout=60).run()
            network.reset_mock()
            app.switch_page("wildlife_reports.py").run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertTrue(any("No evidence yet" in info.value for info in app.info))
            app.session_state["audio_animal_result"] = ("audio-hash", audio_result())
            app.session_state["wildlife_result"] = ("settings-hash", visual_result(), None)
            app.run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(len(app.dataframe[0].value), 2)
            app.multiselect[0].set_value(["audio:audio-hash"]).run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(len(app.dataframe[0].value), 1)
            self.assertEqual(app.dataframe[0].value.iloc[0]["evidence"], "Audio")
            network.assert_not_called()

    def test_completed_image_analysis_flows_into_reports(self):
        image = Image.new("RGB", (32, 32))
        media = io.BytesIO()
        image.save(media, format="PNG")
        media.name = "uploaded_lion.png"
        detection = {"animal": "lion", "score": .8, "box": [0, 0, 20, 20]}
        with patch("requests.get", side_effect=requests.ConnectionError("offline test")), \
             patch("wildlife_ui.ready_detector", return_value=True), \
             patch("wildlife_ui.detect_animals", return_value=(image, [detection], .1)) as detector:
            app = AppTest.from_file("app.py", default_timeout=60).run()
            app.session_state["saved_media_wildlife_Image"] = media
            app.switch_page("live_camera.py").run()
            next(b for b in app.button if b.label == "Analyze animals").click().run()
            self.assertFalse(app.exception, str(app.exception))
            app.switch_page("wildlife_reports.py").run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(app.dataframe[0].value.iloc[0]["source"], "uploaded_lion.png")
            self.assertEqual(app.dataframe[0].value.iloc[0]["animal"], "lion")
            self.assertEqual(len(app.session_state["wildlife_evidence"]), 1)
            detector.assert_called_once()


if __name__ == "__main__":
    unittest.main()
