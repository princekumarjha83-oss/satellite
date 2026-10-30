import hashlib
import io
import unittest
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

from test_animal_audio import sample_data


class SeparatePageTests(unittest.TestCase):
    def test_navigation_isolates_workflows_and_preserves_work(self):
        with patch("requests.get", side_effect=requests.ConnectionError("offline test")) as network:
            app = AppTest.from_file("app.py", default_timeout=60).run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertFalse(any("Bioacoustic" in tab.label for tab in app.tabs))
            network.reset_mock()

            app.switch_page("bioacoustics.py").run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual([r.label for r in app.radio], ["Audio source"])
            self.assertFalse(app.tabs)
            data = sample_data(silent=True)
            clip = io.BytesIO(data)
            clip.name = "saved_recording.wav"
            app.session_state["saved_media_wildlife_audio"] = clip
            app.run()
            next(b for b in app.button if b.label == "Identify animal sounds").click().run()
            self.assertFalse(app.exception, str(app.exception))
            saved_report = app.session_state["audio_animal_result"]
            self.assertEqual(saved_report[0], hashlib.sha256(data).hexdigest())

            app.switch_page("live_camera.py").run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual([r.label for r in app.radio], ["Visual source"])
            self.assertFalse(app.get("plotly_chart"))
            app.text_area[0].set_value("lion, tiger").run()
            app.slider[0].set_value(.35).run()
            app.radio[0].set_value("Video").run()

            app.switch_page("bioacoustics.py").run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(len(app.get("plotly_chart")), 1)
            self.assertEqual(app.session_state["audio_animal_result"], saved_report)
            self.assertTrue(any("Restored from this session" in c.value for c in app.caption))

            app.switch_page("live_camera.py").run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(app.text_area[0].value, "lion, tiger")
            self.assertEqual(app.slider[0].value, .35)
            self.assertEqual(app.radio[0].value, "Video")
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
