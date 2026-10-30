import io
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from animal_audio import ANIMAL_SOUNDS, BACKGROUNDS, decide_match, identify_audio_animals


def sample_data(seconds=1, sample_rate=16000, silent=False):
    samples = np.zeros(int(seconds * sample_rate)) if silent else .2 * np.sin(
        2 * np.pi * 440 * np.arange(int(seconds * sample_rate)) / sample_rate)
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    return buffer.getvalue()


def scores_for(name, value=.5):
    names = [a[0] for a in ANIMAL_SOUNDS] + [b[0] for b in BACKGROUNDS]
    scores = np.full(len(names), .1)
    scores[names.index(name)] = value
    return scores


class AnimalAudioTests(unittest.TestCase):
    def test_animal_match_requires_both_similarity_and_margin(self):
        self.assertEqual(decide_match(scores_for("Dog"))["animal"], "Dog")
        weak = decide_match(scores_for("Dog", .2))
        self.assertEqual(weak["status"], "uncertain")
        ambiguous = scores_for("Lion", .5)
        ambiguous[[a[0] for a in ANIMAL_SOUNDS].index("Tiger")] = .49
        self.assertIsNone(decide_match(ambiguous)["animal"])

    def test_background_never_reports_animal(self):
        result = decide_match(scores_for("Rain"))
        self.assertEqual(result["status"], "background")
        self.assertIsNone(result["animal"])

    def test_invalid_scores_rejected(self):
        with self.assertRaises(ValueError):
            decide_match([.5])
        scores = scores_for("Dog")
        scores[0] = np.nan
        with self.assertRaises(ValueError):
            decide_match(scores)

    def test_silence_does_not_load_model(self):
        with patch("animal_audio.get_audio_matcher") as load:
            report = identify_audio_animals(sample_data(6, silent=True))
        load.assert_not_called()
        self.assertEqual(report["animals"], [])
        self.assertEqual([w["status"] for w in report["windows"]], ["silence", "silence"])

    def test_too_short_audio_not_classified(self):
        with patch("animal_audio.get_audio_matcher") as load:
            report = identify_audio_animals(sample_data(.05))
        load.assert_not_called()
        self.assertEqual(report["windows"][0]["status"], "uncertain")

    def test_window_coverage_multiple_animals_and_resampling(self):
        match = MagicMock(side_effect=[scores_for("Dog"), scores_for("Cat"), scores_for("Dog")])
        progress = MagicMock()
        with patch("animal_audio.get_audio_matcher", return_value=(match, threading.Lock())):
            report = identify_audio_animals(sample_data(11), progress)
        self.assertEqual([(w["start_seconds"], w["end_seconds"]) for w in report["windows"]], [(0, 5), (5, 10), (10, 11)])
        self.assertEqual([len(call.args[0]) for call in match.call_args_list], [240000, 240000, 48000])
        waveform = match.call_args_list[0].args[0]
        dominant = np.argmax(np.abs(np.fft.rfft(waveform))) * 48000 / len(waveform)
        self.assertAlmostEqual(dominant, 440, delta=1)
        animals = {a["animal"]: a for a in report["animals"]}
        self.assertEqual(animals["Dog"]["windows_matched"], 2)
        self.assertEqual(animals["Cat"]["intervals"], [[5, 10]])
        progress.assert_called_with(3, 3)

    def test_model_failure_does_not_return_partial_success(self):
        match = MagicMock(side_effect=[scores_for("Dog"), RuntimeError("model failure")])
        with patch("animal_audio.get_audio_matcher", return_value=(match, threading.Lock())):
            with self.assertRaises(RuntimeError):
                identify_audio_animals(sample_data(6))

    def test_ui_renders_report_and_replay(self):
        from streamlit.testing.v1 import AppTest
        source = "from test_animal_audio import sample_data\nfrom wildlife_ui import render_audio_identification\nrender_audio_identification(sample_data(), 'test.wav')"
        match = MagicMock(return_value=scores_for("Dog"))
        with patch("animal_audio.get_audio_matcher", return_value=(match, threading.Lock())):
            app = AppTest.from_string(source).run()
            app.button[0].click().run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(app.selectbox[0].value, "Dog")
            self.assertEqual(app.dataframe[0].value.iloc[0]["Animal"], "Dog")
            app.run()
        match.assert_called_once()

    def test_ui_does_not_show_stale_report_for_another_file(self):
        from streamlit.testing.v1 import AppTest
        source = "from test_animal_audio import sample_data\nfrom wildlife_ui import render_audio_identification\nimport streamlit as st\nn=st.number_input('Duration', min_value=1, value=1)\nrender_audio_identification(sample_data(n), 'test.wav')"
        with patch("animal_audio.get_audio_matcher", return_value=(lambda samples: scores_for("Dog"), threading.Lock())):
            app = AppTest.from_string(source).run()
            app.button[0].click().run()
            self.assertTrue(app.selectbox)
            app.number_input[0].set_value(2).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.selectbox)

    def test_uploaded_audio_shows_spectrogram_and_animal_panel(self):
        from streamlit.testing.v1 import AppTest
        source = ("import io\nfrom unittest.mock import patch\n"
                  "from test_animal_audio import sample_data\n"
                  "from wildlife_ui import render_audio\n"
                  "clip=io.BytesIO(sample_data())\nclip.name='uploaded.wav'\n"
                  "with patch('wildlife_ui.st.file_uploader', return_value=clip):\n    render_audio()")
        with patch("animal_audio.get_audio_matcher", return_value=(lambda samples: scores_for("Cat"), threading.Lock())):
            app = AppTest.from_string(source).run()
            app.button[0].click().run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertEqual(app.selectbox[0].value, "Cat")


if __name__ == "__main__":
    unittest.main()
