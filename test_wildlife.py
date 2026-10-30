import io
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image
import soundfile as sf

from wildlife_analysis import (
    DEFAULT_ANIMALS, analyze_audio, animal_labels, detect_animals,
    generate_picture_name, make_report,
)


def wav(samples, rate=16000):
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format="WAV")
    return buffer.getvalue()


class AudioTests(unittest.TestCase):
    def test_frequency_matches_recording(self):
        tone = 0.4 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)
        result = analyze_audio(wav(np.column_stack((tone, tone))))
        self.assertEqual(result["duration_seconds"], 1)
        self.assertEqual(result["channels"], 2)
        self.assertAlmostEqual(result["dominant_frequency_hz"], 1000, delta=16)
        self.assertAlmostEqual(result["rms"], 0.4 / np.sqrt(2), places=3)
        self.assertEqual(result["db"].shape, (len(result["frequencies"]), len(result["times"])))

    def test_silence_does_not_create_species_or_frequency(self):
        result = analyze_audio(wav(np.zeros(16000)))
        self.assertEqual(result["aci"], 0)
        self.assertIsNone(result["dominant_frequency_hz"])
        self.assertTrue(np.isfinite(result["db"]).all())

    def test_invalid_and_too_short_audio(self):
        with self.assertRaises(Exception):
            analyze_audio(b"not audio")
        with self.assertRaises(ValueError):
            analyze_audio(wav(np.zeros(12)))


class VisualTests(unittest.TestCase):
    def test_detections_are_based_on_model_and_duplicate_boxes_removed(self):
        detector = MagicMock(return_value=[
            {"label": "a photo of a lion", "score": .8, "box": dict(xmin=2, ymin=2, xmax=50, ymax=50)},
            {"label": "a photo of a tiger", "score": .4, "box": dict(xmin=3, ymin=3, xmax=50, ymax=50)},
        ])
        with patch("wildlife_analysis.get_detector", return_value=(detector, threading.Lock())):
            image, detections, _ = detect_animals(Image.new("RGB", (100, 100)), ["lion", "tiger"])
        self.assertEqual([d["animal"] for d in detections], ["lion"])
        self.assertEqual(image.size, (100, 100))
        self.assertEqual(detector.call_args.kwargs["candidate_labels"], ["a photo of a lion", "a photo of a tiger"])

    def test_empty_model_output_produces_no_animals(self):
        with patch("wildlife_analysis.get_detector", return_value=(MagicMock(return_value=[]), threading.Lock())):
            _, detections, _ = detect_animals(Image.new("RGB", (50, 50)), ["lion"])
        self.assertEqual(detections, [])
        report = make_report("blank", [{"seconds": 0, "detections": detections}], 1, .15, ["lion"])
        self.assertEqual(report["animals"], [])

    def test_report_preserves_timestamps(self):
        observations = [{"seconds": t, "detections": [{"animal": "lion", "score": s, "box": [0, 0, 2, 2]}]}
                        for t, s in [(1, .3), (5, .6)]]
        lion = make_report("clip", observations, 2, .15, ["lion"])["animals"][0]
        self.assertEqual((lion["first_seen_seconds"], lion["last_seen_seconds"]), (1, 5))
        self.assertEqual(lion["observations"], 2)
        self.assertEqual(lion["max_score"], .6)

    def test_labels(self):
        self.assertEqual(animal_labels(" Lion, tiger, lion "), ["lion", "tiger"])
        with self.assertRaises(ValueError):
            animal_labels(" , ")

    def test_picture_name_generation_and_defaults(self):
        self.assertIn("person", DEFAULT_ANIMALS)
        self.assertIn("human", DEFAULT_ANIMALS)
        name_person = generate_picture_name([{"animal": "person", "score": 0.88}])
        self.assertTrue(name_person.startswith("person_"))
        self.assertTrue(name_person.endswith(".jpg"))

        name_multi = generate_picture_name([
            {"animal": "person", "score": 0.90},
            {"animal": "dog", "score": 0.75},
        ])
        self.assertTrue(name_multi.startswith("person_dog_"))

        name_empty = generate_picture_name([])
        self.assertTrue(name_empty.startswith("camera_photo_"))

    def test_video_decodes_actual_frames(self):
        import cv2
        from wildlife_ui import analyze_video
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "clip.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
            self.assertTrue(writer.isOpened())
            for i in range(20):
                writer.write(np.full((64, 64, 3), i * 10, dtype=np.uint8))
            writer.release()
            with open(path, "rb") as source:
                data = source.read()
        means = []
        def detector(image, labels, threshold):
            means.append(float(np.asarray(image).mean()))
            return image, [], 0.0
        with patch("wildlife_ui.detect_animals", side_effect=detector), patch("wildlife_ui.st.progress"), patch("wildlife_ui.st.empty"):
            report, _ = analyze_video(data, "clip.avi", ["lion"], .15, 1)
        self.assertEqual(report["frames_analyzed"], 2)
        self.assertEqual([o["seconds"] for o in report["observations"]], [0, 1])
        self.assertGreater(means[1], means[0] + 80)

    def test_live_frames_run_detector(self):
        import av
        from wildlife_ui import LiveProcessor
        processor = LiveProcessor(["lion"], .15)
        frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64)))
        with patch("wildlife_ui.detect_animals", return_value=(frame.to_image(), [], .1)) as detector:
            result = processor.recv(frame)
        self.assertEqual(result.width, 64)
        self.assertEqual(processor.frames, 1)
        self.assertEqual(processor.observations[0]["detections"], [])
        detector.assert_called_once()


class BackendAudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_measures_audio_and_rejects_invalid_file(self):
        import backend
        from fastapi import UploadFile, HTTPException
        with tempfile.TemporaryDirectory() as folder, patch.object(backend, "AUDIO_DIR", Path(folder)), patch.object(backend, "ingestion_ledger", []):
            tone = .2 * np.sin(2 * np.pi * 500 * np.arange(16000) / 16000)
            result = await backend.process_audio(UploadFile(file=io.BytesIO(wav(tone)), filename="tone.wav"), "test", "mic")
            self.assertEqual(result.species_list, [])
            self.assertEqual(result.species_detected, 0)
            self.assertAlmostEqual(result.dominant_frequency_hz, 500, delta=16)
            self.assertEqual(len(backend.ingestion_ledger), 1)
            with self.assertRaises(HTTPException) as context:
                await backend.process_audio(UploadFile(file=io.BytesIO(b"invalid"), filename="bad.wav"), "test", "mic")
            self.assertEqual(context.exception.status_code, 422)
            self.assertEqual(len(backend.ingestion_ledger), 1)


if __name__ == "__main__":
    unittest.main()
