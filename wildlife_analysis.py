"""Analysis of actual media; no preset or random species results."""
import io
import threading
import time
from model_loading import load_model, model_resource

import numpy as np
from PIL import Image, ImageDraw, ImageOps

MODEL_ID = "google/owlvit-base-patch32"
DEFAULT_ANIMALS = (
    "person, human, lion, tiger, leopard, cheetah, jaguar, elephant, rhinoceros, hippopotamus, "
    "giraffe, zebra, deer, antelope, buffalo, bison, bear, wolf, fox, hyena, "
    "wild boar, monkey, gorilla, chimpanzee, orangutan, dog, cat, horse, cow, "
    "goat, sheep, camel, rabbit, squirrel, otter, crocodile, snake, turtle, "
    "lizard, frog, bird, peacock, eagle, owl, duck, penguin, fish, shark, "
    "dolphin, whale, butterfly, bee, spider"
)


def generate_picture_name(detections, base_prefix="camera_photo", ext="jpg"):
    """Generate a clean, descriptive file name based on detected subjects."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if detections:
        seen = []
        for d in sorted(detections, key=lambda x: x.get("score", 0), reverse=True):
            clean = d.get("animal", "").lower().replace(" ", "_").strip()
            if clean and clean not in seen:
                seen.append(clean)
        if seen:
            species_prefix = "_".join(seen[:3])
            return f"{species_prefix}_{timestamp}.{ext}"
    return f"{base_prefix}_{timestamp}.{ext}"


def warmup_models():
    """Pre-load models at startup to avoid delay during first use."""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load OWL-ViT detector
        get_detector()
        # Load CLAP audio model
        from animal_audio import get_audio_matcher
        get_audio_matcher()
        return True, f"Models warmed up on {device}"
    except Exception as e:
        return False, f"Model warmup failed: {e}"


def animal_labels(text):
    labels = list(dict.fromkeys(s.strip().lower() for s in text.split(",") if s.strip()))
    if not labels or len(labels) > 120 or any(len(s) > 60 for s in labels):
        raise ValueError("Enter between 1 and 120 animal/subject names, each at most 60 characters.")
    return labels


def decode_audio(data):
    import soundfile as sf

    if len(data) > 50 * 1024 * 1024:
        raise ValueError("Audio must be smaller than 50 MB.")
    with sf.SoundFile(io.BytesIO(data)) as source:
        sr = source.samplerate
        if source.frames < 64 or source.frames / sr > 300:
            raise ValueError("Use a recording between 64 samples and 5 minutes long.")
        samples = source.read(dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if not np.isfinite(mono).all():
        raise ValueError("The recording contains invalid sample values.")
    return mono, sr, samples.shape[1]


def analyze_audio(data):
    from scipy.signal import stft

    mono, sr, channels = decode_audio(data)
    nperseg = min(1024, len(mono))
    # Limit plot memory while retaining time coverage of the entire recording.
    hop = max(nperseg // 4, int(np.ceil(len(mono) / 1500)))
    if hop >= nperseg:
        nperseg = min(8192, max(nperseg, hop))
    frequencies, times, spectrum = stft(
        mono, fs=sr, nperseg=nperseg, noverlap=max(0, nperseg - hop),
        boundary=None, padded=False,
    )
    magnitude = np.abs(spectrum)
    db = 20 * np.log10(np.maximum(magnitude, 1e-6))
    rms = float(np.sqrt(np.mean(mono ** 2)))
    energy = np.sum(magnitude, axis=1)
    aci = float(np.sum(np.divide(
        np.sum(np.abs(np.diff(magnitude, axis=1)), axis=1), energy,
        out=np.zeros_like(energy), where=energy > 1e-8,
    )))
    return {
        "sample_rate": sr, "duration_seconds": len(mono) / sr,
        "channels": channels, "rms": rms,
        "peak": float(np.max(np.abs(mono))), "aci": aci,
        "dominant_frequency_hz": float(frequencies[np.argmax(energy)]) if rms > 1e-6 else None,
        "frequencies": frequencies, "times": times, "db": db,
    }


@model_resource
def get_detector():
    from transformers import OwlViTForObjectDetection, OwlViTProcessor
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = OwlViTProcessor.from_pretrained(MODEL_ID)
    model = load_model(OwlViTForObjectDetection, MODEL_ID, device)

    def detector(image, candidate_labels, threshold):
        # All names share one vision pass. The generic pipeline repeats that
        # expensive pass once per name, which makes a live feed impractical.
        inputs = processor(text=[candidate_labels], images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs, threshold=threshold,
            target_sizes=torch.tensor([[image.height, image.width]], device=device),
        )[0]
        return [
            {"label": candidate_labels[int(label)], "score": float(score),
             "box": dict(zip(("xmin", "ymin", "xmax", "ymax"), box.tolist()))}
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"])
        ]

    return detector, threading.Lock()


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def detect_animals(image, labels, threshold=0.15):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((1280, 1280))
    detector, lock = get_detector()
    prompts = [f"a photo of a {label}" for label in labels]
    names = dict(zip(prompts, labels))
    started = time.perf_counter()
    with lock:
        predictions = detector(image, candidate_labels=prompts, threshold=threshold)
    detections = []
    # Suppress alternate species labels for the same box.
    for prediction in sorted(predictions, key=lambda p: p["score"], reverse=True):
        box = prediction["box"]
        coords = [max(0, min(image.width if i % 2 == 0 else image.height, int(box[k])))
                  for i, k in enumerate(("xmin", "ymin", "xmax", "ymax"))]
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        if any(overlap(coords, d["box"]) > 0.45 for d in detections):
            continue
        detections.append({"animal": names.get(prediction["label"], prediction["label"]),
                           "score": float(prediction["score"]), "box": coords})
    draw = ImageDraw.Draw(image)
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        draw.rectangle((x1, y1, x2, y2), outline="#22c55e", width=3)
        label = f"{item['animal']} ({item['score']:.2f})"
        text_box = draw.textbbox((x1, max(0, y1 - 16)), label)
        draw.rectangle(text_box, fill="black")
        draw.text((x1, max(0, y1 - 16)), label, fill="white")
    return image, detections, time.perf_counter() - started


def make_report(source, observations, frames_analyzed, threshold, labels, **extra):
    from datetime import datetime, timezone

    summary = {}
    for observation in observations:
        for detection in observation["detections"]:
            item = summary.setdefault(detection["animal"], {
                "animal": detection["animal"], "max_score": 0.0, "observations": 0,
                "first_seen_seconds": observation["seconds"], "last_seen_seconds": observation["seconds"],
            })
            item["max_score"] = max(item["max_score"], detection["score"])
            item["observations"] += 1
            item["last_seen_seconds"] = observation["seconds"]
    return {
        "source": source, "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID, "threshold": threshold, "candidate_animals": labels,
        "frames_analyzed": frames_analyzed, "animals": list(summary.values()),
        "observations": observations,
        "interpretation": "Model predictions require review. Scores are not calibrated probabilities. "
                          "Repeated observations do not count unique animals. No match does not prove absence.",
        **extra,
    }
