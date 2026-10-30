"""Windowed animal-call matching with a pretrained audio/text model.

Results are possible sound matches, not verified species inventories.
"""
import hashlib
import math
import threading
from datetime import datetime, timezone
from model_loading import load_model, model_resource

import numpy as np

from wildlife_analysis import decode_audio

AUDIO_MODEL_ID = "laion/clap-htsat-unfused"
MIN_SIMILARITY = 0.25
MIN_MARGIN = 0.03
WINDOW_SECONDS = 5.0

# These are comparison descriptions, never preassigned detections.
ANIMAL_SOUNDS = [
    ("Dog", "barking", "A dog barking."),
    ("Cat", "meowing", "A cat meowing."),
    ("Lion", "roaring", "A lion roaring."),
    ("Tiger", "roaring and growling", "A tiger roaring and growling."),
    ("Elephant", "trumpeting", "An elephant trumpeting."),
    ("Wolf", "howling", "A wolf howling."),
    ("Fox", "yelping", "A fox screaming and yelping."),
    ("Hyena", "laughing calls", "A hyena laughing."),
    ("Monkey", "chattering and calling", "Monkeys chattering and calling."),
    ("Horse", "neighing", "A horse neighing."),
    ("Cow", "mooing", "A cow mooing."),
    ("Sheep", "bleating", "A sheep bleating."),
    ("Goat", "bleating", "A goat bleating."),
    ("Pig", "oinking and grunting", "A pig oinking and grunting."),
    ("Bird (unspecified)", "chirping and singing", "Birds chirping and singing."),
    ("Owl", "hooting", "An owl hooting."),
    ("Crow", "cawing", "A crow cawing."),
    ("Duck", "quacking", "Ducks quacking."),
    ("Goose", "honking", "Geese honking."),
    ("Rooster", "crowing", "A rooster crowing."),
    ("Chicken", "clucking", "Chickens clucking."),
    ("Peacock", "calling", "A peacock calling."),
    ("Frog", "croaking", "Frogs croaking."),
    ("Cricket", "chirping", "Crickets chirping."),
    ("Cicada", "buzzing", "Cicadas buzzing."),
    ("Bee", "buzzing", "Bees buzzing."),
]
BACKGROUNDS = [
    ("Human speech", "People talking."),
    ("Music", "Music playing."),
    ("Rain", "Rain falling."),
    ("Wind", "Wind blowing into a microphone."),
    ("Flowing water", "Water flowing in a stream."),
    ("Vehicle", "A vehicle engine running."),
    ("Footsteps", "Footsteps on the ground."),
    ("Mechanical noise", "A machine making noise."),
    ("Static", "White noise and microphone static."),
    ("Electronic tone", "An electronic beep or sine wave tone."),
    ("Silence", "Silence with faint background noise."),
]


@model_resource
def get_audio_matcher():
    import torch
    from transformers import ClapModel, ClapProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ClapProcessor.from_pretrained(AUDIO_MODEL_ID)
    model = load_model(ClapModel, AUDIO_MODEL_ID, device)
    descriptions = [a[2] for a in ANIMAL_SOUNDS] + [b[1] for b in BACKGROUNDS]
    text = processor(text=descriptions, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        text_features = model.get_text_features(**text)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

    def match(samples):
        features = processor(audio=samples, sampling_rate=48000, return_tensors="pt").to(device)
        with torch.inference_mode():
            audio_features = model.get_audio_features(**features)
            audio_features = torch.nn.functional.normalize(audio_features, dim=-1)
            scores = audio_features @ text_features.T
        return scores[0].cpu().numpy()

    return match, threading.Lock()


def decide_match(scores):
    """Reject weak, ambiguous, and non-animal matches rather than force a name."""
    scores = np.asarray(scores, dtype=float)
    names = [a[0] for a in ANIMAL_SOUNDS] + [b[0] for b in BACKGROUNDS]
    if scores.shape != (len(names),) or not np.isfinite(scores).all():
        raise ValueError("Audio model returned invalid scores.")
    ranking = np.argsort(-scores)
    best, second = int(ranking[0]), int(ranking[1])
    margin = float(scores[best] - scores[second])
    is_animal = best < len(ANIMAL_SOUNDS)
    if scores[best] < MIN_SIMILARITY:
        status, reason = "uncertain", "No sound description met the minimum similarity."
    elif not is_animal:
        status, reason = "background", "A non-animal sound was the closest match."
    elif margin < MIN_MARGIN:
        status, reason = "uncertain", "Similar-scoring sound descriptions prevent a reliable distinction."
    else:
        status, reason = "possible_animal", "Animal description passed the similarity and separation thresholds."
    return {
        "status": status, "animal": names[best] if status == "possible_animal" else None,
        "call_description": ANIMAL_SOUNDS[best][1] if status == "possible_animal" else None,
        "closest_match": names[best], "similarity": float(scores[best]),
        "margin": margin, "reason": reason,
        "alternatives": [{"sound": names[int(i)], "similarity": float(scores[i])} for i in ranking[:3]],
    }


def identify_audio_animals(data, progress=None):
    from scipy.signal import resample_poly

    mono, sr, _ = decode_audio(data)
    # Resample the waveform, not just its metadata, to CLAP's required 48 kHz.
    divisor = math.gcd(sr, 48000)
    samples = resample_poly(mono, 48000 // divisor, sr // divisor) if sr != 48000 else mono
    size = int(WINDOW_SECONDS * 48000)
    windows = []
    total = math.ceil(len(samples) / size)
    matcher = None
    for index, offset in enumerate(range(0, len(samples), size)):
        segment = samples[offset:offset + size]
        start, end = offset / 48000, min((offset + len(segment)) / 48000, len(mono) / sr)
        rms = float(np.sqrt(np.mean(segment ** 2)))
        if rms < 1e-5:
            result = {"status": "silence", "animal": None, "call_description": None,
                      "closest_match": "Silence", "similarity": None, "margin": None,
                      "reason": "The window is silent or almost silent.", "alternatives": []}
        elif len(segment) < 4800:
            result = {"status": "uncertain", "animal": None, "call_description": None,
                      "closest_match": None, "similarity": None, "margin": None,
                      "reason": "Less than 0.1 seconds of audio; too short to classify.", "alternatives": []}
        else:
            if matcher is None:
                matcher, lock = get_audio_matcher()
            with lock:
                result = decide_match(matcher(segment))
        windows.append({"start_seconds": round(start, 3), "end_seconds": round(end, 3),
                        "rms": rms, **result})
        if progress:
            progress(index + 1, total)
    animals = {}
    for window in windows:
        if window["status"] != "possible_animal":
            continue
        animal = animals.setdefault(window["animal"], {
            "animal": window["animal"], "call_description": window["call_description"],
            "max_similarity": -1.0, "windows_matched": 0, "intervals": [],
        })
        animal["max_similarity"] = max(animal["max_similarity"], window["similarity"])
        animal["windows_matched"] += 1
        animal["intervals"].append([window["start_seconds"], window["end_seconds"]])
    return {
        "model": AUDIO_MODEL_ID, "audio_sha256": hashlib.sha256(data).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": len(mono) / sr,
        "window_seconds": WINDOW_SECONDS, "minimum_similarity": MIN_SIMILARITY,
        "minimum_margin": MIN_MARGIN, "animals": sorted(animals.values(), key=lambda a: -a["max_similarity"]),
        "windows": windows, "candidate_animals": [a[0] for a in ANIMAL_SOUNDS],
        "interpretation": "Possible animal-call matches, not confirmed species. Cosine similarity is not an accuracy "
                          "percentage. Thresholds are heuristic, not calibrated confidence. Only the strongest sound "
                          "per 5-second window is evaluated; overlapping or quiet calls may be missed. "
                          "Timestamps mark analyzed windows, not exact call boundaries. No match does not prove absence.",
    }
