# ECO-MRV local setup

From this folder on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. For the other dashboard API features, run this in a second terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend:app --port 8000 --reload
```

## Wildlife analysis

## Multi-modal evidence workflow

The Reports page now preserves the existing camera, audio and satellite evidence
and derives a traceable project-specific workflow:

```text
camera / video (OWL-ViT) + audio (CLAP) + satellite measurements
→ structured raw features + normalized evidence
→ missing-data-aware weighted fusion
→ agreement + historical anomaly assessment
→ ecosystem evidence score + Proposed Multi-Modal Ecosystem Health Score (MEHS)
→ human-reviewed report
```

The initial configurable weights are camera 0.30, audio 0.30 and satellite
0.40 in `ecomrv_config.py`. If a modality is missing, its weight is
renormalized across the available modalities; missing data is never assumed to
be zero. MEHS is a project-specific evidence indicator, not a certified or
validated ecological metric. An anomaly result is only available when at least
three historical observations are supplied to the processing layer.

An ordinary RGB satellite image remains evidence of an uploaded file only. It
is excluded from numerical fusion until a calibrated spatial vegetation
measurement and suitable metadata are available.

Use the sidebar to open **Bioacoustics** or **Live Camera & Video**. They are separate pages; only the selected page runs. The existing satellite, map, metrics and reports tabs are under **Forest Dashboard**.

**Reports**, directly below **Live Camera & Video**, combines user-provided audio, image, video and webcam evidence. Upload and analyze media on the respective pages, then select which evidence to include in Reports. Supply project, zone, monitoring date and observation notes. The page shows evidence cards, animal findings and timestamps, uncertain/background events, cross-source label comparisons, and a final summary. Download Markdown for reading or JSON with full analysis evidence. No inference runs when opening Reports. Multiple media items are retained in session memory; reanalyzing the same media replaces its saved analysis. Webcam entries are snapshots of the latest available 300 analyzed frames for each camera session.

Report notes and project metadata are user supplied. Matching animal labels across uploads do not prove the same event or animal; recordings are not time/location synchronized. The report does not manufacture ecological health scores, calibrated confidence, environmental risk or satellite findings. Pending analysis is distinguished from an analyzed source with no matches. The existing demonstration reports under Forest Dashboard are not included.

Satellite images can also be included: upload PNG/JPEG/TIFF in **Reports → Add satellite image** or the Forest Dashboard's satellite upload section. The report includes a preview, image dimensions, format, file size and SHA-256 identity. Satellite inputs are kept separate from animal detections. The former fixed vegetation/carbon figures in the upload flow have been removed. Real vegetation indices and change estimates are not computed; they require a calibrated spectral-band pipeline and aligned observations. Existing satellite demonstration maps and the simulated backend endpoint do not supply report measurements.

Switching pages in the same browser session preserves uploaded media, source selections, detection settings and completed reports. A restored upload is labeled with its filename and can be cleared. Analysis only starts when requested; returning to a page does not rerun completed inference. Live camera access must be enabled again after leaving its page. Work is held in browser-session memory, not permanent storage.

- **Audio & spectrogram:** upload WAV/MP3/FLAC/OGG or record a microphone clip. Playback and an interactive STFT spectrogram use the actual samples. Audio measurements can be downloaded. Limit: 50 MB, 5 minutes. Click **Identify animal sounds** below the plot to see possible animals, call descriptions, scores, timestamps, replay of the strongest match, and a downloadable detailed JSON report.
- Audio identification uses pretrained `laion/clap-htsat-unfused`, downloaded and cached on first use (approximately 615 MB). No API key is needed. It compares 26 animal-call descriptions and 11 background descriptions using the actual waveform, resampled to 48 kHz. Every non-overlapping 5-second window, including the final partial window, is inspected. Speech, music, rain and other non-animal sounds compete with the animal labels. Quiet windows skip model inference.
- Audio matches are **possible animal calls**, not confirmed species. A match must exceed a heuristic cosine similarity of 0.25 and beat the runner-up by at least 0.03; these thresholds are not calibrated confidence or accuracy estimates. Similar calls such as lion/tiger or sheep/goat can be ambiguous. Each window reports only its strongest match, so simultaneous or quiet animals may be missed. Window timestamps are not exact call boundaries, and repeated matching windows do not count individual animals. The timeline includes uncertain and background results so that rejection is visible.
- **Animal camera & video:** upload an image, capture a camera photo, upload video, or enable the live browser camera. Click **Analyze animals** for uploads. For live video, enable the detector, click **START**, and grant camera access. Use **STOP** to release the camera.
- Detection uses pretrained Google OWL-ViT with editable animal names, including lion and tiger. The first use downloads model weights (roughly 600 MB) from Hugging Face; subsequent runs reuse the local cache. No API key is required. CPU processing may take seconds per frame. A GPU improves speed. Live inference is sampled at most once per second, subject to processing speed.
- Reports appear below the analyzed image/video or camera with species predictions, scores, timestamps, and downloadable JSON. Video analysis samples up to 120 frames; its report states the covered duration. Detections between sampled frames can be missed. Live reports retain the last 300 analyzed frames, and repeated observations do not count unique animals.
- Detection scores are not calibrated accuracy percentages. The model only searches the supplied names; it cannot identify every animal or guarantee a correct species. Review predictions, especially similar species, obscured subjects and poor lighting. No match does not establish that no animal is present.
- Local camera access uses the browser's webcam on localhost. Remote deployment requires HTTPS and may require WebRTC STUN/TURN configuration. No real forest camera feeds, thermal hardware, or remote PTZ controls are connected.

The audio endpoint `/api/process/audio` returns measured duration, sample rate, RMS amplitude, dominant frequency and ACI. This endpoint remains measurements-only with an empty species list; animal-call identification runs in the dashboard audio panel through `animal_audio.py`. Simulated confidence and biotic-ratio fields were removed. Other dashboard sections retain their existing demonstration data.

## Checks

If an existing Streamlit process reports `Cannot copy out of meta tensor`, stop it with Ctrl+C and restart using the command above. A browser refresh alone does not clear old model-loading state. The current loader initializes on CPU, validates and loads the complete checkpoint, and then moves the model to the selected device. Visual and audio model initialization is serialized. It does not use `to_empty()` to bypass missing trained weights.

```powershell
.\.venv\Scripts\python.exe -m unittest test_wildlife test_animal_audio test_pages test_wildlife_reporting -v
```

The multimodal scoring checks can also be run with:

```powershell
.\.venv\Scripts\python.exe -m unittest test_ecosystem_pipeline -v
```

These tests check actual audio processing, resampling and time coverage, rejection of silent/ambiguous/background sounds, report display and replay, result invalidation for changed uploads, video decoding, detection postprocessing, reports and live-frame callbacks. Mocked model tests do not measure model accuracy or browser camera permissions.

Real-model smoke checks used four public [ESC-50 recordings](https://github.com/karolpiczak/ESC-50): `1-100032-A-0.wav` matched Dog, `1-26806-A-1.wav` matched Rooster, and `1-17367-A-10.wav` matched Rain without reporting an animal. `1-34094-A-5.wav` ranked Cat first but was marked uncertain because Fox scored within the separation threshold. These examples verify operation, not general species-identification accuracy.

Implementation references: [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit), [CLAP](https://huggingface.co/docs/transformers/model_doc/clap), [Streamlit WebRTC](https://github.com/whitphx/streamlit-webrtc).


".\mrv\.venv\Scripts\python.exe" -m streamlit run ".\mrv\app.py"