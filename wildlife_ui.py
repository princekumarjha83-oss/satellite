"""Streamlit audio, image, video and browser webcam workflows."""
import hashlib
import io
import json
import os
import tempfile
import threading
import time
import uuid

import streamlit as st
from PIL import Image
from wildlife_reporting import save_evidence
from ui_state import saved_control

from wildlife_analysis import (
    DEFAULT_ANIMALS, MODEL_ID, analyze_audio, animal_labels, detect_animals,
    generate_picture_name, get_detector, make_report,
)


def show_report(report, key):
    st.subheader("Animal detection report")
    if report["animals"]:
        for animal in report["animals"]:
            st.success(f"{animal['animal'].capitalize()} detected — highest model score {animal['max_score']:.2f}.")
        st.dataframe(report["animals"], hide_index=True)
    else:
        st.info("No animal matched the selected names above the detection threshold in the analyzed frames.")
    st.caption(f"{report['frames_analyzed']} frame(s) analyzed. {report['interpretation']}")
    if report.get("coverage"):
        st.caption(report["coverage"])
    st.download_button("Download detection report (JSON)", json.dumps(report, indent=2),
                       file_name="animal_detection_report.json", mime="application/json", key=key)


def saved_media(widget, label, key, **kwargs):
    saved_key = f"saved_media_{key}"

    def changed():
        # Removing a file explicitly also removes the saved copy.
        st.session_state[saved_key] = st.session_state.get(key)

    media = widget(label, key=key, on_change=changed, **kwargs)
    if media:
        st.session_state[saved_key] = media
    elif st.session_state.get(saved_key):
        media = st.session_state[saved_key]
        name = media.name if hasattr(media, "name") else (media[0].name if isinstance(media, list) and media else "media")
        st.caption(f"Restored from this session: {name}")
        if st.button("Clear saved media", key=f"clear_{key}"):
            st.session_state.pop(saved_key, None)
            st.rerun()
    return media


def render_audio():
    st.subheader("Audio playback and spectrogram")
    st.caption("Record your microphone or upload audio. The spectrogram is computed from that recording's samples.")
    mode = saved_control(st.radio, "audio_source", "Audio source", ["Upload recording", "Record microphone"], horizontal=True)
    clip = (saved_media(st.audio_input, "Record audio", key="wildlife_mic") if mode == "Record microphone"
            else saved_media(st.file_uploader, "Upload audio (up to 50 MB / 5 minutes)",
                                  type=["wav", "mp3", "flac", "ogg"], key="wildlife_audio"))
    if clip is None:
        st.info("Provide a recording to hear the audio and see its frequency content.")
        return
    data = clip.getvalue()
    st.audio(data)
    try:
        result = cached_audio(data)
        import plotly.graph_objects as go
        fig = go.Figure(go.Heatmap(x=result["times"], y=result["frequencies"], z=result["db"],
                                 colorscale="Viridis", zmin=-100, zmax=0,
                                 colorbar={"title": "dB"}))
        fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)", height=420)
        st.plotly_chart(fig, use_container_width=True)
        cols = st.columns(3)
        cols[0].metric("Duration", f"{result['duration_seconds']:.2f} s")
        cols[1].metric("Sample rate", f"{result['sample_rate']} Hz")
        cols[2].metric("RMS amplitude", f"{result['rms']:.4f}")
        if result["rms"] < 1e-6:
            st.warning("This recording is silent or almost silent.")
        st.caption("The plot shows measured acoustic energy. Use the animal-sound analysis below to look for possible animal calls.")
        metadata = {k: v for k, v in result.items() if k not in ("db", "times", "frequencies")}
        save_evidence(st.session_state, "audio", hashlib.sha256(data).hexdigest(), clip.name, measurements=metadata)
        st.download_button("Download audio measurements", json.dumps(metadata, indent=2),
                           file_name="audio_report.json", mime="application/json")
    except Exception as exc:
        st.error(f"Cannot analyze this recording: {exc}. Try a valid WAV, FLAC, OGG or MP3 file.")
        return
    render_audio_identification(data, clip.name)


def render_audio_identification(data, filename):
    from animal_audio import identify_audio_animals, AUDIO_MODEL_ID

    st.subheader("Animals identified from audio")
    st.caption("Look for possible animal calls throughout this recording. The first analysis downloads an audio model; longer recordings take more time.")
    fingerprint = hashlib.sha256(data).hexdigest()
    if st.button("Identify animal sounds", type="primary", key="identify_audio_animals"):
        st.session_state.pop("audio_animal_result", None)
        progress = st.progress(0, text="Loading the audio model and preparing the recording...")
        try:
            def update(done, total):
                progress.progress(done / total, text=f"Analyzed audio section {done} of {total}")
            with st.spinner("Comparing the recording with animal calls and background sounds..."):
                report = identify_audio_animals(data, progress=update)
            report["source"] = filename
            st.session_state["audio_animal_result"] = (fingerprint, report)
            save_evidence(st.session_state, "audio", fingerprint, filename, analysis=report)
        except Exception as exc:
            st.error(f"Animal-sound analysis could not finish: {exc}")
            st.info("The first run needs internet access to download the audio model. Your spectrogram remains available; no animal result was generated.")
        finally:
            progress.empty()
    stored = st.session_state.get("audio_animal_result")
    if stored is None or stored[0] != fingerprint:
        st.info("Click Identify animal sounds to populate the animal list and detailed report.")
        return
    report = stored[1]
    summary_col, detail_col = st.columns([1, 1.2])
    with summary_col:
        st.markdown("**Possible animals**")
        if report["animals"]:
            rows = [{"Animal": a["animal"], "Call matched": a["call_description"],
                     "Similarity score": round(a["max_similarity"], 3),
                     "Audio sections": a["windows_matched"]} for a in report["animals"]]
            st.dataframe(rows, hide_index=True)
            st.caption("These are model suggestions for review. Similarity is not a confidence percentage or a count of individual animals.")
        else:
            st.info("No clear animal match. The recording may contain background noise, ambiguous calls, or animals outside this model's comparison list.")
    with detail_col:
        st.markdown("**Detection details and playback**")
        if report["animals"]:
            selected = st.selectbox("Inspect an animal match", [a["animal"] for a in report["animals"]],
                                    key=f"audio_animal_{fingerprint[:12]}")
            animal = next(a for a in report["animals"] if a["animal"] == selected)
            st.write(f"Possible **{selected}** call: **{animal['call_description']}**.")
            st.write("Matched audio sections: " + "; ".join(f"{start:.2f}–{end:.2f} s" for start, end in animal["intervals"]))
            matches = [w for w in report["windows"] if w["animal"] == selected]
            best = max(matches, key=lambda w: w["similarity"])
            st.write(f"Strongest match: {best['start_seconds']:.2f}–{best['end_seconds']:.2f} s. "
                     f"Similarity {best['similarity']:.3f}; separation from next match {best['margin']:.3f}.")
            st.caption("Replay the strongest matching section. The call description is the model's comparison label, not a verified behavioral observation.")
            st.audio(data, start_time=best["start_seconds"], end_time=best["end_seconds"])
            st.dataframe(best["alternatives"], hide_index=True)
        else:
            st.write("Expand the timeline below to inspect the closest sounds and why no animal was reported.")
    with st.expander("Full audio timeline — including uncertain and background sounds", expanded=not bool(report["animals"])):
        timeline = [{"Start (s)": w["start_seconds"], "End (s)": w["end_seconds"],
                     "Result": w["status"].replace("_", " "), "Closest sound": w["closest_match"],
                     "Similarity": round(w["similarity"], 3) if w["similarity"] is not None else None,
                     "Other comparisons": "; ".join(f"{a['sound']} ({a['similarity']:.3f})" for a in w["alternatives"][1:]),
                     "Explanation": w["reason"]} for w in report["windows"]]
        st.dataframe(timeline, hide_index=True)
    st.caption(report["interpretation"])
    st.caption(f"Model: {AUDIO_MODEL_ID}. Compared animal calls: {', '.join(report['candidate_animals'])}.")
    st.download_button("Download detailed animal audio report", json.dumps(report, indent=2),
                       file_name="animal_audio_report.json", mime="application/json", key="audio_animal_download")


@st.cache_data(max_entries=4, show_spinner="Calculating spectrogram...")
def cached_audio(data):
    return analyze_audio(data)


def ready_detector():
    try:
        with st.spinner("Loading animal detector (first use downloads model weights)..."):
            get_detector()
        return True
    except Exception as exc:
        st.error(f"Animal detector could not load: {exc}")
        if "meta" in str(exc).lower():
            st.info("Stop the Streamlit terminal with Ctrl+C, then run: .\\.venv\\Scripts\\python.exe -m streamlit run app.py. Refreshing the browser does not reset model initialization state.")
        else:
            st.info("Install requirements.txt and allow internet access for the first model download. No detection result was generated.")
        return False


def render_visual():
    st.subheader("Animal detection — image, video and live camera")
    st.caption("Analyze your own media. No remote forest cameras are connected.")
    names = saved_control(st.text_area, "visual_animal_names", "Animal names to look for (comma separated)", DEFAULT_ANIMALS,
                         help="Add a species such as lion or red panda. This model searches these names; it cannot guarantee recognition of every animal.")
    threshold = saved_control(st.slider, "visual_threshold", "Detection score threshold", 0.05, 0.90, 0.15, 0.05)
    st.caption(f"Model: {MODEL_ID}. Scores are model matching scores, not accuracy percentages. CPU inference may take several seconds per frame.")
    try:
        labels = animal_labels(names)
    except ValueError as exc:
        st.warning(str(exc))
        return
    mode = saved_control(st.radio, "visual_source", "Visual source", ["Image", "Video", "Camera photo", "Live camera"], horizontal=True)
    if mode == "Live camera":
        render_live(labels, threshold)
        return

    # ── Image mode: supports multiple uploads ──────────────────────────────
    if mode == "Image":
        media_list = saved_media(
            st.file_uploader,
            "Upload image(s) — select multiple for batch detection",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="wildlife_Image",
        )
        if not media_list:
            st.info("Upload one or more images to detect animals.")
            return
        if not isinstance(media_list, list):
            media_list = [media_list]

        # Validate and preview all uploaded images
        valid_files = []
        num_cols = min(len(media_list), 3)
        preview_cols = st.columns(num_cols)
        for i, media in enumerate(media_list):
            try:
                raw = media.getvalue()
                with Image.open(io.BytesIO(raw)) as source:
                    source.verify()
                preview_cols[i % num_cols].image(raw, caption=media.name, use_container_width=True)
                valid_files.append(media)
            except Exception as exc:
                st.error(f"Cannot open {media.name}: {exc}")

        if not valid_files:
            return

        # Fingerprint covers ALL images + current settings — ensures
        # stale results from a different upload set are never shown.
        combined_bytes = b"".join(f.getvalue() for f in valid_files)
        fingerprint = hashlib.sha256(
            combined_bytes + repr((labels, threshold, mode)).encode()
        ).hexdigest()

        if st.button("Analyze animals", type="primary"):
            st.session_state.pop("wildlife_result", None)
            if ready_detector():
                observations = []
                annotated_previews = []
                total_elapsed = 0.0
                try:
                    for idx, media in enumerate(valid_files):
                        # Each image is opened and analysed from its own bytes —
                        # no data from another image can leak in.
                        img_data = media.getvalue()
                        img_hash = hashlib.sha256(img_data).hexdigest()
                        with st.spinner(f"Analyzing {media.name} ({idx + 1}/{len(valid_files)})..."):
                            with Image.open(io.BytesIO(img_data)) as src:
                                annotated, detections, elapsed = detect_animals(src, labels, threshold)
                        # Per-image report with THIS image's specific detections only
                        img_report = make_report(
                            media.name, [{"seconds": 0.0, "detections": detections}], 1,
                            threshold, labels, inference_seconds=elapsed,
                            media_type="image", media_sha256=img_hash,
                        )
                        observations.append({"seconds": float(idx), "detections": detections})
                        # Store as PNG bytes — PIL Image objects can fail session-state
                        # serialization when navigating between pages.
                        buf = io.BytesIO()
                        annotated.save(buf, format="PNG")
                        annotated_previews.append((media.name, buf.getvalue()))
                        total_elapsed += elapsed
                        # Save each image WITH its own analysis → Reports shows correct counts
                        save_evidence(st.session_state, "image", img_hash, media.name, analysis=img_report)

                    source_label = (
                        valid_files[0].name if len(valid_files) == 1
                        else f"{len(valid_files)} images"
                    )
                    # For a single image, include media_sha256 so import_session_evidence
                    # updates the existing record rather than creating a duplicate entry.
                    extra = {"media_type": "image"}
                    if len(valid_files) == 1:
                        extra["media_sha256"] = hashlib.sha256(valid_files[0].getvalue()).hexdigest()
                    report = make_report(
                        source_label, observations, len(valid_files),
                        threshold, labels, inference_seconds=total_elapsed, **extra,
                    )
                    st.session_state["wildlife_result"] = (fingerprint, report, annotated_previews)
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

        stored = st.session_state.get("wildlife_result")
        if stored and stored[0] == fingerprint:
            _, report, annotated_previews = stored
            if annotated_previews:
                if len(annotated_previews) == 1:
                    st.image(annotated_previews[0][1], caption=f"Analyzed: {annotated_previews[0][0]}")
                else:
                    st.subheader("Analyzed images")
                    res_cols = st.columns(min(len(annotated_previews), 3))
                    for i, (name, img) in enumerate(annotated_previews):
                        res_cols[i % 3].image(img, caption=f"Detected: {name}", use_container_width=True)
            show_report(report, "visual_report")
        return

    # ── Camera photo mode: auto-detect on capture, identify subject & give correct name ──
    if mode == "Camera photo":
        media = saved_media(st.camera_input, "Capture photo with camera", key="wildlife_camera_photo")
        if media is None:
            return
        data = media.getvalue()
        media_kind = "camera_photo"
        media_hash = hashlib.sha256(data).hexdigest()
        fingerprint = hashlib.sha256(data + repr((labels, threshold, mode)).encode()).hexdigest()

        stored = st.session_state.get("wildlife_result")
        # Automatically run detection on new capture or when settings change
        if stored is None or stored[0] != fingerprint:
            if ready_detector():
                try:
                    with st.spinner("Analyzing captured photo & identifying subject..."):
                        with Image.open(io.BytesIO(data)) as source:
                            preview, detections, elapsed = detect_animals(source, labels, threshold)
                    picture_name = generate_picture_name(detections, base_prefix="camera_photo")
                    st.session_state[f"pic_name_{media_hash}"] = picture_name
                    report = make_report(
                        picture_name, [{"seconds": 0.0, "detections": detections}],
                        1, threshold, labels, inference_seconds=elapsed,
                        media_type=media_kind, media_sha256=media_hash,
                    )
                    preview_bytes = None
                    if preview is not None:
                        buf = io.BytesIO()
                        preview.save(buf, format="JPEG", quality=95)
                        preview_bytes = buf.getvalue()
                    st.session_state["wildlife_result"] = (fingerprint, report, [(picture_name, preview_bytes)])
                    save_evidence(st.session_state, media_kind, media_hash, picture_name, analysis=report)
                    stored = st.session_state["wildlife_result"]
                except Exception as exc:
                    st.error(f"Detection failed: {exc}")

        stored = st.session_state.get("wildlife_result")
        if stored and stored[0] == fingerprint:
            _, report, previews = stored
            detected_animals = report.get("animals", [])
            current_name = st.session_state.get(f"pic_name_{media_hash}", report.get("source", "camera_photo.jpg"))

            # Display identified subject banner
            if detected_animals:
                det_summary = ", ".join(f"**{a['animal'].capitalize()}** ({a['max_score']:.0%})" for a in detected_animals)
                st.success(f"🎯 **Identified Subject:** {det_summary}")
            else:
                st.warning(f"⚠️ No matching subject found above threshold {threshold:.2f}.")
                st.info("💡 **Tip:** Lower the **Detection score threshold** slider above (e.g. to 0.10 or 0.15) if you want to detect lower confidence matches, or check the subject list.")

            # Name and download section
            col_name, col_dl = st.columns([2, 1])
            with col_name:
                user_name = st.text_input("🏷️ Picture Name (detected)", value=current_name, key=f"name_{media_hash[:8]}")
                if user_name != current_name:
                    st.session_state[f"pic_name_{media_hash}"] = user_name
                    current_name = user_name
                    report["source"] = user_name
                    if previews:
                        previews[0] = (user_name, previews[0][1])
                    save_evidence(st.session_state, media_kind, media_hash, user_name, analysis=report)
            with col_dl:
                dl_data = previews[0][1] if (previews and previews[0][1] is not None) else data
                st.download_button(
                    f"📥 Download '{current_name}'",
                    data=dl_data,
                    file_name=current_name,
                    mime="image/jpeg",
                    use_container_width=True,
                    key=f"dl_btn_{media_hash[:8]}",
                )

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                if st.button("🔄 Re-detect photo", key=f"redetect_{media_hash[:8]}"):
                    st.session_state.pop("wildlife_result", None)
                    st.rerun()

            if previews and previews[0][1] is not None:
                st.image(previews[0][1], caption=f"Analyzed photo: {current_name}", use_container_width=True)
            else:
                st.image(data, caption=f"Captured photo: {current_name}", width=600)
            show_report(report, "visual_report")
        return

    # ── Video mode ────────────────────────────────────────────────────────
    media = saved_media(st.file_uploader, "Upload a video (up to 100 MB)",
                        type=["mp4", "mov", "avi", "mkv", "webm"],
                        key=f"wildlife_{mode}")
    if media is None:
        return
    data = media.getvalue()
    media_kind = "video"
    media_hash = hashlib.sha256(data).hexdigest()
    if len(data) > 100 * 1024 * 1024:
        st.error("Please upload a video smaller than 100 MB.")
        return
    st.video(data)
    interval = saved_control(st.slider, "visual_interval", "Seconds between analyzed frames", 0.5, 10.0, 1.0, 0.5)
    st.caption("Up to 120 frames are sampled. Reports state the analyzed coverage; brief appearances between samples may be missed.")
    fingerprint = hashlib.sha256(data + repr((labels, threshold, interval, mode)).encode()).hexdigest()
    save_evidence(st.session_state, media_kind, media_hash, media.name)
    if st.button("Analyze animals", type="primary"):
        st.session_state.pop("wildlife_result", None)
        if ready_detector():
            try:
                report, preview = analyze_video(data, media.name, labels, threshold, interval)
                report.update(media_type=media_kind, media_sha256=media_hash)
                preview_bytes = None
                if preview is not None:
                    buf = io.BytesIO()
                    preview.save(buf, format="PNG")
                    preview_bytes = buf.getvalue()
                st.session_state["wildlife_result"] = (fingerprint, report, [(media.name, preview_bytes)])
                save_evidence(st.session_state, media_kind, media_hash, media.name, analysis=report)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
    stored = st.session_state.get("wildlife_result")
    if stored and stored[0] == fingerprint:
        _, report, previews = stored
        if previews and previews[0][1] is not None:
            st.image(previews[0][1], caption="Analyzed frame with predicted animal locations")
        show_report(report, "visual_report")


def analyze_video(data, filename, labels, threshold, interval):
    import cv2
    path = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as tmp:
            path = tmp.name
            tmp.write(data)
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise ValueError("Video codec could not be opened. Try an MP4 encoded with H.264.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not (0 < fps < 1000 and count > 0):
            raise ValueError("Cannot determine video duration/frame rate.")
        duration = count / fps
        step = max(1, round(fps * interval))
        indices = list(range(0, int(count), step))[:120]
        observations, preview = [], None
        progress = st.progress(0, text="Analyzing video frames...")
        frame_slot = st.empty()
        for index, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not decode frame at {frame_index / fps:.2f} seconds. Analysis incomplete.")
            annotated, detections, _ = detect_animals(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), labels, threshold)
            observations.append({"seconds": round(frame_index / fps, 3), "detections": detections})
            if preview is None or detections:
                preview = annotated
            frame_slot.image(annotated, caption=f"Analyzed frame at {frame_index / fps:.2f}s")
            progress.progress((index + 1) / len(indices))
        if not observations:
            raise ValueError("No frames could be decoded.")
        report = make_report(filename, observations, len(observations), threshold, labels,
                             duration_seconds=duration, sample_interval_seconds=step / fps,
                             coverage=f"Sampled {len(observations)} frames from 0 to {observations[-1]['seconds']:.2f}s of a {duration:.2f}s video. Maximum 120 samples.")
        # Convert preview PIL Image to bytes for safe session-state serialization
        preview_bytes = None
        if preview is not None:
            buf = io.BytesIO()
            preview.save(buf, format="PNG")
            preview_bytes = buf.getvalue()
        return report, preview_bytes
    finally:
        if capture is not None:
            capture.release()
        if path is not None:
            os.unlink(path)


class LiveProcessor:
    def __init__(self, labels, threshold):
        self.labels, self.threshold = labels, threshold
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.session_id = uuid.uuid4().hex
        self.last_analysis = 0.0
        self.observations = []
        self.frames = 0
        self.error = None

    async def recv_queued(self, frames):
        # Drop queued old frames so analysis follows the current scene.
        return [self.recv(frames[-1])]

    def recv(self, frame):
        import av
        now = time.monotonic()
        if now - self.last_analysis < 1.0:
            return frame
        self.last_analysis = now
        try:
            annotated, detections, _ = detect_animals(frame.to_image(), self.labels, self.threshold)
            with self.lock:
                self.frames += 1
                self.observations.append({"seconds": round(now - self.started, 3), "detections": detections})
                self.observations = self.observations[-300:]
                self.error = None
            return av.VideoFrame.from_image(annotated)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
            return frame


def render_live(labels, threshold):
    st.caption("Press START and allow browser camera access. Live inference is sampled at most once per second and may be slower on CPU. Use localhost or HTTPS.")
    if not st.toggle("Enable live animal detector"):
        return
    if not ready_detector():
        return
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        config_key = hashlib.sha256(repr((labels, threshold)).encode()).hexdigest()[:12]
        context = webrtc_streamer(
            key=f"animal_live_{config_key}", mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
            video_processor_factory=lambda: LiveProcessor(labels, threshold),
            async_processing=True,
        )
        @st.fragment(run_every=2)
        def live_report():
            processor = context.video_processor
            if processor is not None:
                with processor.lock:
                    observations = list(processor.observations)
                    error = processor.error
                    frames = processor.frames
                if error:
                    st.error(f"Live inference error: {error}")
                if observations:
                    st.session_state["live_animal_report"] = make_report(
                        "Browser webcam", observations, frames, threshold, labels,
                        session_id=processor.session_id,
                        coverage="Report retains the latest 300 analyzed frames. Counts are observations, not unique animals.")
                    save_evidence(st.session_state, "live_camera", processor.session_id, "Browser webcam",
                                  analysis=st.session_state["live_animal_report"])
            report = st.session_state.get("live_animal_report")
            if report and report["candidate_animals"] == labels and report["threshold"] == threshold:
                show_report(report, "live_report_download")
            else:
                st.info("Waiting for camera frames. The report updates while the camera is running.")
        live_report()
    except ImportError:
        st.error("Install streamlit-webrtc from requirements.txt to enable the live camera.")
