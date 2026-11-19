from __future__ import annotations

import json
import os
import re
import subprocess
import shutil
import threading
import time
import uuid
import wave
import numpy as np
import cv2
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import imageio_ffmpeg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL

ROOT = Path(__file__).resolve().parent / "workspace"
ROOT.mkdir(parents=True, exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()
MODEL: WhisperModel | None = None

app = FastAPI(title="ClipForge Local AI Engine", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngestRequest(BaseModel):
    url: HttpUrl

class RenderRequest(BaseModel):
    clip_index: int
    aspect_ratio: str = "9:16"

def update(job_id: str, **values: Any) -> None:
    with LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values)

def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([FFMPEG, "-hide_banner", "-y", *args], capture_output=True, text=True)

def get_model() -> WhisperModel:
    global MODEL
    if MODEL is None:
        MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    return MODEL

def validate_source(raw: str) -> None:
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    allowed = ("youtube.com", "youtu.be", "twitch.tv", "vimeo.com")
    if not any(host == domain or host.endswith("." + domain) for domain in allowed):
        raise ValueError("For safety, this worker currently accepts YouTube, Twitch and Vimeo VOD links only.")
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only public HTTP(S) video links are supported.")

def download_video(job_id: str, url: str) -> Path:
    folder = ROOT / job_id
    folder.mkdir(parents=True, exist_ok=True)
    update(job_id, stage="Checking source", progress=3, message="Checking public VOD availability…")
    def progress_hook(data: dict[str, Any]) -> None:
        if data.get("status") == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
            downloaded = data.get("downloaded_bytes") or 0
            pct = int(downloaded / total * 55) if total else 15
            update(job_id, stage="Downloading VOD", progress=min(58, max(5, pct)), message="Downloading the authorised replay…")
        elif data.get("status") == "finished":
            update(job_id, stage="Preparing media", progress=60, message="Download complete. Preparing audio…")
    options = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "outtmpl": str(folder / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [progress_hook],
        "ffmpeg_location": str(Path(FFMPEG).parent),
        "socket_timeout": 30,
        "retries": 3,
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
        if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
            raise ValueError("This is a live/upcoming stream, not an archived VOD. Analyse it after the replay is available.")
        if info.get("duration") and info["duration"] > 6 * 60 * 60:
            raise ValueError("This VOD is longer than the current 6-hour local analysis limit.")
        ydl.download([url])
    files = [p for p in folder.glob("source.*") if p.is_file() and p.suffix.lower() not in (".part", ".ytdl")]
    if not files:
        raise RuntimeError("The video download did not produce a media file.")
    source = files[0]
    duration = info.get("duration") or 0
    if not duration:
        probe = run_ffmpeg(["-i", str(source), "-f", "null", "-"])
        match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr)
        if match:
            duration = int(match.group(1))*3600 + int(match.group(2))*60 + float(match.group(3))
    update(job_id, source=str(source), title=info.get("title") or source.stem, duration=duration)
    return source

def extract_audio(source: Path, folder: Path) -> Path:
    audio = folder / "analysis.wav"
    result = run_ffmpeg(["-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio)])
    if result.returncode != 0 or not audio.exists():
        raise RuntimeError("Could not extract audio: " + result.stderr[-1200:])
    return audio

def audio_energy(audio: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    try:
        with wave.open(str(audio), "rb") as wav:
            rate = wav.getframerate()
            width = wav.getsampwidth()
            channels = wav.getnchannels()
            per_second = rate * width * channels
            second = 0
            while True:
                raw = wav.readframes(rate)
                if not raw:
                    break
                if width == 2:
                    samples = np.frombuffer(raw, dtype=np.int16)
                    if samples.size:
                        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                        values[second] = min(1.0, rms / 32768.0)
                second += 1
    except Exception:
        return {}
    return values

HYPE = re.compile(r"\b(wow|no way|oh my god|let'?s go|insane|crazy|amazing|unbelievable|bro|wait|finally|winner|win|won|clutch|goal|fail|omg|seriously|really|why|how|secret|best|worst|never|first|last|actually|incredible|what happened|you won)\b", re.I)

def find_face_center(source: Path, start: float, end: float) -> tuple[float, float, int, int] | None:
    """Find a stable dominant face centre for a portrait/square crop."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        return None
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    centres: list[tuple[float, float, float]] = []
    try:
        for second in range(int(start), int(end), 2):
            capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(36, 36))
            if len(faces):
                x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                centres.append(((x + w / 2) / width, (y + h / 2) / height, float(w*h)))
    finally:
        capture.release()
    if not centres:
        return None
    # Weight larger, clearer faces more heavily while resisting single-frame jumps.
    total = sum(item[2] for item in centres)
    cx = sum(item[0] * item[2] for item in centres) / total
    cy = sum(item[1] * item[2] for item in centres) / total
    return (max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)), width, height)

def score_highlights(words: list[dict[str, Any]], duration: float, energy: dict[int, float]) -> list[dict[str, Any]]:
    if not words:
        return []
    window = 32.0 if duration > 90 else max(15.0, min(32.0, duration / 3))
    step = max(8.0, window * 0.48)
    candidates: list[dict[str, Any]] = []
    start = 0.0
    while start + 8 < duration:
        end = min(duration, start + window)
        selected = [w for w in words if w["start"] >= start and w["start"] <= end]
        if len(selected) >= 8:
            text = " ".join(w["text"] for w in selected).strip()
            hype = len(HYPE.findall(text))
            punctuation = len(re.findall(r"[!?]", text))
            wpm = len(selected) / max(1, (end - start) / 60)
            unique = len(set(w["text"].lower() for w in selected)) / max(1, len(selected))
            rms = [energy.get(int(t), 0) for t in range(int(start), int(end))]
            loud = sum(rms) / max(1, len(rms))
            density = min(1, wpm / 150)
            hook = min(1, (hype * 1.4 + punctuation * .5) / 7)
            energy_score = min(1, loud / .18)
            variety = min(1, unique / .72)
            score = round(min(99, max(1, density*24 + hook*34 + energy_score*20 + variety*12 + min(1, len(selected)/35)*10)))
            reasons = []
            if hype: reasons.append(f"{hype} high-interest phrases")
            if wpm > 115: reasons.append("fast, energetic speech")
            elif wpm > 65: reasons.append("steady conversation")
            if loud > .12: reasons.append("strong audio energy")
            if punctuation: reasons.append(f"{punctuation} expressive beats")
            if not reasons: reasons.append("clear continuous speech")
            title = re.split(r"[.!?]", text)[0][:72] or "Highlight moment"
            clip_start = max(0.0, selected[0]["start"] - 2.0)
            clip_end = min(duration, selected[-1]["end"] + 2.0)
            if clip_end - clip_start < 15.0:
                midpoint = (clip_start + clip_end) / 2
                clip_start = max(0.0, midpoint - 7.5)
                clip_end = min(duration, midpoint + 7.5)
            candidates.append({
                "start": round(clip_start, 2),
                "end": round(clip_end, 2),
                "score": score,
                "title": title,
                "reason": reasons,
                "transcript": text[:260],
            })
        start += step
    candidates.sort(key=lambda item: item["score"], reverse=True)
    chosen: list[dict[str, Any]] = []
    for candidate in candidates:
        overlap = False
        for prior in chosen:
            shared = max(0, min(candidate["end"], prior["end"]) - max(candidate["start"], prior["start"]))
            if shared / max(1, min(candidate["end"]-candidate["start"], prior["end"]-prior["start"])) > .45:
                overlap = True
                break
        if not overlap:
            candidate["id"] = str(len(chosen) + 1)
            chosen.append(candidate)
        if len(chosen) >= 12:
            break
    return chosen


def probe_duration(source: Path) -> float:
    probe = run_ffmpeg(["-i", str(source), "-f", "null", "-"])
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr)
    if not match:
        return 0.0
    return int(match.group(1))*3600 + int(match.group(2))*60 + float(match.group(3))

def analyse_live_segment(job_id: str, source: Path) -> None:
    job = JOBS.get(job_id)
    if not job:
        return
    folder = source.parent
    audio = folder / (source.stem + "_audio.wav")
    result = run_ffmpeg(["-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio)])
    if result.returncode != 0 or not audio.exists():
        raise RuntimeError("Could not extract audio from live segment.")
    model = get_model()
    segments, _ = model.transcribe(str(audio), word_timestamps=True, vad_filter=True, beam_size=3)
    words: list[dict[str, Any]] = []
    transcript: list[str] = []
    for segment in segments:
        transcript.append(segment.text.strip())
        if segment.words:
            for word in segment.words:
                words.append({"text": word.word.strip(), "start": round(float(word.start), 3), "end": round(float(word.end), 3)})
        else:
            words.append({"text": segment.text.strip(), "start": float(segment.start), "end": float(segment.end)})
    duration = probe_duration(source)
    if duration <= 0:
        duration = max((word["end"] for word in words), default=0)
    candidates = score_highlights(words, duration, audio_energy(audio))
    for candidate in candidates:
        candidate["source_path"] = str(source)
        candidate["words"] = words
        candidate["transcript"] = candidate.get("transcript", "")[:260]
    with LOCK:
        current = JOBS.get(job_id)
        if not current:
            return
        combined = list(current.get("clips", [])) + candidates
        combined.sort(key=lambda item: item.get("score", 0), reverse=True)
        # Retain the strongest 24 unique moments across the ongoing broadcast.
        kept: list[dict[str, Any]] = []
        for candidate in combined:
            duplicate = any(
                candidate.get("source_path") == prior.get("source_path")
                and max(0, min(candidate["end"], prior["end"]) - max(candidate["start"], prior["start"]))
                    / max(1, min(candidate["end"]-candidate["start"], prior["end"]-prior["start"])) > .45
                for prior in kept
            )
            if not duplicate:
                candidate["id"] = str(len(kept) + 1)
                kept.append(candidate)
            if len(kept) >= 24:
                break
        current["clips"] = kept
        current["live_segments_processed"] = int(current.get("live_segments_processed", 0)) + 1
        current["stage"] = f"Live AI · {current['live_segments_processed']} segments analysed"
        current["message"] = f"Found {len(kept)} ranked moments so far. Capture is still running."
        current["progress"] = 0

def run_live_capture(job_id: str, url: str) -> None:
    folder = ROOT / job_id
    folder.mkdir(parents=True, exist_ok=True)
    stop_event = JOBS[job_id]["stop_event"]
    streamlink = shutil.which("streamlink")
    if not streamlink:
        update(job_id, status="error", stage="Live capture unavailable", live_active=False,
               message="Streamlink was not installed. Restart the local engine after pip installs requirements.")
        return
    stream_proc = None
    ffmpeg_proc = None
    processed: set[str] = set()
    try:
        update(job_id, status="processing", live_active=True, stage="Connecting to live stream",
               message="Connecting to the active broadcast and opening a rolling local recording…",
               progress=0, title="Live broadcast")
        stream_proc = subprocess.Popen(
            [streamlink, "--stdout", "--retry-streams", "5", url, "best"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
        )
        if stream_proc.stdout is None:
            raise RuntimeError("Could not open the live stream output.")
        pattern = str(folder / "live_%05d.ts")
        ffmpeg_proc = subprocess.Popen(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-fflags", "+genpts",
             "-i", "pipe:0", "-map", "0", "-c", "copy", "-f", "segment",
             "-segment_time", "60", "-reset_timestamps", "1", "-segment_format", "mpegts", pattern],
            stdin=stream_proc.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        stream_proc.stdout.close()
        update(job_id, stage="Live capture running", message="Recording in 60-second chunks. AI scans each completed chunk.")
        while True:
            files = sorted(folder.glob("live_*.ts"))
            for segment_file in files:
                if str(segment_file) in processed:
                    continue
                # The newest segment may still be written; only analyse closed chunks.
                if ffmpeg_proc.poll() is None and segment_file == (files[-1] if files else None):
                    continue
                try:
                    if segment_file.stat().st_size < 128_000:
                        processed.add(str(segment_file))
                        continue
                    analyse_live_segment(job_id, segment_file)
                except Exception as segment_error:
                    update(job_id, message=f"Skipped one live segment: {segment_error}")
                processed.add(str(segment_file))
            if stop_event.is_set():
                if stream_proc.poll() is None:
                    stream_proc.terminate()
                try:
                    stream_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    stream_proc.kill()
                try:
                    ffmpeg_proc.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    ffmpeg_proc.kill()
                    ffmpeg_proc.wait(timeout=5)
                break
            if stream_proc.poll() is not None or ffmpeg_proc.poll() is not None:
                break
            stop_event.wait(4)
        # FFmpeg closes the final partial segment after the input ends.
        for segment_file in sorted(folder.glob("live_*.ts")):
            if str(segment_file) in processed:
                continue
            try:
                if segment_file.stat().st_size >= 128_000:
                    analyse_live_segment(job_id, segment_file)
            except Exception as segment_error:
                update(job_id, message=f"Final segment could not be analysed: {segment_error}")
            processed.add(str(segment_file))
        clips = JOBS[job_id].get("clips", [])
        if clips:
            update(job_id, status="ready", live_active=False, stage="Live session analysed",
                   progress=100, message=f"Capture finished. {len(clips)} ranked moments are ready to render.")
        else:
            update(job_id, status="error", live_active=False, stage="No highlights found",
                   message="The stream ended without enough clear speech to score clips. Check audio and try again.")
    except Exception as error:
        update(job_id, status="error", live_active=False, stage="Live capture error", message=str(error))
    finally:
        for process in (stream_proc, ffmpeg_proc):
            if process and process.poll() is None:
                process.terminate()
        update(job_id, live_active=False)

def analyse_job(job_id: str, url: str) -> None:
    try:
        source = download_video(job_id, url)
        folder = source.parent
        update(job_id, stage="Extracting audio", progress=63, message="Extracting audio for local speech recognition…")
        audio = extract_audio(source, folder)
        update(job_id, stage="Loading Whisper", progress=68, message="Loading Whisper base model on CPU (first run downloads it)…")
        model = get_model()
        update(job_id, stage="Transcribing", progress=72, message="Transcribing the VOD with word timestamps…")
        segments, _ = model.transcribe(str(audio), word_timestamps=True, vad_filter=True, beam_size=3)
        words: list[dict[str, Any]] = []
        full_text: list[str] = []
        for segment in segments:
            full_text.append(segment.text.strip())
            if segment.words:
                for word in segment.words:
                    words.append({"text": word.word.strip(), "start": round(float(word.start), 3), "end": round(float(word.end), 3)})
            else:
                words.append({"text": segment.text.strip(), "start": float(segment.start), "end": float(segment.end)})
        update(job_id, stage="Scoring highlights", progress=91, message="Scoring hooks, pace, audio energy and transcript variety…")
        energy = audio_energy(audio)
        candidates = score_highlights(words, float(JOBS[job_id].get("duration") or 0), energy)
        if not candidates:
            raise RuntimeError("No speech highlights were found. Try a VOD with clear spoken audio.")
        update(job_id, stage="Ready", progress=100, message=f"Found {len(candidates)} ranked moments.", transcript=" ".join(full_text), words=words, clips=candidates, status="ready")
    except Exception as error:
        update(job_id, stage="Error", status="error", message=str(error))

@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "engine": "ClipForge Local AI Engine", "ffmpeg": Path(FFMPEG).exists(), "model": "Whisper base (downloads on first analysis)", "active_jobs": sum(1 for j in JOBS.values() if j.get("status") == "processing")}

@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, str]:
    raw = str(request.url)
    try:
        validate_source(raw)
        with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "socket_timeout": 20}) as ydl:
            info = ydl.extract_info(raw, download=False)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not inspect this public video URL: {error}")
    live_status = info.get("live_status")
    is_live = bool(info.get("is_live")) or live_status == "is_live"
    if live_status == "is_upcoming":
        raise HTTPException(status_code=409, detail="This broadcast has not started yet. Paste its link again once it is live.")
    job_id = uuid.uuid4().hex[:12]
    with LOCK:
        JOBS[job_id] = {"id": job_id, "status": "processing", "stage": "Queued", "progress": 0,
                        "message": "Queued local analysis…", "clips": [], "created": time.time(),
                        "live_active": is_live, "live_segments_processed": 0, "stop_event": threading.Event()}
    if is_live:
        if urlparse(raw).hostname and any(x in (urlparse(raw).hostname or "") for x in ("youtube.com", "youtu.be", "twitch.tv")):
            thread = threading.Thread(target=run_live_capture, args=(job_id, raw), daemon=True)
            thread.start()
            return {"job_id": job_id, "message": "Active livestream capture and rolling AI analysis started.", "mode": "live"}
        update(job_id, status="error", live_active=False, stage="Unsupported live platform",
               message="Active capture currently supports public YouTube and Twitch live streams.")
        raise HTTPException(status_code=400, detail="Active live capture currently supports public YouTube and Twitch streams.")
    thread = threading.Thread(target=analyse_job, args=(job_id, raw), daemon=True)
    thread.start()
    return {"job_id": job_id, "message": "Local VOD analysis started.", "mode": "vod"}

@app.post("/jobs/{job_id}/stop")
def stop_live(job_id: str) -> dict[str, str]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not job.get("live_active"):
        return {"ok": True, "message": "This job is not actively recording."}
    job["stop_event"].set()
    update(job_id, stage="Stopping live capture", message="Closing the recording cleanly and analysing the final chunk…")
    return {"ok": True, "message": "Live capture is stopping. Final clips will be ranked shortly."}

@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found. The local worker may have restarted.")
    safe = {k: v for k, v in job.items() if k not in ("source", "words", "stop_event")}
    safe["clips"] = [{k: v for k, v in clip.items() if k not in ("words", "source_path")} for clip in job.get("clips", [])]
    return safe

@app.post("/jobs/{job_id}/render")
def render_clip(job_id: str, request: RenderRequest) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job or (job.get("status") != "ready" and not (job.get("live_active") and job.get("clips"))):
        raise HTTPException(status_code=409, detail="Wait for the first live highlight or finish the analysis first.")
    clips = job.get("clips", [])
    if request.clip_index < 0 or request.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip index not found.")
    if request.aspect_ratio not in ("9:16", "1:1", "16:9"):
        raise HTTPException(status_code=400, detail="Unsupported aspect ratio.")
    clip = clips[request.clip_index]
    source = Path(clip.get("source_path") or job["source"])
    folder = source.parent
    output = folder / f"clip_{request.clip_index+1}_{int(clip['start']*1000)}_{request.aspect_ratio.replace(':','x')}.mp4"
    duration = max(.5, clip["end"] - clip["start"])
    if request.aspect_ratio == "9:16":
        width, height = 1080, 1920
    elif request.aspect_ratio == "1:1":
        width, height = 1080, 1080
    else:
        width, height = 1920, 1080
    face_center = find_face_center(source, clip["start"], clip["end"])
    if face_center:
        center_x, center_y, source_width, source_height = face_center
        scale = max(width / source_width, height / source_height)
        scaled_width = int(round(source_width * scale))
        scaled_height = int(round(source_height * scale))
        crop_x = int(round(center_x * scaled_width - width / 2))
        crop_y = int(round(center_y * scaled_height - height / 2))
        crop_x = max(0, min(scaled_width - width, crop_x)) // 2 * 2
        crop_y = max(0, min(scaled_height - height, crop_y)) // 2 * 2
        framing = f"scale={scaled_width}:{scaled_height}:flags=lanczos,crop={width}:{height}:{crop_x}:{crop_y}"
        smart_reframe = "Face-aware crop"
    else:
        framing = f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,crop={width}:{height}"
        smart_reframe = "Centre crop (no clear face detected)"
    srt = folder / f"clip_{request.clip_index+1}.srt"
    rows = []
    clip_words = [w for w in clip.get("words", job.get("words", [])) if w["start"] >= clip["start"] and w["end"] <= clip["end"]]
    for index in range(0, len(clip_words), 6):
        group = clip_words[index:index+6]
        if not group: continue
        def stamp(value: float) -> str:
            millis = max(0, int(value * 1000))
            return f"{millis//3600000:02}:{(millis//60000)%60:02}:{(millis//1000)%60:02},{millis%1000:03}"
        rows.append(f"{len(rows)+1}\n{stamp(max(0,group[0]['start']-clip['start']))} --> {stamp(max(0,group[-1]['end']-clip['start']))}\n{' '.join(w['text'] for w in group)}\n")
    srt.write_text("\n".join(rows), encoding="utf-8")
    escaped = str(srt).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    vf = f"{framing},fps=30,fade=t=in:st=0:d=0.25,fade=t=out:st={max(0,duration-.25):.3f}:d=0.25,subtitles='{escaped}':force_style='FontName=Arial,FontSize=26,Outline=2,Shadow=1,Alignment=2,MarginV=120'"
    args = ["-ss", str(clip["start"]), "-i", str(source), "-t", str(duration), "-vf", vf, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=in:st=0:d=0.2,afade=t=out:st="+str(max(0,duration-.25))+":d=0.25", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
    result = run_ffmpeg(args)
    if result.returncode != 0:
        # Some bundled FFmpeg builds omit libass; keep the video export usable without burned-in captions.
        vf = f"{framing},fps=30,fade=t=in:st=0:d=0.25,fade=t=out:st={max(0,duration-.25):.3f}:d=0.25"
        args[args.index("-vf")+1] = vf
        result = run_ffmpeg(args)
    if result.returncode != 0 or not output.exists():
        raise HTTPException(status_code=500, detail="Render failed: " + result.stderr[-1000:])
    return {"ok": True, "filename": output.name, "smart_reframe": smart_reframe, "captions_burned_in": "subtitles=" in vf, "download": f"/jobs/{job_id}/files/{output.name}"}

@app.get("/jobs/{job_id}/files/{filename}")
def download_rendered_file(job_id: str, filename: str) -> FileResponse:
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid rendered filename.")
    target = ROOT / job_id / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Rendered clip file not found.")
    return FileResponse(target, media_type="video/mp4", filename=safe_name)

@app.get("/jobs/{job_id}/clips/{clip_index}/download")
def download_clip(job_id: str, clip_index: int) -> FileResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    clip = job.get("clips", [])[clip_index] if 0 <= clip_index < len(job.get("clips", [])) else None
    if not clip:
        raise HTTPException(status_code=404, detail="Clip index not found.")
    folder = Path(clip.get("source_path") or job.get("source", str(ROOT / job_id / "source.mp4"))).parent
    matches = list(folder.glob(f"clip_{clip_index+1}_*.mp4"))
    if not matches:
        raise HTTPException(status_code=404, detail="Render this clip first.")
    return FileResponse(matches[0], media_type="video/mp4", filename=matches[0].name)

@app.get("/jobs/{job_id}/source")
def download_source(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if not job or not job.get("source"):
        raise HTTPException(status_code=404, detail="Source not ready.")
    return FileResponse(job["source"], media_type="video/mp4", filename="clipforge-source.mp4")

@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ClipForge Local AI Engine is running. See /docs for local API documentation."}
