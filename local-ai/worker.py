from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import uuid
import wave
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
    update(job_id, source=str(source), title=info.get("title") or source.stem, duration=info.get("duration") or 0)
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
                    import array
                    samples = array.array("h")
                    samples.frombytes(raw)
                    if samples:
                        values[second] = min(1.0, (sum(x*x for x in samples) / len(samples)) ** 0.5 / 32768)
                second += 1
    except Exception:
        return {}
    return values

HYPE = re.compile(r"\b(wow|no way|oh my god|let'?s go|insane|crazy|amazing|unbelievable|bro|wait|finally|winner|win|won|clutch|goal|fail|omg|seriously|really|why|how|secret|best|worst|never|first|last|actually|incredible|what happened|you won)\b", re.I)

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
            candidates.append({
                "start": round(max(0, selected[0]["start"]), 2),
                "end": round(min(duration, selected[-1]["end"]), 2),
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
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    job_id = uuid.uuid4().hex[:12]
    with LOCK:
        JOBS[job_id] = {"id": job_id, "status": "processing", "stage": "Queued", "progress": 0, "message": "Queued local analysis…", "clips": [], "created": time.time()}
    thread = threading.Thread(target=analyse_job, args=(job_id, raw), daemon=True)
    thread.start()
    return {"job_id": job_id, "message": "Local VOD analysis started."}

@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found. The local worker may have restarted.")
    return {k: v for k, v in job.items() if k not in ("source", "words")}

@app.post("/jobs/{job_id}/render")
def render_clip(job_id: str, request: RenderRequest) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job or job.get("status") != "ready":
        raise HTTPException(status_code=409, detail="Wait for analysis to finish first.")
    clips = job.get("clips", [])
    if request.clip_index < 0 or request.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip index not found.")
    if request.aspect_ratio not in ("9:16", "1:1", "16:9"):
        raise HTTPException(status_code=400, detail="Unsupported aspect ratio.")
    clip = clips[request.clip_index]
    source = Path(job["source"])
    folder = source.parent
    output = folder / f"clip_{request.clip_index+1}_{request.aspect_ratio.replace(':','x')}.mp4"
    duration = max(.5, clip["end"] - clip["start"])
    if request.aspect_ratio == "9:16":
        width, height = 1080, 1920
    elif request.aspect_ratio == "1:1":
        width, height = 1080, 1080
    else:
        width, height = 1920, 1080
    srt = folder / f"clip_{request.clip_index+1}.srt"
    rows = []
    clip_words = [w for w in job.get("words", []) if w["start"] >= clip["start"] and w["end"] <= clip["end"]]
    for index in range(0, len(clip_words), 6):
        group = clip_words[index:index+6]
        if not group: continue
        def stamp(value: float) -> str:
            millis = max(0, int(value * 1000))
            return f"{millis//3600000:02}:{(millis//60000)%60:02}:{(millis//1000)%60:02},{millis%1000:03}"
        rows.append(f"{len(rows)+1}\n{stamp(max(0,group[0]['start']-clip['start']))} --> {stamp(max(0,group[-1]['end']-clip['start']))}\n{' '.join(w['text'] for w in group)}\n")
    srt.write_text("\n".join(rows), encoding="utf-8")
    escaped = str(srt).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps=30,fade=t=in:st=0:d=0.25,fade=t=out:st={max(0,duration-.25):.3f}:d=0.25,subtitles='{escaped}':force_style='FontName=Arial,FontSize=26,Outline=2,Shadow=1,Alignment=2,MarginV=120'"
    args = ["-ss", str(clip["start"]), "-i", str(source), "-t", str(duration), "-vf", vf, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=in:st=0:d=0.2,afade=t=out:st="+str(max(0,duration-.25))+":d=0.25", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
    result = run_ffmpeg(args)
    if result.returncode != 0:
        # Some bundled FFmpeg builds omit libass; keep the video export usable without burned-in captions.
        vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps=30,fade=t=in:st=0:d=0.25,fade=t=out:st={max(0,duration-.25):.3f}:d=0.25"
        args[args.index("-vf")+1] = vf
        result = run_ffmpeg(args)
    if result.returncode != 0 or not output.exists():
        raise HTTPException(status_code=500, detail="Render failed: " + result.stderr[-1000:])
    return {"ok": True, "filename": output.name, "captions_burned_in": "subtitles=" in vf, "download": f"/jobs/{job_id}/clips/{request.clip_index}/download"}

@app.get("/jobs/{job_id}/clips/{clip_index}/download")
def download_clip(job_id: str, clip_index: int) -> FileResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    folder = Path(job["source"]).parent
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
