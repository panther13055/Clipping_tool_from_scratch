# ClipForge Local AI Engine

This optional free local companion powers full VOD link ingestion, local transcription, ranked highlights and MP4 rendering. It runs on your Windows computer; no paid API key or cloud GPU is used.

## Start on Windows

1. Install Python 3.10–3.12 (64-bit).
2. Open the repository folder and run `local-ai/START_WINDOWS.bat`.
3. The first launch installs dependencies and downloads the Whisper base model when analysis first runs. Keep the terminal open.
4. In ClipForge Studio, open **AI Clips**, paste a public YouTube/Twitch/Vimeo VOD link and choose **Import video link**.
5. Wait for the local analysis to finish. Pick 9:16, 1:1 or 16:9 and choose **Render & download MP4**.

The local API listens only on `127.0.0.1:8765`. Open `http://127.0.0.1:8765/docs` to inspect its API.

## What it does

- Downloads public archived VODs using yt-dlp (up to 6 hours, capped at 1080p).
- Rejects live/upcoming streams; analyse the replay after it is available.
- Transcribes speech locally with faster-whisper base and word timestamps.
- Scores moments using transcript hooks, speaking pace, lexical variety and audio energy, with visible reasons.
- Renders separate H.264/AAC MP4 clips in vertical, square or landscape layouts, with audio normalisation and fades. Captions are burned in when the bundled FFmpeg build supports its subtitles filter; otherwise the video still renders and the app reports that captions were not burned in.

## Important

Only process videos you own or have permission to download and edit. The worker does not log in, bypass private-video restrictions, defeat DRM, or access protected streams. YouTube live streams are not analysed in real time; use the archived VOD. Some platforms may block downloads or require an account, in which case export your own VOD through the platform and import the MP4.

Analysis is local and CPU-based. The first model download and long VOD processing can take time. The current worker is a local companion, not a serverless Vercel backend.
