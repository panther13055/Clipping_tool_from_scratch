# ClipForge Local AI Engine

This optional free local companion powers full VOD link ingestion, local transcription, ranked highlights and MP4 rendering. It runs on your Windows computer; no paid API key or cloud GPU is used.

For the easiest setup, extract the repository ZIP and double-click `START_CLIPFORGE_WINDOWS.bat` in the repository root. It starts both the web editor and the local AI worker. If Chrome asks for local-network access, allow ClipForge to connect to `127.0.0.1`.

## Start on Windows

1. Install Python 3.11 (64-bit).
2. Double-click `START_CLIPFORGE_WINDOWS.bat` in the repository root (requires Node.js and Python 3.11).
3. The first launch installs dependencies and downloads the Whisper base model when analysis first runs. Keep the terminal open.
4. In ClipForge Studio, open **AI Editor**, paste a public YouTube or Twitch livestream/VOD link and choose **Connect stream / import VOD**. Active streams are captured in rolling 60-second chunks; the AI scans each completed chunk and updates ranked highlights while the stream continues. Enable automatic rendering to create and download new 9:16, square or landscape clips scoring 70+ as they are found. Use **Stop live capture & finish clips** to close the recording and analyse the final chunk.
5. Wait for the local analysis to finish. Pick 9:16, 1:1 or 16:9 and choose **Render & download MP4**.

The local API listens only on `127.0.0.1:8765`. Open `http://127.0.0.1:8765/docs` to inspect its API.

## What it does

- Downloads public archived VODs using yt-dlp (up to 6 hours, capped at 1080p).
- Captures active public YouTube and Twitch livestreams with Streamlink, writing rolling 60-second MPEG-TS chunks without re-encoding the source.
- Transcribes and scores each completed live chunk while capture continues. The strongest 24 moments are kept; high-scoring highlights can be auto-rendered before the broadcast ends.
- Stop capture from the app to close the stream cleanly and analyse the final partial chunk. Scheduled/upcoming streams are not auto-started.
- Transcribes speech locally with faster-whisper base and word timestamps.
- Scores moments using transcript hooks, speaking pace, lexical variety and audio energy, with visible reasons. The score is an explainable ranking signal, not a trained virality predictor.
- Detects faces locally with OpenCV and centres vertical/square crops around the dominant face when one is visible; falls back to a centred crop when it cannot find a face.
- Renders separate H.264/AAC MP4 clips in vertical, square or landscape layouts, with audio normalisation and fades. Captions are burned in when the bundled FFmpeg build supports its subtitles filter; otherwise the video still renders and the app reports that captions were not burned in.

## Important

Only process videos you own or have permission to download and edit. The worker does not log in, bypass private-video restrictions, defeat DRM, or access protected streams. Active capture currently supports public YouTube and Twitch streams through Streamlink. Availability can vary by region, account restrictions, stream format and platform changes. Some platforms may block downloads or require an account, in which case export your own VOD through the platform and import the MP4.

Analysis is local and CPU-based. The first model download and long VOD processing can take time. The current worker is a local companion, not a serverless Vercel backend.
