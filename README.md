# ClipForge Studio

A browser-based, local-first video editing workspace built with Next.js, React and TypeScript.

## Run locally

```bash
npm install
npm run dev
```

Open http://localhost:3000.

## Editing features

- Import multiple local video, audio and image assets into the media bin.
- Preview the selected video and assign separate gameplay, facecam and music sources.
- Build a vertical 9:16, landscape 16:9 or square 1:1 composition.
- Drag and resize gameplay/facecam regions; animate their position and size with keyframes.
- Adjust selected overlay opacity.
- Generate multilingual word-timed captions with Whisper through Transformers.js; edit caption text, choose from the caption style library, position and scale captions, and export SRT.
- Detect scene changes locally and add split markers.
- Mark ranges for deletion; preview playback and export skip those ranges.
- Use MediaPipe Face Landmarker for local face tracking and visual mouth-movement-based reframing.
- Apply live colour/effect previews and export denoise, sharpen, deshake, motion blur, deinterlace and film-grain filters.
- Record a composited canvas export with captions and overlays to WebM; convert it to H.264 MP4 with FFmpeg.wasm.
- Export at 1080p or 4K where the browser has enough memory and encoder support.
- Local workspace settings, timed captions, undo and redo are saved in the browser.

## First-run downloads

The browser downloads model/runtime assets when the corresponding feature is first used:
- Whisper model weights for speech recognition.
- MediaPipe face-landmarker model and WASM runtime.
- FFmpeg.wasm runtime for MP4 conversion.

These are free/open model and runtime downloads; no paid API key is required. Model inference and editing happen in the browser. Imported media is not uploaded by these features.

## AI VOD clipping (optional local engine)

ClipForge includes an optional Windows local AI worker in `local-ai/`. Run `local-ai/START_WINDOWS.bat`, keep its terminal open, then use the AI Editor tab to submit a public archived YouTube, Twitch or Vimeo VOD you own or are authorised to process. The worker uses yt-dlp, faster-whisper (CPU), and FFmpeg to download the VOD, transcribe it, rank candidate moments with explainable transcript/audio signals, and render individual H.264 MP4 clips in 9:16, 1:1 or 16:9. First launch installs free dependencies and downloads the Whisper base model. See `local-ai/README.md`. On Windows, `START_CLIPFORGE_WINDOWS.bat` starts both the web app and the local AI worker in separate terminals. The worker uses OpenCV face detection to centre the crop on a visible face when possible and renders individual MP4s; the top five can be batch-rendered.

The local worker does not bypass private access, DRM, login walls or platform restrictions. Live capture depends on public stream availability and supported platform formats. It runs on your PC, not on Vercel; local processing avoids paid inference APIs.

## Current limitations

- Media files themselves are not persisted in local storage; re-import them after a page reload.
- Face tracking estimates a visually active face from mouth movement. It is not voice diarization and does not reliably identify speakers from audio.
- AI video super-resolution/frame-by-frame Real-ESRGAN processing is not implemented. The 4K setting renders the composition at 4K canvas dimensions; it does not invent source detail.
- FFmpeg denoise, sharpen, deshake, motion blur, deinterlace and film-grain options are conventional filters, not AI enhancement.
- The timeline currently provides a main source with separate gameplay, facecam, caption and audio layers; it is not yet a full unlimited multi-clip NLE timeline.
- WebM export includes the composited preview; audio fade filters are currently applied during MP4 conversion only.
- Very long clips and 4K exports can exceed browser memory or encoder limits. Use shorter ranges or 1080p if the browser becomes unstable.

## Keyboard shortcuts

- Space: play/pause
- S: add split at playhead
- Ctrl/Cmd + O: import media
- Ctrl/Cmd + Z: undo
- Ctrl/Cmd + Shift + Z: redo
- Escape: close shortcut help
