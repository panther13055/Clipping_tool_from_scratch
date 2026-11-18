# ClipForge Studio

A browser-based video editing studio for creators, streamers, and video editors.

## Product direction

- Multi-track editing workspace for short-form and long-form video
- Automatic captions with editable styling and speaker labels
- Streamer/gameplay layouts with adjustable facecam and gameplay regions
- Motion graphics, transitions, effects, and keyframe workflows
- AI clipping, speaker tracking, and enhancement integrations
- High-quality export through a dedicated rendering service

## Architecture

The web application is hosted on Vercel. Media processing and long-running AI/render jobs should run in dedicated workers rather than Vercel request handlers.

## Status

Early foundation. Features are added incrementally and must be tested before being described as production-ready.
