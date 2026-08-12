"""Pure route-to-feature classification used by app assembly and tests."""

from __future__ import annotations


def route_group(path: str) -> str | None:
    if path == "/":
        return "ui"
    if path.startswith("/internal/"):
        return "internal"
    if path in {
        "/api/estimate_duration",
        "/api/clear_outputs",
        "/api/prompt_templates",
    }:
        return "utilities"
    if path.startswith("/api/models/"):
        return "models"
    if path.startswith("/api/stable-audio/"):
        return "stable-audio"
    if path.startswith(("/api/cookies", "/api/video", "/api/downloaded_videos", "/api/translated_videos")):
        return "video"
    if path.startswith(("/api/translate", "/api/segment_preview")):
        return "translation"
    if path in {
        "/add_speaker",
        "/delete_speaker",
        "/delete_all_speakers",
        "/audio_roles",
        "/speaker_effects",
    } or path.startswith(("/api/speaker_preview", "/api/design-voice")):
        return "speakers"
    if path in {"/speak", "/clone_voice", "/server_info", "/speak_stream", "/clone_voice_stream"}:
        return "tts"
    return None
