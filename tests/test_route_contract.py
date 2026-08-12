import ast
import unittest
from pathlib import Path

from indextts_web.route_groups import route_group

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_ROUTES = {
    ("POST", "/internal/snapshot/warmup"),
    ("POST", "/internal/snapshot/sleep"),
    ("POST", "/internal/snapshot/wake"),
    ("GET", "/"),
    ("POST", "/api/estimate_duration"),
    ("POST", "/api/clear_outputs"),
    ("GET", "/api/prompt_templates"),
    ("GET", "/api/stable-audio/status"),
    ("GET", "/api/stable-audio/models"),
    ("POST", "/api/stable-audio/unload"),
    ("GET", "/api/models/status"),
    ("POST", "/api/models/unload"),
    ("POST", "/api/models/wake"),
    ("POST", "/api/stable-audio/generate"),
    ("GET", "/api/cookies"),
    ("GET", "/api/cookies/{domain}/download"),
    ("GET", "/api/video_ytdlp_diagnostics"),
    ("POST", "/api/cookies/import_curl"),
    ("POST", "/api/cookies/upload"),
    ("DELETE", "/api/cookies/{domain}"),
    ("GET", "/api/downloaded_videos"),
    ("GET", "/api/translated_videos"),
    ("DELETE", "/api/translated_videos/{filename}"),
    ("DELETE", "/api/downloaded_videos/{filename}"),
    ("GET", "/api/downloaded_videos/{filename}"),
    ("GET", "/api/downloaded_videos/{filename}/snapshot"),
    ("GET", "/api/downloaded_videos/{filename}/audio"),
    ("POST", "/api/video_info"),
    ("POST", "/api/video_download"),
    ("POST", "/api/video_replace_audio"),
    ("POST", "/api/translate_split_audio"),
    ("POST", "/api/translate_merge_chunks"),
    ("POST", "/api/translate_generate_chunks"),
    ("POST", "/api/translate_segments"),
    ("POST", "/api/translate_generate_segments"),
    ("POST", "/api/translate_segment_preview"),
    ("GET", "/api/translate_backing_track/{session_id}"),
    ("GET", "/api/translate_vocals/{session_id}"),
    ("GET", "/api/translate_download_chunks/{batch_id}"),
    ("POST", "/api/translate_upload_transcriptions/{batch_id}"),
    ("GET", "/api/translate_outputs/{filename}"),
    ("GET", "/api/translate_outputs/{filename}/snapshot"),
    ("POST", "/api/translate_audio"),
    ("POST", "/add_speaker"),
    ("POST", "/delete_speaker"),
    ("POST", "/delete_all_speakers"),
    ("GET", "/audio_roles"),
    ("GET", "/api/speaker_preview/{speaker_name}"),
    ("GET", "/speaker_effects"),
    ("POST", "/api/design-voice"),
    ("POST", "/api/design-voice/save-preset"),
    ("GET", "/api/design-voice/languages"),
    ("GET", "/api/design-voice/status"),
    ("GET", "/api/segment_preview/{session_id}/{segment_index}"),
    ("POST", "/speak"),
    ("POST", "/clone_voice"),
    ("GET", "/server_info"),
    ("POST", "/speak_stream"),
    ("POST", "/clone_voice_stream"),
}


def source_routes():
    source = (ROOT / "fastapi_webui_v2_impl.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    routes = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                continue
            owner = decorator.func.value
            if isinstance(owner, ast.Name) and owner.id == "app" and decorator.args:
                routes.add((decorator.func.attr.upper(), ast.literal_eval(decorator.args[0])))
    return routes


class RouteContractTests(unittest.TestCase):
    def test_legacy_route_inventory_is_stable(self):
        self.assertEqual(source_routes(), EXPECTED_ROUTES)
        self.assertEqual(len(EXPECTED_ROUTES), 59)

    def test_every_compatibility_route_has_one_feature_group(self):
        grouped = {path: route_group(path) for _method, path in EXPECTED_ROUTES}
        self.assertNotIn(None, grouped.values())
        self.assertEqual(set(grouped.values()), {
            "internal",
            "models",
            "speakers",
            "stable-audio",
            "translation",
            "tts",
            "ui",
            "utilities",
            "video",
        })


if __name__ == "__main__":
    unittest.main()
