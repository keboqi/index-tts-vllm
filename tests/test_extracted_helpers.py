import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from indextts_web.api.responses import json_event_bytes, status_error
from indextts_web.infrastructure.ffmpeg import (
    codec_args_for_format,
    escape_filter_path,
    escape_filter_quoted_value,
    filter_thread_args,
)
from indextts_web.infrastructure.filenames import sanitize_base_filename
from indextts_web.services.translation.gemini_response import (
    canonicalize_speaker_id,
    extract_response_text,
    parse_gemini_json,
)
from indextts_web.services.translation.timestamps import (
    coerce_segment_ms,
    format_srt_timestamp,
    parse_timestamp_ms,
)
from indextts_web.services.video.cookies import (
    CookieStore,
    cookie_filename_for_domain,
    extract_domain_from_curl,
    extract_domain_from_url,
    parse_curl_cookies,
)
from indextts_web.services.video.library import (
    VideoLibrary,
    decode_downloaded_video_id,
    encode_downloaded_video_id,
)
from indextts_web.services.video.subtitle_style import SubtitleStyle
from indextts_web.services.video.ytdlp import (
    download_format_fallbacks,
    extract_format_ids_from_selector,
    select_raw_video_formats,
)


class GeminiResponseTests(unittest.TestCase):
    def test_fenced_object_normalizes_speakers(self):
        text = '```json\n{"speakers": [{"id": "Speaker 2"}], "segments": [{"text": "hi"}]}\n```'
        segments, speakers = parse_gemini_json(text)
        self.assertEqual(segments, [{"text": "hi"}])
        self.assertEqual(speakers[0]["id"], "speaker2")
        self.assertEqual(speakers[0]["description"], "Speaker2 voice")

    def test_bare_array_and_coerced_pair(self):
        self.assertEqual(parse_gemini_json('[{"a": 1}]'), ([{"a": 1}], []))
        segments, speakers = parse_gemini_json('[{"id": "speaker1"}], "segments": [{"b": 2}]')
        self.assertEqual(segments, [{"b": 2}])
        self.assertEqual(speakers[0]["id"], "speaker1")

    def test_invalid_json_raises_value_error(self):
        with self.assertRaises(ValueError):
            parse_gemini_json("{not json")
        with self.assertRaises(ValueError):
            parse_gemini_json('{"speakers": null}')

    def test_speaker_ids_and_response_text(self):
        self.assertEqual(canonicalize_speaker_id("Narrator 3", 1), "speaker3")
        self.assertEqual(canonicalize_speaker_id("", 4), "speaker4")
        part = SimpleNamespace(text=" hello ")
        response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
        self.assertEqual(extract_response_text(response), "hello")
        self.assertEqual(extract_response_text(None), "")


class TimestampTests(unittest.TestCase):
    def test_lenient_parsing(self):
        self.assertEqual(parse_timestamp_ms(1.5), 1500)
        self.assertEqual(parse_timestamp_ms(2500), 2500)
        self.assertEqual(parse_timestamp_ms("01:02.5"), 62_500)
        self.assertEqual(parse_timestamp_ms("00:00:03,250"), 3250)
        self.assertIsNone(parse_timestamp_ms("abc"))
        self.assertEqual(coerce_segment_ms("4.0"), 4000)
        self.assertEqual(coerce_segment_ms(12.9), 12)
        self.assertIsNone(coerce_segment_ms(""))

    def test_srt_format(self):
        self.assertEqual(format_srt_timestamp(3_723_045), "01:02:03,045")
        self.assertEqual(format_srt_timestamp(-5), "00:00:00,000")


class ResponseHelperTests(unittest.TestCase):
    def test_status_error_payload(self):
        response = status_error("bad", 422, field="x")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(json.loads(response.body), {"status": "error", "message": "bad", "field": "x"})

    def test_events_are_padded_for_flushing(self):
        frame = json_event_bytes({"event": "progress"})
        self.assertTrue(frame.startswith(b"data: ") and frame.endswith(b"\n\n"))
        self.assertGreaterEqual(len(frame), 4096)


class FfmpegFilterTests(unittest.TestCase):
    def test_filter_escaping(self):
        escaped = escape_filter_path("/tmp/a:b,[c];'d'.srt")
        self.assertIn("a\\:b\\,\\[c\\]\\;\\'d\\'", escaped)
        self.assertEqual(escape_filter_quoted_value("it's"), "it\\'s")


class VideoHelperTests(unittest.TestCase):
    def test_cookie_domains(self):
        self.assertEqual(extract_domain_from_url("https://youtu.be/abc"), "youtube.com")
        self.assertEqual(extract_domain_from_url("https://m.bilibili.com/x"), "bilibili.com")
        self.assertEqual(extract_domain_from_curl("curl 'https://www.youtube.com/watch'"), "youtube.com")
        self.assertEqual(parse_curl_cookies("curl x -H 'Cookie: a=1; b=2'"), {"a": "1", "b": "2"})
        self.assertEqual(cookie_filename_for_domain("www.YouTube.com"), "youtube_com_cookies.txt")
        with self.assertRaises(ValueError):
            cookie_filename_for_domain("../etc")

    def test_format_selection(self):
        self.assertEqual(extract_format_ids_from_selector("137+140/best"), ["137", "140"])
        fallbacks = download_format_fallbacks("custom", "720p")
        self.assertEqual(fallbacks[0], "custom")
        self.assertEqual(fallbacks[-2:], ["best", None])
        raw_info = {"formats": [
            {"format_id": "18", "url": "u", "vcodec": "avc1", "acodec": "mp4a", "height": 360},
            {"format_id": "22", "url": "u", "vcodec": "avc1", "acodec": "mp4a", "height": 720},
            {"format_id": "37", "url": "u", "vcodec": "avc1", "acodec": "mp4a", "height": 1080},
        ]}
        video, audio = select_raw_video_formats(raw_info, "720p", "unknown")
        self.assertEqual(video["format_id"], "22")
        self.assertIsNone(audio)


class FilenameAndFfmpegArgTests(unittest.TestCase):
    def test_sanitize_base_filename(self):
        self.assertEqual(sanitize_base_filename("dir/Tom & Jerry — Ep.1.mp4"), "Tom_and_Jerry_-_Ep.1")
        self.assertIsNone(sanitize_base_filename("..."))
        self.assertLessEqual(len(sanitize_base_filename("a" * 400 + ".wav")), 180)

    def test_thread_count_is_applied(self):
        self.assertEqual(codec_args_for_format("ogg", "96k", threads=2),
                         ["-c:a", "libvorbis", "-b:a", "96k", "-threads", "2"])
        self.assertEqual(codec_args_for_format("wav", "96k", threads=0), ["-c:a", "pcm_s16le", "-threads", "0"])
        self.assertEqual(filter_thread_args(3)[1], "3")
        self.assertGreaterEqual(int(filter_thread_args(0)[1]), 1)


class CookieStoreTests(unittest.TestCase):
    def test_lookup_across_primary_and_fallback_dirs(self):
        with tempfile.TemporaryDirectory() as primary, tempfile.TemporaryDirectory() as fallback:
            store = CookieStore(primary, fallback_dirs=(fallback, primary))
            self.assertEqual(store.candidate_dirs(), [os.path.abspath(primary), os.path.abspath(fallback)])
            saved = os.path.join(fallback, "youtube_com_cookies.txt")
            with open(saved, "w", encoding="utf-8") as fh:
                fh.write("# Netscape\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tx\n")
            self.assertEqual(store.saved_sites(), {"youtube.com": 1})
            self.assertEqual(store.file_for_url("https://youtu.be/abc"), saved)
            self.assertEqual(store.ytdlp_opts_for_url("https://www.youtube.com/watch"), {"cookiefile": saved})
            self.assertEqual(store.ytdlp_opts_for_url("https://vimeo.com/1"), {})
            self.assertEqual(store.find_saved_file_for_domain("www.youtube.com"), (saved, "youtube.com"))
            self.assertEqual(store.path_for_domain("youtube.com"),
                             os.path.join(os.path.abspath(primary), "youtube_com_cookies.txt"))


class VideoLibraryTests(unittest.TestCase):
    def test_ids_paths_and_listings(self):
        with tempfile.TemporaryDirectory() as root:
            dirs = [os.path.join(root, name) for name in ("dl", "out", "audio", "snap")]
            for directory in dirs[:2]:
                os.makedirs(directory)
            library = VideoLibrary(*dirs)
            video_id = encode_downloaded_video_id("clip [x].mp4")
            self.assertEqual(decode_downloaded_video_id(video_id), "clip [x].mp4")
            self.assertEqual(library.downloaded_path(video_id), os.path.join(os.path.abspath(dirs[0]), "clip [x].mp4"))
            with self.assertRaises(ValueError):
                library.downloaded_path("../escape.mp4")
            with self.assertRaises(ValueError):
                library.translated_path("notes.txt")

            first = library.output_path_for({"title": "My Video", "id": "abc"}, "webm")
            self.assertEqual(os.path.basename(first), "My_Video [abc].webm")
            open(first, "wb").close()
            self.assertTrue(library.output_path_for({"title": "My Video", "id": "abc"}, "webm").endswith("(2).webm"))
            self.assertTrue(library.output_path_for({"id": "abc"}, "exe").endswith(".mp4"))
            open(os.path.join(dirs[0], "ignored.txt"), "wb").close()

            with patch("indextts_web.services.video.library.ffprobe_duration_seconds", return_value=65.0):
                entries = library.list_downloaded()
            self.assertEqual([entry["filename"] for entry in entries], ["My_Video [abc].webm"])
            self.assertEqual(entries[0]["duration_label"], "1:05")
            self.assertEqual(library.list_translated(), [])
            self.assertEqual(library.find_recent_download(time.time(), {"id": "abc"}), first)
            self.assertIsNone(library.find_recent_download(time.time(), {"id": "zzz"}))
            self.assertTrue(library.snapshot_cache_path(first).startswith(dirs[3]))
            self.assertTrue(library.audio_cache_path(first).endswith(".mp3"))


class SubtitleStyleTests(unittest.TestCase):
    def test_configured_font_file_and_filter_arg(self):
        with tempfile.TemporaryDirectory() as fonts:
            font_file = os.path.join(fonts, "msyh.ttc")
            open(font_file, "wb").close()
            style = SubtitleStyle(font_name="", font_file=font_file, fonts_dir="", font_size=24, local_fonts_dir=fonts)
            self.assertEqual(style.resolve_font(), ("Microsoft YaHei", fonts))
            with patch("indextts_web.services.video.subtitle_style.ffprobe_video_dimensions", return_value=(1280, 720)):
                self.assertEqual(style.metrics_for_video("v.mp4"), (16, 1.0, 0.47, 24))
                arg = style.filter_arg("subs.srt", "v.mp4")
            self.assertTrue(arg.startswith("subtitles=filename="))
            self.assertIn("charenc=UTF-8", arg)
            self.assertIn("FontName=Microsoft YaHei,FontSize=16", arg)
            self.assertEqual(style.metrics_for_video(None), (24, 1.5, 0.7, 36))


if __name__ == "__main__":
    unittest.main()
