import unittest

from indextts_web.services.translation.subtitles import (
    parse_subtitle_entries,
    parse_subtitle_input,
    parse_timestamp,
)


class SubtitleTests(unittest.TestCase):
    def test_srt_and_vtt_timestamps(self):
        self.assertEqual(parse_timestamp("01:02:03,045"), 3_723_045)
        self.assertEqual(parse_timestamp("02:03.5"), 123_500)
        self.assertEqual(parse_timestamp("invalid"), 0)

    def test_multiline_srt(self):
        entries = parse_subtitle_entries(
            "1\n00:00:01,000 --> 00:00:02,500\nHello\nworld\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nNext\n"
        )
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["text"], "Hello\nworld")
        self.assertEqual(entries[1]["sequence"], 2)

    def test_bilingual_subtitles_match_by_sequence(self):
        original = "1\n00:00:01,000 --> 00:00:02,000\nHello\n"
        translated = "1\n00:00:01,000 --> 00:00:02,000\n你好\n"
        segments, speakers = parse_subtitle_input(original, translated)
        self.assertEqual(segments[0]["source_text"], "Hello")
        self.assertEqual(segments[0]["translated_text"], "你好")
        self.assertEqual(segments[0]["start"], "00:01")
        self.assertEqual(speakers[0]["id"], "speaker1")


if __name__ == "__main__":
    unittest.main()
