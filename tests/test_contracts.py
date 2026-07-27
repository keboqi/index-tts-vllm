import unittest

from indextts_web.contracts import (
    audio_chunk_frame,
    decode_keepalive,
    keepalive_frame,
    parse_stream_frame,
)


class StreamingContractTests(unittest.TestCase):
    def test_audio_frame_is_byte_compatible(self):
        frame = audio_chunk_frame(7, b"\x00\x01\x02", is_last=False)
        self.assertEqual(frame, b"CHUNK:7:3:MORE\n\x00\x01\x02")
        parsed = parse_stream_frame(frame)
        self.assertEqual(parsed.chunk_index, 7)
        self.assertFalse(parsed.is_last)
        self.assertEqual(parsed.payload, b"\x00\x01\x02")

    def test_last_audio_frame(self):
        parsed = parse_stream_frame(audio_chunk_frame(2, b"abc", is_last=True))
        self.assertTrue(parsed.is_last)

    def test_keepalive_preserves_unicode(self):
        frame = keepalive_frame("模型加载中", elapsed_seconds=15)
        self.assertEqual(
            decode_keepalive(frame),
            {"message": "模型加载中", "elapsed_seconds": 15},
        )

    def test_corrupt_length_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_stream_frame(b"CHUNK:1:99:LAST\nabc")


if __name__ == "__main__":
    unittest.main()

