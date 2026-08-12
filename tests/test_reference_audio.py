import io
import tempfile
import unittest
import wave
from pathlib import Path

from indextts_web.services.tts.reference_audio import ensure_minimum_reference_duration


def wav_bytes(frames: int, rate: int = 22050) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(b"\x01\x00" * frames)
    return output.getvalue()


class ReferenceAudioTests(unittest.TestCase):
    def test_short_wav_is_padded_once_for_every_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "short.wav"
            prompt.write_bytes(wav_bytes(11025))
            first = ensure_minimum_reference_duration(prompt.as_posix(), cache_dir=root / "cache")
            second = ensure_minimum_reference_duration(prompt.as_posix(), cache_dir=root / "cache")
            self.assertEqual(first, second)
            self.assertNotEqual(first, prompt.as_posix())
            with wave.open(str(first), "rb") as source:
                self.assertEqual(source.getnframes(), source.getframerate())

    def test_long_wav_and_remote_reference_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "long.wav"
            prompt.write_bytes(wav_bytes(22050))
            self.assertEqual(
                ensure_minimum_reference_duration(str(prompt), cache_dir=root / "cache"),
                str(prompt),
            )
        remote = "https://example.com/reference.wav"
        self.assertEqual(
            ensure_minimum_reference_duration(remote, cache_dir=Path("unused")),
            remote,
        )


if __name__ == "__main__":
    unittest.main()
