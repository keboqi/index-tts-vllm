import io
import tempfile
import unittest
import wave
from pathlib import Path

from indextts_web.config import AppSettings
from indextts_web.services.tts.index25_manager import (
    ManagedIndexTTS25Backend,
    allocate_durations,
    fit_wav_duration,
    join_wav,
    split_text,
)


def wav_bytes(frames=2205):
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(22050)
        target.writeframes(b"\x01\x00" * frames)
    return output.getvalue()


class IndexTTS25ManagerTests(unittest.TestCase):
    def manager(self, directory):
        settings = AppSettings(
            indextts25_repo_dir=directory,
            indextts25_model_dir=str(Path(directory) / "model"),
            indextts25_data_dir=str(Path(directory) / "data"),
        )
        return ManagedIndexTTS25Backend(
            settings,
            app_dir=Path(directory),
            output_root=Path(directory) / "outputs",
        )

    def test_split_duration_and_exact_wav_contract(self):
        texts = split_text("One. Two words! Three.", 120)
        durations = allocate_durations(texts, 1800, 100)
        self.assertEqual(texts, ["One.", "Two words!", "Three."])
        self.assertEqual(sum(durations) + 200, 1800)
        joined = join_wav([wav_bytes(), wav_bytes()], 100)
        fitted = fit_wav_duration(joined, 250)
        with wave.open(io.BytesIO(fitted), "rb") as source:
            self.assertEqual(source.getnframes(), round(22050 * 0.25))

    def test_payload_matches_vllm_omni_speech_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "prompt.wav"
            prompt.write_bytes(wav_bytes())
            manager = self.manager(directory)
            payload = manager.build_payload(
                text="Hello.",
                language="en",
                prompt_wav=str(prompt),
                reference_text="reference",
                target_duration_ms=1200,
                diffusion_steps=12,
                emotion_audio=None,
                emotion_text="happy",
                emotion_weight=0.7,
                cache_prompt_audio=True,
                seed=42,
                sampling={"temperature": 0.8, "top_k": 30, "max_new_tokens": 700},
            )
        self.assertEqual(payload["model"], "IndexTeam/IndexTTS-2.5")
        self.assertTrue(payload["ref_audio"].startswith("data:audio/"))
        self.assertEqual(payload["max_new_tokens"], 700)
        self.assertEqual(payload["extra_params"]["lang"], "en")
        self.assertEqual(payload["extra_params"]["target_duration_ms"], 1200)
        self.assertTrue(payload["extra_params"]["use_emo_text"])

    def test_language_contract_rejects_untested_languages(self):
        self.assertEqual(ManagedIndexTTS25Backend.resolve_language("Arabic", "hello"), "ar")
        self.assertEqual(ManagedIndexTTS25Backend.resolve_language("auto", "こんにちは"), "ja")
        with self.assertRaisesRegex(ValueError, "unsupported IndexTTS 2.5 language"):
            ManagedIndexTTS25Backend.resolve_language("French", "bonjour")


if __name__ == "__main__":
    unittest.main()
