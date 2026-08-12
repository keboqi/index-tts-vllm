import unittest
from pathlib import Path
from types import SimpleNamespace

from indextts_web.services.tts.base import SynthesisRequest
from indextts_web.services.tts.confucius import ConfuciusBackend
from indextts_web.services.tts.index import IndexBackend
from indextts_web.services.tts.index25 import IndexTTS25Backend
from indextts_web.services.tts.registry import BackendRegistry


class AsyncContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class FakeTTS:
    def __init__(self):
        self.kwargs = None

    async def infer(self, **kwargs):
        self.kwargs = kwargs
        return kwargs["output_path"]


class FakeIndexManager:
    def __init__(self):
        self.tts = FakeTTS()
        self.awake = False

    async def ensure_awake(self):
        self.awake = True

    def get_tts(self):
        return self.tts

    def is_ready(self):
        return True

    def vllm_status(self):
        return {"indextts_vllm_sleeping": False}


class FakeExternalManager:
    def __init__(self):
        self.kwargs = None

    async def synthesize_to_file(self, **kwargs):
        self.kwargs = kwargs
        return kwargs["output_path"]

    async def status(self):
        return {"ready": True}

    async def shutdown(self):
        return None


class BackendAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_index_forced_ffmpeg_disables_native_duration(self):
        manager = FakeIndexManager()
        postprocess_calls = []

        async def postprocess(path, duration):
            postprocess_calls.append((path, duration))

        legacy = SimpleNamespace(
            INDEXTTS_GPU_WORK_SLOTS=AsyncContext(),
            tts_manager=manager,
            _postprocess_ffmpeg_duration=postprocess,
        )
        backend = IndexBackend(legacy)
        request = SynthesisRequest(
            text="hello",
            output_path=Path("output.wav"),
            target_duration_ms=1200,
            duration_control="ffmpeg",
        )
        self.assertEqual(await backend.synthesize(request), Path("output.wav"))
        self.assertEqual(manager.tts.kwargs["speech_length"], 0)
        self.assertEqual(postprocess_calls, [("output.wav", 1200)])

    async def test_external_backend_request_mapping(self):
        confucius = FakeExternalManager()

        async def resolve_confucius(**_kwargs):
            return "confucius.wav"

        async def postprocess(_path, _duration):
            return None

        legacy = SimpleNamespace(
            confucius_backend_manager=confucius,
            _resolve_confucius_prompt_audio=resolve_confucius,
            _postprocess_ffmpeg_duration=postprocess,
        )
        request = SynthesisRequest(
            text="hello",
            output_path=Path("output.wav"),
            speaker_preset="voice",
            target_duration_ms=1000,
            sampling={"temperature": 0.4, "top_k": 9},
        )
        await ConfuciusBackend(legacy).synthesize(request)
        self.assertEqual(confucius.kwargs["prompt_wav"], "confucius.wav")
        self.assertEqual(confucius.kwargs["speech_length"], 1000)

    async def test_indextts25_backend_maps_native_controls(self):
        manager = FakeExternalManager()

        async def resolve_indextts25(**_kwargs):
            return "speaker.wav"

        legacy = SimpleNamespace(
            indextts25_backend_manager=manager,
            _resolve_indextts25_prompt_audio=resolve_indextts25,
            _postprocess_ffmpeg_duration=lambda *_args: None,
        )
        request = SynthesisRequest(
            text="hello",
            output_path=Path("output.wav"),
            speaker_preset="voice",
            language="en",
            target_duration_ms=1500,
            emotion_text="happy",
            emotion_weight=0.7,
            seed=41,
            sampling={"temperature": 0.6, "max_new_tokens": 700},
        )
        await IndexTTS25Backend(legacy).synthesize(request)
        self.assertEqual(manager.kwargs["prompt_wav"], "speaker.wav")
        self.assertEqual(manager.kwargs["speech_length"], 1500)
        self.assertEqual(manager.kwargs["emotion_text"], "happy")
        self.assertEqual(manager.kwargs["sampling"]["max_new_tokens"], 700)

    async def test_registry_rejects_unknown_backend(self):
        legacy = SimpleNamespace()
        registry = BackendRegistry([IndexBackend(legacy)], default="index")
        self.assertEqual(registry.get().name, "index")
        with self.assertRaises(ValueError):
            registry.get("missing")


if __name__ == "__main__":
    unittest.main()
