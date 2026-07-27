import unittest
from pathlib import Path
from types import SimpleNamespace

from indextts_web.services.tts.base import SynthesisRequest
from indextts_web.services.tts.confucius import ConfuciusBackend
from indextts_web.services.tts.higgs import HiggsBackend
from indextts_web.services.tts.index import IndexBackend
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
        higgs = FakeExternalManager()

        async def resolve_confucius(**_kwargs):
            return "confucius.wav"

        async def resolve_external(**_kwargs):
            return "higgs.wav"

        async def postprocess(_path, _duration):
            return None

        legacy = SimpleNamespace(
            confucius_backend_manager=confucius,
            higgs_backend_manager=higgs,
            _resolve_confucius_prompt_audio=resolve_confucius,
            _resolve_external_prompt_audio=resolve_external,
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
        await HiggsBackend(legacy).synthesize(request)
        self.assertEqual(confucius.kwargs["prompt_wav"], "confucius.wav")
        self.assertEqual(confucius.kwargs["speech_length"], 1000)
        self.assertEqual(higgs.kwargs["prompt_wav"], "higgs.wav")
        self.assertEqual(higgs.kwargs["temperature"], 0.4)
        self.assertEqual(higgs.kwargs["top_k"], 9)

    async def test_registry_rejects_unknown_backend(self):
        legacy = SimpleNamespace()
        registry = BackendRegistry([IndexBackend(legacy)], default="index")
        self.assertEqual(registry.get().name, "index")
        with self.assertRaises(ValueError):
            registry.get("missing")


if __name__ == "__main__":
    unittest.main()
