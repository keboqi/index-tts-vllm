import asyncio
import io
import unittest
import wave

import numpy as np

from indextts_web.services.tts.chunking import join_wav, render_segments, split_text
from indextts_web.services.tts.drama3_manager import DRAMA3_MARKUP, emotion_prefix


class SplitTextTests(unittest.TestCase):
    def test_plain_text_merges_short_sentences_up_to_budget(self):
        self.assertEqual(split_text("One. Two words! Three.", 120), ["One. Two words! Three."])
        self.assertEqual(split_text("One. Two. Three.", 4), ["One. Two.", "Three."])

    def test_cut_long_word_never_gains_inner_spaces(self):
        chunks = split_text("abcdefghij", 3)
        self.assertEqual("".join(chunks), "abcdefghij")

    def test_point_tag_with_punctuation_and_spaces_stays_whole(self):
        text = "[horrified, staring at the plate. really] No. Absolutely not. [sighs] Fine."
        self.assertEqual(split_text(text, 3, markup=DRAMA3_MARKUP), [
            "[horrified, staring at the plate. really] No.", "Absolutely not.", "[sighs] Fine.",
        ])

    def test_speaker_tag_carries_into_later_chunks(self):
        text = "<|speaker:0|> Hi there. How are you? <|speaker:1|> Fine. Bye now."
        self.assertEqual(split_text(text, 3, markup=DRAMA3_MARKUP), [
            "<|speaker:0|> Hi there.", "<|speaker:0|> How are you?",
            "<|speaker:1|> Fine.", "<|speaker:1|> Bye now.",
        ])

    def test_paired_tag_cut_by_boundary_is_closed_and_reopened(self):
        text = "<whisper>Go on without me. Please, just go.</whisper> Okay."
        self.assertEqual(split_text(text, 4, markup=DRAMA3_MARKUP), [
            "<whisper>Go on without me.</whisper>", "<whisper>Please, just go.</whisper>", "Okay.",
        ])

    def test_tag_only_tail_joins_previous_chunk(self):
        self.assertEqual(split_text("Hello there. Bye now. [laughs]", 2, markup=DRAMA3_MARKUP),
                         ["Hello there.", "Bye now. [laughs]"])

    def test_cjk_slicing_never_cuts_inside_a_tag(self):
        text = "你好世界[开心地笑]今天天气很好我们出去玩吧"
        chunks = split_text(text, 6, markup=DRAMA3_MARKUP)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(any("[开心地笑]" in chunk for chunk in chunks))
        for chunk in chunks:
            self.assertEqual(chunk.count("["), chunk.count("]"))

    def test_unclosed_bracket_is_plain_text(self):
        self.assertEqual(split_text("a [b. c.", 4, markup=DRAMA3_MARKUP), ["a [b.", "c."])

    def test_emotion_prefix_keeps_speaker_tag_first(self):
        self.assertEqual(emotion_prefix("<|speaker:1|> Fine.", "sad"), "<|speaker:1|> [sad] Fine.")
        self.assertEqual(emotion_prefix("Fine.", "[sad]"), "[sad] Fine.")


class RenderSegmentsTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_under_cap_and_ordered(self):
        running = peak = 0

        async def render(value):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01 * (5 - value))
            running -= 1
            return bytes([value])

        result = await render_segments(range(5), render, slots=asyncio.Semaphore(3))
        self.assertEqual(result, [bytes([i]) for i in range(5)])
        self.assertEqual(peak, 3)

    async def test_failure_waits_for_running_and_queued_segments(self):
        running = asyncio.Event()
        queued_started = asyncio.Event()
        release = asyncio.Event()
        finished = []
        failure = RuntimeError("first chunk failed")

        async def render(value):
            if value == 0:
                await running.wait()
                raise failure
            (running if value == 1 else queued_started).set()
            await release.wait()
            finished.append(value)
            if value == 2:
                raise RuntimeError("later chunk failed")
            return b"ok"

        task = asyncio.create_task(render_segments(range(3), render, slots=asyncio.Semaphore(2)))
        try:
            await asyncio.wait_for(queued_started.wait(), timeout=1)
            await asyncio.sleep(0)
            self.assertFalse(task.done())
        finally:
            release.set()
            with self.assertRaises(RuntimeError) as caught:
                await task
        self.assertIs(caught.exception, failure)
        self.assertEqual(sorted(finished), [1, 2])

    async def test_cancellation_waits_for_workers(self):
        started = asyncio.Event()
        release = asyncio.Event()
        finished = []

        async def render(value):
            started.set()
            await release.wait()
            finished.append(value)
            return b"ok"

        task = asyncio.create_task(render_segments(range(2), render, slots=asyncio.Semaphore(1)))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
        finally:
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(finished, [0, 1])


if __name__ == "__main__":
    unittest.main()


RATE = 24000


def pcm_wav(*parts: tuple[float, float]) -> bytes:
    """(seconds, amplitude) runs of a 220 Hz tone; amplitude 0 is silence."""
    samples = np.concatenate([
        amplitude * np.sin(2 * np.pi * 220 * np.arange(int(RATE * seconds)) / RATE) for seconds, amplitude in parts
    ])
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(RATE)
        audio.writeframes((samples * 32767).astype("<i2").tobytes())
    return stream.getvalue()


def seconds(content: bytes) -> float:
    with wave.open(io.BytesIO(content), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


class JoinWavTests(unittest.TestCase):
    def test_trim_silence_tidies_edges_and_long_internal_gaps(self):
        padded = pcm_wav((1.5, 0), (0.5, 0.3), (3.0, 0), (0.5, 0.3), (2.0, 0))
        self.assertAlmostEqual(seconds(join_wav([padded, padded], 100)), 15.1, places=2)
        # Each segment: 0.08 pad + 0.5 voice + 0.5 kept gap + 0.5 voice + 0.08 pad.
        self.assertAlmostEqual(seconds(join_wav([padded, padded], 100, trim_silence=True)), 3.42, places=2)

    def test_trim_silence_keeps_quiet_speech_and_silent_segments(self):
        whisper = pcm_wav((0.5, 0), (1.0, 0.003), (0.5, 0))
        silent = pcm_wav((0.4, 0))
        joined = join_wav([whisper, silent], trim_silence=True)
        self.assertAlmostEqual(seconds(joined), 1.16 + 0.4, places=2)
