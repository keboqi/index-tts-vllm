"""Backend-independent long-text chunking for segment-at-a-time TTS servers.

Backends whose server synthesizes one segment per request (IndexTTS 2.5,
Drama 3) share this module: split text by ``max_text_tokens``, divide a target
duration across segments, render segments concurrently under a cap, and join
the WAVs in order.

IndexTTS 2.0 (``index``) is excluded on purpose: it splits inside the model
with its own BPE tokenizer and streams a short first chunk. Confucius4-TTS
splits on its own server, which receives ``max_text_tokens`` directly.
"""

from __future__ import annotations

import asyncio
import io
import re
import wave
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from indextts_web.infrastructure.gpu_work import await_gpu_job

_TEXT_BOUNDARY = re.compile(r"(?<=[.!?。！？；;])\s*")
_TEXT_TOKEN_UNIT = re.compile(r"[一-鿿]|[A-Za-z0-9]+|[^\s]")
_WORD = re.compile(r"\S+")
_EMPTY_PAIR = re.compile(r"<([A-Za-z][\w-]*)>\s*</\1>")

T = TypeVar("T")

# Silence tidying for joined segments, ported from minimax_h3's Drama 3 joiner:
# an RMS activity detector over short frames, adaptive for quiet takes.
SILENCE_DBFS = -45.0
FRAME_SECONDS = 0.02
EDGE_PAD_SECONDS = 0.08
MAX_INTERNAL_GAP_SECONDS = 1.0  # longer silences inside a segment are shortened ...
KEPT_GAP_SECONDS = 0.5  # ... to this
_PCM_DTYPES = {2: "<i2", 4: "<i4"}


@dataclass(frozen=True)
class TextMarkup:
    """Inline control tags that chunking must keep intact.

    ``atomic`` matches every tag; a chunk boundary never falls inside one and
    tags carry no token weight. A ``speaker`` tag stays in force until the next
    one, so it is repeated at the start of later chunks. ``paired_open`` and
    ``paired_close`` (group 1 = tag name) scope a span; a span cut by a chunk
    boundary is closed in that chunk and reopened in the next.
    """

    atomic: re.Pattern[str]
    speaker: re.Pattern[str] | None = None
    paired_open: re.Pattern[str] | None = None
    paired_close: re.Pattern[str] | None = None


def text_token_units(text: str) -> int:
    """Estimate token weight without loading the backend tokenizer."""
    units = 0
    for match in _TEXT_TOKEN_UNIT.finditer(text):
        token = match.group(0)
        if token.isascii() and token.isalnum():
            units += max(1, (len(token) + 3) // 4)
        else:
            units += 1
    return max(1, units) if text.strip() else 0


def _weight(text: str, markup: TextMarkup | None) -> int:
    return text_token_units(markup.atomic.sub(" ", text) if markup else text)


def _strip_span(value: str, start: int, end: int) -> tuple[int, int]:
    while start < end and value[start].isspace():
        start += 1
    while end > start and value[end - 1].isspace():
        end -= 1
    return start, end


def split_text(text: str, max_tokens: int, *, markup: TextMarkup | None = None) -> list[str]:
    value = text.strip()
    if not value:
        return []
    tags = [match.span() for match in markup.atomic.finditer(value)] if markup else []

    def tag_at(position: int) -> tuple[int, int] | None:
        return next(((start, end) for start, end in tags if start < position < end), None)

    def after_closers(position: int) -> int:
        # "go.</whisper>" ends its sentence after the closing tag, not before it.
        while markup and markup.paired_close:
            next_text = len(value) - len(value[position:].lstrip())
            closer = markup.paired_close.match(value, next_text)
            if closer is None:
                return position
            position = closer.end()
        return position

    cuts = [0, *(after_closers(m.start()) for m in _TEXT_BOUNDARY.finditer(value)
                 if 0 < m.start() < len(value) and tag_at(m.start()) is None), len(value)]
    pieces = [_strip_span(value, start, end) for start, end in zip(cuts, cuts[1:], strict=False)]
    spans: list[tuple[int, int]] = []
    for start, end in (piece for piece in pieces if piece[0] < piece[1]):
        words: list[list[int]] = []
        for match in _WORD.finditer(value, start, end):
            # Whitespace inside a tag such as "[staring at the plate]" is not a word gap.
            if words and tag_at(match.start()) is not None:
                words[-1][1] = match.end()
            else:
                words.append([match.start(), match.end()])
        if len(words) == 1 and end - start > max_tokens:
            position = start
            while position < end:
                cut = min(end, position + max_tokens)
                if (tag := tag_at(cut)) is not None:
                    cut = tag[1] if tag[1] - position <= max_tokens or tag[0] <= position else tag[0]
                spans.append((position, cut))
                position = cut
        elif len(words) <= max_tokens:
            spans.append((start, end))
        else:
            spans.extend((words[index][0], words[min(index + max_tokens, len(words)) - 1][1])
                         for index in range(0, len(words), max_tokens))
    # Match IndexTTS 2.0: merge adjacent short sentences so clause startup
    # cadence and punctuation pauses do not consume separate hard budgets.
    merged: list[tuple[int, int]] = []
    for span in spans:
        if merged and _weight(value[merged[-1][0]:span[1]], markup) <= max_tokens:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    # A chunk of only tags has nothing to say; tags apply forward, so it joins
    # the next chunk (or the previous one when it is last).
    if markup:
        for index in range(len(merged) - 1, -1, -1):
            if len(merged) > 1 and _weight(value[slice(*merged[index])], markup) == 0:
                if index + 1 < len(merged):
                    merged[index + 1] = (merged[index][0], merged[index + 1][1])
                else:
                    merged[index - 1] = (merged[index - 1][0], merged[index][1])
                del merged[index]
    chunks = [value[start:end].strip() for start, end in merged]
    return _carry_markup(chunks, markup) if markup else chunks


def _carry_markup(chunks: list[str], markup: TextMarkup) -> list[str]:
    speaker: str | None = None
    open_tags: list[str] = []
    result = []
    for chunk in chunks:
        reopen = "".join(f"<{name}>" for name in open_tags)
        lead = markup.speaker.match(chunk) if markup.speaker else None
        head, body = (lead.group(0), chunk[lead.end():].lstrip()) if lead else (speaker or "", chunk)
        for match in markup.atomic.finditer(chunk):
            tag = match.group(0)
            if markup.speaker and markup.speaker.fullmatch(tag):
                speaker = tag
            elif markup.paired_open and (opened := markup.paired_open.fullmatch(tag)):
                open_tags.append(opened.group(1))
            elif markup.paired_close and (closed := markup.paired_close.fullmatch(tag)):
                name = closed.group(1)
                if name in open_tags:
                    del open_tags[len(open_tags) - 1 - open_tags[::-1].index(name)]
        close = "".join(f"</{name}>" for name in reversed(open_tags))
        scoped = _EMPTY_PAIR.sub("", f"{reopen}{body}{close}").strip()
        result.append(" ".join(part for part in (head, scoped) if part))
    return result


def allocate_durations(texts: list[str], total_ms: int, silence_ms: int) -> list[int]:
    if not texts:
        return []
    if total_ms <= 0:
        return [0] * len(texts)
    available = total_ms - max(0, len(texts) - 1) * silence_ms
    if available < len(texts):
        raise ValueError("target duration is shorter than requested inter-sentence silence")
    # IndexTTS 2.0 distributes duration by tokenizer length. This lightweight
    # estimate avoids character-count bias before the remote tokenizer
    # performs the authoritative tokenization.
    weights = [text_token_units(text) for text in texts]
    weight_sum = sum(weights)
    durations = [max(1, round(available * weight / weight_sum)) for weight in weights]
    durations[-1] += available - sum(durations)
    return durations


async def render_segments(
    segments: Sequence[T],
    render: Callable[[T], Awaitable[bytes]],
    *,
    slots: asyncio.Semaphore,
) -> list[bytes]:
    """Render every segment concurrently under ``slots``; results keep input order."""
    async def run(segment: T) -> bytes:
        async with slots:
            return await render(segment)

    # Drain every worker before returning or raising: executor cancellation
    # cannot stop inference threads, and callers release their GPU lease here.
    results = await await_gpu_job(asyncio.gather(
        *(run(segment) for segment in segments), return_exceptions=True,
    ))
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return list(results)


def _silent_frames(samples: np.ndarray, frame: int) -> np.ndarray:
    frames = samples[: len(samples) // frame * frame].reshape(-1, frame)
    if not len(frames):
        return np.zeros(0, dtype=bool)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    # Quiet or whispered segments need a lower threshold than the fixed floor.
    threshold = max(10 ** (-75 / 20), min(10 ** (SILENCE_DBFS / 20), float(rms.max()) * 10 ** (-30 / 20)))
    return rms < threshold


def _silent_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) frame ranges where ``mask`` holds."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def tidy_frames(frames: bytes, params) -> bytes:
    """Trim edge silence and shorten long internal silences of PCM frames.

    Formats the detector cannot read, and segments with no detectable
    activity, are returned unchanged so a join never drops audio outright.
    """
    dtype = _PCM_DTYPES.get(params.sampwidth)
    if dtype is None or params.comptype != "NONE" or params.framerate <= 0:
        return frames
    channels = params.nchannels
    pcm = np.frombuffer(frames, dtype=dtype)
    pcm = pcm[: len(pcm) // channels * channels].reshape(-1, channels)
    mono = pcm.astype(np.float64).mean(axis=1) / float(2 ** (8 * params.sampwidth - 1))
    size = max(1, int(params.framerate * FRAME_SECONDS))
    silent = _silent_frames(mono, size)
    voiced = np.flatnonzero(~silent)
    if not len(voiced):
        return frames
    pad = int(EDGE_PAD_SECONDS / FRAME_SECONDS)
    first = max(0, int(voiced[0]) - pad)
    last = min(len(silent), int(voiced[-1]) + 1 + pad)
    keep = int(KEPT_GAP_SECONDS / FRAME_SECONDS)
    pieces, cursor = [], first
    for start, end in _silent_runs(silent[first:last]):
        start, end = start + first, end + first
        if (end - start) * FRAME_SECONDS > MAX_INTERNAL_GAP_SECONDS:
            pieces.append(pcm[cursor * size:(start + keep // 2) * size])
            cursor = end - (keep - keep // 2)
    # The final partial frame is never assessed; keep it only when speech runs to the end.
    tail = len(pcm) if last == len(silent) else last * size
    pieces.append(pcm[cursor * size:tail])
    return np.concatenate(pieces).astype(dtype).tobytes()


def join_wav(chunks: list[bytes], silence_ms: int = 0, *, trim_silence: bool = False) -> bytes:
    """Join WAV segments in order with ``silence_ms`` between them.

    ``trim_silence`` first tidies each segment (see ``tidy_frames``) so silence
    a backend leaves around chunks does not stack up into long gaps.
    """
    if not chunks:
        raise ValueError("TTS backend returned no WAV chunks")
    output = io.BytesIO()
    params = None
    frames: list[bytes] = []
    for index, chunk in enumerate(chunks):
        with wave.open(io.BytesIO(chunk), "rb") as source:
            current = source.getparams()
            signature = (current.nchannels, current.sampwidth, current.framerate, current.comptype)
            if params is None:
                params = current
            elif signature != (params.nchannels, params.sampwidth, params.framerate, params.comptype):
                raise ValueError("TTS WAV chunks use incompatible audio formats")
            content = source.readframes(source.getnframes())
            frames.append(tidy_frames(content, current) if trim_silence else content)
            if index + 1 < len(chunks) and silence_ms:
                silence_frames = round(current.framerate * silence_ms / 1000)
                frames.append(b"\x00" * silence_frames * current.nchannels * current.sampwidth)
    assert params is not None
    with wave.open(output, "wb") as target:
        target.setparams(params)
        target.writeframes(b"".join(frames))
    return output.getvalue()
