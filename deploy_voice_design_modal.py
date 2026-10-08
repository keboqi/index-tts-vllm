"""
Standalone Modal Deployment Script: Qwen3-TTS Voice Design Service
===================================================================
Extracted from `index-tts-vllm`.
Runs on NVIDIA L4 GPU (24GB VRAM) with persistent model caching and scale-to-zero.

Features:
- Powered by Alibaba's `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- Native NVIDIA L4 GPU hardware allocation
- Endpoints:
    - POST `/api/design-voice`: Direct drop-in compatibility with `index-tts-vllm`
    - POST `/v1/audio/speech`: OpenAI-compatible TTS endpoint
    - GET  `/api/design-voice/languages`: Supported languages list
    - GET  `/api/design-voice/status`: Health check & GPU metrics
    - GET  `/`: Embedded modern interactive web studio UI
- CLI testing: `modal run deploy_voice_design_modal.py`
- Deployment:  `modal deploy deploy_voice_design_modal.py`
"""

import io
import os
import time
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List

import modal

# ============================================================================
# 1. MODAL APP & STORAGE CONFIGURATION
# ============================================================================

APP_NAME = "qwen3-voice-design-service"
MODEL_REPO_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CACHE_DIR = "/cache"
MODEL_CACHE_DIR = f"{CACHE_DIR}/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("voice-design-model-cache", create_if_missing=True)

# Container image with PyTorch, CUDA, and Qwen-TTS dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch>=2.5.0",
        "torchaudio",
        "numpy<2",
        "soundfile",
        "pydub",
        "scipy",
        "accelerate",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "huggingface_hub",
        "pydantic",
    )
    .pip_install("qwen-tts")
    .env({
        "HF_HOME": f"{CACHE_DIR}/huggingface",
        "TORCH_HOME": f"{CACHE_DIR}/torch",
        "PYTHONUNBUFFERED": "1",
    })
)

# ============================================================================
# 2. FASTAPI WEB APPLICATION & STATIC UI
# ============================================================================

from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

web_app = FastAPI(
    title="Qwen3 Voice Design Service",
    description="Natural language instruction-driven voice synthesis powered by Qwen3-TTS on NVIDIA L4 GPU",
    version="1.0.0",
)

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]


class VoiceDesignRequest(BaseModel):
    """Request model matching index-tts-vllm's /api/design-voice schema."""
    text: str = Field(..., description="Text content to synthesize into speech")
    voice_description: str = Field(
        ...,
        description="Natural language description of desired voice timbre, age, accent, and mood",
        examples=["A calm, gentle female narrator with a clear British accent and soothing tone"]
    )
    language: str = Field(default="Auto", description="Target language")
    output_format: Literal["mp3", "wav"] = Field(default="mp3", description="Audio container format")
    temperature: float = Field(default=0.9, ge=0.1, le=2.0)
    top_p: float = Field(default=1.0, ge=0.1, le=1.0)
    top_k: int = Field(default=50, ge=1, le=200)
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0)
    max_new_tokens: int = Field(default=2048, ge=64, le=4096)


class OpenAITTSRequest(BaseModel):
    """OpenAI-compatible /v1/audio/speech schema."""
    model: str = Field(default="qwen3-voice-design")
    input: str = Field(..., description="Input text to speak")
    voice: str = Field(..., description="Natural language prompt describing voice design")
    response_format: Literal["mp3", "wav"] = Field(default="mp3")
    speed: Optional[float] = Field(default=1.0)


# Embedded Modern HTML Web Playground
HTML_STUDIO_UI = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Design Studio - Qwen3-TTS (L4 GPU)</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0b0c10;
            --bg-secondary: #13161f;
            --bg-card: #1c2130;
            --border: #2d354a;
            --accent: #8b0000;
            --accent-glow: #e63946;
            --crimson: #c1121f;
            --text-main: #f0f3f8;
            --text-muted: #9ba3b4;
            --radius: 12px;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: radial-gradient(circle at 50% 0%, #1d101a 0%, var(--bg-primary) 70%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 30px 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        header {
            text-align: center;
            margin-bottom: 30px;
            position: relative;
        }
        h1 {
            font-family: 'Cinzel', serif;
            font-size: 2.3rem;
            color: #f7e1d7;
            letter-spacing: 1px;
            text-shadow: 0 0 20px rgba(230, 57, 70, 0.4);
            margin-bottom: 8px;
        }
        .subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
        }
        .badge {
            display: inline-block;
            background: rgba(193, 18, 31, 0.2);
            color: #ff8b94;
            border: 1px solid rgba(230, 57, 70, 0.4);
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 8px;
        }
        .card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            margin-bottom: 24px;
        }
        .form-group { margin-bottom: 20px; }
        label {
            display: block;
            font-weight: 600;
            font-size: 0.88rem;
            margin-bottom: 8px;
            color: #e2e8f0;
        }
        textarea, select, input {
            width: 100%;
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-main);
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 0.95rem;
            font-family: inherit;
            transition: all 0.2s ease;
        }
        textarea:focus, select:focus, input:focus {
            outline: none;
            border-color: var(--accent-glow);
            box-shadow: 0 0 10px rgba(230, 57, 70, 0.3);
        }
        .preset-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
        }
        .tag-btn {
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 5px 12px;
            border-radius: 16px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tag-btn:hover {
            background: var(--crimson);
            color: #fff;
            border-color: var(--accent-glow);
        }
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .collapsible-toggle {
            cursor: pointer;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-bottom: 12px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .collapsible-content {
            display: none;
            padding-top: 10px;
            border-top: 1px dashed var(--border);
            margin-bottom: 16px;
        }
        .collapsible-content.open { display: grid; }
        button.btn-primary {
            width: 100%;
            background: linear-gradient(135deg, #a71d2a 0%, #d62828 100%);
            border: none;
            color: white;
            padding: 14px 20px;
            border-radius: 8px;
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(214, 40, 40, 0.4);
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        button.btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(214, 40, 40, 0.6);
        }
        button.btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .player-container {
            display: none;
            margin-top: 24px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }
        audio {
            width: 100%;
            margin-top: 10px;
            border-radius: 8px;
        }
        .stats-bar {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 8px;
        }
        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .api-docs-link {
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 16px;
        }
        .api-docs-link a { color: #ff8b94; text-decoration: none; }
        .api-docs-link a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>VOICE DESIGN STUDIO</h1>
            <div class="subtitle">Natural Language Voice Synthesis with Qwen3-TTS</div>
            <div class="badge">NVIDIA L4 Acceleration • 1.7B Parameter Engine</div>
        </header>

        <div class="card">
            <div class="form-group">
                <label>Voice Description (Prompt / Persona)</label>
                <textarea id="voice_desc" rows="3" placeholder="Describe the voice: timbre, gender, age, emotion, accent, mood...">A gentle, articulate female voice with a warm, melodic British accent, narrating a mysterious Victorian tale with calm confidence.</textarea>
                <div class="preset-tags">
                    <span class="tag-btn" onclick="setPreset('gentle_narrator')">📖 Victorian Narrator</span>
                    <span class="tag-btn" onclick="setPreset('eldritch_mystic')">🔮 Eldritch Whisper</span>
                    <span class="tag-btn" onclick="setPreset('energetic_podcaster')">🎙️ Energetic Host</span>
                    <span class="tag-btn" onclick="setPreset('wise_elder')">📜 Wise Elder</span>
                    <span class="tag-btn" onclick="setPreset('cyber_ai')">🤖 Calm Synthesizer</span>
                </div>
            </div>

            <div class="form-group">
                <label>Text to Synthesize</label>
                <textarea id="text" rows="4" placeholder="Enter text to synthesize...">The crimson moon hung high above the Backlund skyline, veiled in thin mist. A sudden chill swept through the dim gaslit streets, carrying the faint scent of old parchment and sea salt.</textarea>
            </div>

            <div class="grid-2">
                <div class="form-group">
                    <label>Language</label>
                    <select id="language">
                        <option value="Auto" selected>Auto Detect</option>
                        <option value="English">English</option>
                        <option value="Chinese">Chinese (中文)</option>
                        <option value="Japanese">Japanese (日本語)</option>
                        <option value="Korean">Korean (한국어)</option>
                        <option value="French">French (Français)</option>
                        <option value="German">German (Deutsch)</option>
                        <option value="Spanish">Spanish (Español)</option>
                        <option value="Italian">Italian (Italiano)</option>
                        <option value="Russian">Russian (Русский)</option>
                        <option value="Portuguese">Portuguese (Português)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Output Format</label>
                    <select id="output_format">
                        <option value="mp3" selected>MP3 (192 kbps)</option>
                        <option value="wav">WAV (24kHz Studio Lossless)</option>
                    </select>
                </div>
            </div>

            <div class="collapsible-toggle" onclick="toggleParams()">
                <span>⚙️ Advanced Sampling Parameters</span>
                <span id="params-arrow">▼</span>
            </div>
            <div id="collapsible-params" class="collapsible-content grid-2">
                <div class="form-group">
                    <label>Temperature (<span id="val_temp">0.9</span>)</label>
                    <input type="range" id="temperature" min="0.1" max="1.5" step="0.05" value="0.9" oninput="document.getElementById('val_temp').innerText=this.value">
                </div>
                <div class="form-group">
                    <label>Top-P (<span id="val_topp">1.0</span>)</label>
                    <input type="range" id="top_p" min="0.1" max="1.0" step="0.05" value="1.0" oninput="document.getElementById('val_topp').innerText=this.value">
                </div>
                <div class="form-group">
                    <label>Top-K (<span id="val_topk">50</span>)</label>
                    <input type="range" id="top_k" min="1" max="100" step="1" value="50" oninput="document.getElementById('val_topk').innerText=this.value">
                </div>
                <div class="form-group">
                    <label>Repetition Penalty (<span id="val_rep">1.05</span>)</label>
                    <input type="range" id="repetition_penalty" min="1.0" max="1.5" step="0.01" value="1.05" oninput="document.getElementById('val_rep').innerText=this.value">
                </div>
            </div>

            <button id="gen-btn" class="btn-primary" onclick="generateVoice()">
                <div class="spinner" id="spinner"></div>
                <span id="btn-text">Synthesize Designed Voice</span>
            </button>

            <div class="player-container" id="player-box">
                <label>Generated Audio</label>
                <audio id="audio-player" controls autoplay></audio>
                <div class="stats-bar">
                    <span id="stat-latency">Latency: --</span>
                    <span id="stat-format">Format: --</span>
                </div>
            </div>
        </div>

        <div class="api-docs-link">
            Integration API available at <a href="/docs" target="_blank">Swagger OpenAPI Docs (/docs)</a> • Compatible with <code>/api/design-voice</code> & <code>/v1/audio/speech</code>
        </div>
    </div>

    <script>
        const PRESETS = {
            gentle_narrator: "A gentle, articulate female voice with a warm, melodic British accent, narrating a mysterious Victorian tale with calm confidence.",
            eldritch_mystic: "A deep, raspy, echoing male voice speaking in low, solemn cadences, tinged with age, ancient knowledge, and restrained emotion.",
            energetic_podcaster: "A vibrant, enthusiastic young male host with clear modern diction, lively pacing, and upbeat conversational warmth.",
            wise_elder: "A grandfatherly, comforting elderly man's voice, soft-spoken with rich texture, gentle pauses, and grandfatherly affection.",
            cyber_ai: "A serene, synthetic female voice with perfect articulation, crisp syllables, neutral emotional inflection, and smooth futuristic resonance."
        };

        function setPreset(key) {
            if (PRESETS[key]) {
                document.getElementById('voice_desc').value = PRESETS[key];
            }
        }

        function toggleParams() {
            const el = document.getElementById('collapsible-params');
            const arrow = document.getElementById('params-arrow');
            el.classList.toggle('open');
            arrow.innerText = el.classList.contains('open') ? '▲' : '▼';
        }

        async function generateVoice() {
            const btn = document.getElementById('gen-btn');
            const spinner = document.getElementById('spinner');
            const btnText = document.getElementById('btn-text');
            const playerBox = document.getElementById('player-box');
            const audioPlayer = document.getElementById('audio-player');
            const statLatency = document.getElementById('stat-latency');
            const statFormat = document.getElementById('stat-format');

            const payload = {
                text: document.getElementById('text').value.trim(),
                voice_description: document.getElementById('voice_desc').value.trim(),
                language: document.getElementById('language').value,
                output_format: document.getElementById('output_format').value,
                temperature: parseFloat(document.getElementById('temperature').value),
                top_p: parseFloat(document.getElementById('top_p').value),
                top_k: parseInt(document.getElementById('top_k').value),
                repetition_penalty: parseFloat(document.getElementById('repetition_penalty').value)
            };

            if (!payload.text) {
                alert("Please enter text to synthesize.");
                return;
            }
            if (!payload.voice_description) {
                alert("Please describe the voice.");
                return;
            }

            btn.disabled = true;
            spinner.style.display = 'block';
            btnText.innerText = "Designing & Synthesizing Voice...";

            const startTime = performance.now();

            try {
                const res = await fetch('/api/design-voice', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!res.ok) {
                    const err = await res.json().catch(() => ({ detail: res.statusText }));
                    throw new Error(err.detail || 'Generation failed');
                }

                const blob = await res.blob();
                const duration = ((performance.now() - startTime) / 1000).toFixed(2);

                const audioUrl = URL.createObjectURL(blob);
                audioPlayer.src = audioUrl;
                playerBox.style.display = 'block';
                statLatency.innerText = `Generation Time: ${duration}s`;
                statFormat.innerText = `Format: ${payload.output_format.toUpperCase()} (${(blob.size / 1024).toFixed(1)} KB)`;
                audioPlayer.play();
            } catch (err) {
                alert("Error generating speech: " + err.message);
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
                btnText.innerText = "Synthesize Designed Voice";
            }
        }
    </script>
</body>
</html>
"""


@web_app.get("/", response_class=HTMLResponse)
async def serve_studio_ui():
    """Serve the interactive Voice Design Studio web interface."""
    return HTMLResponse(content=HTML_STUDIO_UI)


@web_app.get("/api/design-voice/languages")
async def get_languages():
    """Return supported languages for Voice Design."""
    return {"languages": SUPPORTED_LANGUAGES}


@web_app.get("/api/design-voice/status")
async def get_status():
    """Health check and GPU status."""
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    vram_alloc = (
        torch.cuda.memory_allocated(0) / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    vram_reserved = (
        torch.cuda.memory_reserved(0) / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    return {
        "status": "ready",
        "model": MODEL_REPO_ID,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "gpu_name": gpu_name,
        "vram_allocated_gb": round(vram_alloc, 2),
        "vram_reserved_gb": round(vram_reserved, 2),
    }


# ============================================================================
# 3. CORE SERVICE IMPLEMENTATION ON NVIDIA L4
# ============================================================================

@app.cls(
    image=image,
    gpu="L4",
    min_containers=0,
    max_containers=1,
    timeout=600,
    scaledown_window=300,  # Scale to zero after 5 minutes of inactivity
    volumes={CACHE_DIR: cache_volume},
)
class VoiceDesignService:
    """Serverless Voice Design Service deployed on NVIDIA L4 GPU."""

    @modal.enter()
    def load_model(self):
        """Initialize the Qwen3-TTS Voice Design model into GPU memory once per container."""
        import torch
        from huggingface_hub import snapshot_download
        from qwen_tts import Qwen3TTSModel

        print(f"🚀 Initializing Voice Design Service on {torch.cuda.get_device_name(0)}...")

        local_model_path = Path(MODEL_CACHE_DIR)
        if not local_model_path.exists() or not any(local_model_path.iterdir()):
            print(f"📥 Downloading {MODEL_REPO_ID} to persistent cache: {local_model_path}...")
            local_model_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=MODEL_REPO_ID,
                local_dir=str(local_model_path),
                local_dir_use_symlinks=False,
            )
            # Commit the newly downloaded weights to Modal volume
            cache_volume.commit()
            print("✅ Model downloaded and committed to volume successfully.")
        else:
            print(f"📦 Loading pre-cached weights from {local_model_path}...")

        # Load Qwen3TTSModel in bfloat16 for Ada Lovelace / L4 acceleration
        self.model = Qwen3TTSModel.from_pretrained(
            str(local_model_path),
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        print("🎉 Qwen3 Voice Design model loaded and ready for inference!")

    def _convert_to_audio_bytes(
        self,
        waveform,
        sample_rate: int,
        output_format: str = "mp3",
    ) -> tuple[bytes, str]:
        """Convert float numpy waveform to MP3 or WAV bytes."""
        import numpy as np
        import soundfile as sf
        from pydub import AudioSegment

        # Normalize waveform if needed
        wav_data = np.asarray(waveform)
        if wav_data.dtype != np.int16:
            # Clip between -1.0 and 1.0, scale to int16
            wav_data = np.clip(wav_data, -1.0, 1.0)
            wav_data = (wav_data * 32767.0).astype(np.int16)

        if output_format.lower() == "mp3":
            try:
                audio_segment = AudioSegment(
                    wav_data.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=wav_data.dtype.itemsize,
                    channels=1 if len(wav_data.shape) == 1 else wav_data.shape[1],
                )
                buf = io.BytesIO()
                audio_segment.export(buf, format="mp3", bitrate="192k")
                return buf.getvalue(), "audio/mpeg"
            except Exception as exc:
                print(f"⚠️ MP3 export failed ({exc}), falling back to WAV...")

        buf = io.BytesIO()
        sf.write(buf, wav_data, sample_rate, format="WAV")
        return buf.getvalue(), "audio/wav"

    @modal.method()
    def generate_speech(
        self,
        text: str,
        voice_description: str,
        language: str = "Auto",
        output_format: str = "mp3",
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 2048,
    ) -> tuple[bytes, str, float]:
        """Direct inference method callable remotely or locally."""
        t0 = time.time()
        print(
            f"[VoiceDesign] Request: lang={language}, format={output_format}, "
            f"text_len={len(text)}, desc_len={len(voice_description)}"
        )

        gen_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
        }

        wavs, sr = self.model.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            **gen_kwargs,
        )

        audio_bytes, media_type = self._convert_to_audio_bytes(
            waveform=wavs[0],
            sample_rate=sr,
            output_format=output_format,
        )
        latency = round(time.time() - t0, 3)
        print(f"[VoiceDesign] Generated {len(audio_bytes)} bytes in {latency}s")
        return audio_bytes, media_type, latency

    @modal.asgi_app()
    def web(self):
        """Bind FastAPI endpoints directly to this instance."""
        service = self

        @web_app.post("/api/design-voice")
        async def design_voice(request: VoiceDesignRequest):
            """Drop-in endpoint compatible with index-tts-vllm."""
            try:
                audio_bytes, media_type, latency = service.generate_speech.local(
                    text=request.text,
                    voice_description=request.voice_description,
                    language=request.language,
                    output_format=request.output_format,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    max_new_tokens=request.max_new_tokens,
                )
                return Response(
                    content=audio_bytes,
                    media_type=media_type,
                    headers={
                        "X-Generation-Time-Seconds": str(latency),
                        "X-Voice-Design-Engine": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    },
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

        @web_app.post("/v1/audio/speech")
        async def openai_speech(request: OpenAITTSRequest):
            """OpenAI-compatible speech endpoint."""
            try:
                audio_bytes, media_type, latency = service.generate_speech.local(
                    text=request.input,
                    voice_description=request.voice,
                    language="Auto",
                    output_format=request.response_format,
                )
                return Response(
                    content=audio_bytes,
                    media_type=media_type,
                    headers={"X-Generation-Time-Seconds": str(latency)},
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return web_app


# ============================================================================
# 4. CLI LOCAL ENTRYPOINT FOR RAPID TESTING
# ============================================================================

@app.local_entrypoint()
def main(
    text: str = "The crimson moon illuminated the misty streets of Backlund.",
    prompt: str = "A deep, gentle male voice with a slight British accent and calm demeanor",
    language: str = "English",
    output: str = "output_voice_design.mp3",
    format: str = "mp3",
):
    """
    Test the Voice Design service remotely via Modal CLI.

    Usage:
        modal run deploy_voice_design_modal.py
        modal run deploy_voice_design_modal.py --prompt "Gentle female narrator" --text "Hello world"
    """
    print("=" * 70)
    print("🎙️ Testing Voice Design on Modal (L4 GPU)")
    print(f"📝 Text:   {text}")
    print(f"🎨 Voice:  {prompt}")
    print(f"🌐 Lang:   {language}")
    print(f"💾 Output: {output}")
    print("=" * 70)

    service = VoiceDesignService()
    audio_bytes, media_type, latency = service.generate_speech.remote(
        text=text,
        voice_description=prompt,
        language=language,
        output_format=format,
    )

    out_path = Path(output)
    out_path.write_bytes(audio_bytes)
    print(f"✅ Success! Saved {len(audio_bytes)} bytes to {out_path.resolve()} (Latency: {latency}s)")
