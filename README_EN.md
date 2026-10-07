[中文](README.md) | English

# IndexTTS-vLLM

An IndexTTS 2.0 speech studio powered by vLLM, with speaker presets, emotion
controls, streaming audio, speech translation, and an editable segment workflow.
Optional integrations provide IndexTTS 2.5, Confucius4-TTS, Qwen3-TTS voice
design, Stable Audio 3 music/SFX, video downloads, and reference enhancement.

## Quick start

The setup script targets Linux with an NVIDIA CUDA GPU. It installs audio
utilities and Python dependencies, downloads IndexTTS 2.0 weights and HY-MT
translation weights, provisions optional sibling backend repositories, and
starts the WebUI at `http://localhost:8000`:

```bash
git clone https://github.com/keboqi/index-tts-vllm.git
cd index-tts-vllm
EXPORT_TUNNEL=0 bash quickstart.sh
```

Use `bash quickstart.sh --setup-only` to provision without starting the server.
Useful setup variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VENV_DIR` / `PYTHON_VERSION` | `.venv` / `3.12` | Main Python environment |
| `MODEL_DIR` | `checkpoints` | IndexTTS 2.0 weights |
| `INSTALL_CONFUCIUS` / `INSTALL_INDEXTTS25` | `1` / `1` | Set to `0` to skip optional backend checkout provisioning |
| `UPDATE_EXTERNAL_REPOS` | `1` | Set to `0` to keep existing sibling checkouts unchanged |
| `DOWNLOAD_MODEL` / `DOWNLOAD_HY_MT_MODEL` | `1` / `1` | Control checkpoint downloads |
| `SERVER_PORT` | `8000` | WebUI/API port |
| `EXPORT_TUNNEL` | `1` | Set to `0` to disable the optional Cloudflare tunnel |

For an existing model environment:

```bash
python fastapi_webui_v2.py --model_dir checkpoints --host 0.0.0.0 --port 8000
```

GPU memory budgets, scheduler limits, and synthesis concurrency are selected
from detected VRAM. Explicit CLI or environment settings override the profile.
Use `--use_torch_compile` / `--no-use_torch_compile` to control compilation.
The complete CLI definition lives in [indextts_web/config.py](indextts_web/config.py).

## Deployment

For Modal, edit `IndexTTSVllmServer`'s `gpu=` in
[deploy_vllm_indextts_v2.py](deploy_vllm_indextts_v2.py) to `"L4"`, `"L40S"`, or
`"RTX-PRO-6000"`. Prepare fresh persistent volumes before deploying:

```bash
modal run deploy_vllm_indextts_v2.py::prepare_model
modal deploy deploy_vllm_indextts_v2.py
```

See [GPU_DEPLOYMENT.md](GPU_DEPLOYMENT.md) for volumes/secrets, automatic
settings, overrides, model lifecycle controls, and snapshot validation. CPU
tests cover profile and deployment command behavior; actual GPU cold starts,
inference peaks, and snapshot restore still need validation for each profile.

Docker uses [Dockerfile](Dockerfile), [docker-compose.yaml](docker-compose.yaml),
and [entrypoint.sh](entrypoint.sh). Review `.env.example` before running
`docker compose up --build`; the default URL is `http://localhost:8000`.
Modern WebUI deployments use `APP_SERVER=web`
and converted IndexTTS 2.0 weights in `checkpoints`; the legacy API uses
`APP_SERVER=legacy-api` and IndexTTS 1.x weights. Optional features need their
own dependencies/checkpoints.
Keep the WebUI port distinct from the managed backend ports (`8001` for
Confucius, `8092` for IndexTTS 2.5); override the corresponding backend port if
you deliberately assign its default port to the WebUI.

## Optional backends and features

The default TTS backend is `index` (IndexTTS 2.0). Select `index25` or
`confucius` in the UI, per API request, or with `--tts_backend`.

| Backend | Checkout / environment | Local API | Behavior |
| --- | --- | --- | --- |
| IndexTTS 2.0 | This repository's model environment | Main WebUI | Emotion text/audio, duration controls, chunk streaming |
| IndexTTS 2.5 | `../index-tts-2.5-vllm-omni-experiment`; isolated Python 3.11 environment | `127.0.0.1:8092` | Lazy provisioning/startup; Chinese, English, Japanese, Spanish, Arabic |
| Confucius4-TTS | `../Confucius4-TTS`; managed sibling launcher | `127.0.0.1:8001` | Lazy startup; multilingual synthesis; IndexTTS emotion text controls ignored |

Set `--indextts25_repo_dir` / `--confucius_repo_dir` for custom checkout
locations. Each backend also supports `--*_start_command`, `--*_start_timeout`,
and `--*_request_timeout` for custom service setup. External synthesis streams
emit keepalives during startup; IndexTTS 2.5 returns completed audio because
its model is non-streaming. Managed backend switching sleeps/stops competing
TTS engines; it does not release every auxiliary model's GPU allocations.

The translation workflow supports MOSS Transcribe+Diarize (default), Gemini,
WhisperX, Qwen3-ASR + OmniVAD, and NVIDIA Parakeet. The local MOSS Docker manager
can be prepared with:

```bash
bash sglang_omni_moss_transcribe.sh deploy
```

It starts lazily on the first MOSS transcription request. Modal instead uses
the dedicated [moss_transcribe_server.py](moss_transcribe_server.py) service.

Install only the optional integrations needed for a manual deployment:

```bash
pip install -r requirements-optional.txt
# Optional alignment and additional ASR backends:
pip install whisperx 'nemo_toolkit[asr]'
```

Keep Qwen3-ASR in a separate environment: the pinned `qwen-asr` and `qwen-tts`
dependencies require different Transformers versions. Set
`QWEN_OMNIVAD_PYTHON` to that environment's Python executable. Do not install
`qwen-asr[vllm]` into the main environment, which pins `vllm==0.10.2`.
Modal configures `/opt/qwen-asr-venv` automatically; see
[ASR worker configuration](GPU_DEPLOYMENT.md#qwen3-asr--omnivad).

For local setup, provision an independent ASR environment:

```bash
uv venv --python 3.12 --seed .venv-qwen-asr
.venv-qwen-asr/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
.venv-qwen-asr/bin/python -m pip install 'qwen-asr==0.0.6' 'transformers==4.57.6' omnivad litai pydub librosa soundfile scipy google-genai json-repair
# Optional: install only the selected diarization backend's dependencies.
.venv-qwen-asr/bin/python -m pip install 'transformers==4.57.6' whisperx # Pyannote
.venv-qwen-asr/bin/python -m pip install 'transformers==4.57.6' 'nemo_toolkit[asr]' # Sortformer
QWEN_OMNIVAD_PYTHON="$PWD/.venv-qwen-asr/bin/python" .venv/bin/python fastapi_webui_v2.py
```

Diarization also requires access to the selected backend's checkpoints.

### Stable Audio 3

For manual setup, install the source package without replacing the existing
Torch/audio stack, then download the gated models after accepting their terms
and authenticating on Hugging Face:

```bash
pip install -U --no-deps --ignore-requires-python 'git+https://github.com/Stability-AI/stable-audio-tools.git'
pip install alias-free-torch dill einops-exts huggingface_hub importlib-resources nnAudio PyWavelets safetensors scipy soxr torchsde tqdm transformers v-diffusion-pytorch vector-quantize-pytorch

hf auth login
hf download stabilityai/stable-audio-3-medium --local-dir checkpoints/stable-audio-3/medium
hf download stabilityai/stable-audio-3-small-music --local-dir checkpoints/stable-audio-3/small-music
hf download stabilityai/stable-audio-3-small-sfx --local-dir checkpoints/stable-audio-3/small-sfx
```

Inference loads these local folders and does not need `HF_TOKEN` once weights
are downloaded. Model Manager provides load/sleep/wake/unload controls where
supported. Optional model combinations need separate GPU memory validation.

## API

The running application's `/docs` and `/openapi.json` expose the current API
schema; the WebUI's API tab describes interactive workflows. Main endpoints:

| Workflow | Endpoints |
| --- | --- |
| Synthesis | `/speak`, `/clone_voice`, `/speak_stream`, `/clone_voice_stream` |
| Speaker presets | `/add_speaker`, `/delete_speaker`, `/audio_roles`, `/api/speaker_preview/{speaker_name}` |
| Translation/editor | `/api/translate_audio`, `/api/translate_segments`, `/api/translate_generate_segments`, `/api/translate_segment_preview` |
| Long audio | `/api/translate_split_audio`, `/api/translate_generate_chunks`, `/api/translate_merge_chunks` |
| Music/SFX | `/api/stable-audio/models`, `/api/stable-audio/generate`, `/api/stable-audio/unload` |
| Voice design | `/api/design-voice`, `/api/design-voice/save-preset`, `/api/design-voice/languages`, `/api/design-voice/status` |
| Video/cookies | `/api/video_info`, `/api/video_download`, `/api/video_replace_audio`, `/api/cookies` |
| Readiness | `/health`, `/server_info` |

The retained IndexTTS 1.x [api_server.py](api_server.py) separately provides
the OpenAI-compatible `/audio/speech` and `/audio/voices` endpoints.

Register `my_speaker_preset` in the UI, then synthesize:

```bash
curl --fail http://127.0.0.1:8000/speak \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from IndexTTS.","name":"my_speaker_preset","tts_backend":"index"}' \
  --output output.mp3
```

Synthesis streaming uses binary frames, not SSE:
`CHUNK:{idx}:{size}:{MORE|LAST}\n{audio_bytes}` and optional
`KEEPALIVE:{size}\n{json}`. Clients must buffer partial headers and payloads
across transport reads and consume the declared byte count. Translation
progress endpoints use `text/event-stream`.

## Development

[ARCHITECTURE.md](ARCHITECTURE.md) maps the supported entry point, routers,
services, frontend, legacy interfaces, and compatibility rules.

Run development and deployment from the repository checkout. The editable
install below provides development tooling; standalone wheel deployment is
not supported because model code and application assets use the checkout layout.

```bash
pip install -e '.[dev]'
python -m unittest discover -s tests -v
ruff check indextts_web tests fastapi_webui_v2.py
python -m compileall -q indextts_web tests fastapi_webui_v2.py fastapi_webui_v2_impl.py
```

The CPU suite uses fake model/service adapters and needs no CUDA or
checkpoints. Check frontend JavaScript syntax with `node --check` and run
real inference/streaming/backend-switching smoke checks in the GPU environment
before release. Runtime GPU acceptance criteria are in
[GPU_DEPLOYMENT.md](GPU_DEPLOYMENT.md#validation-status).
