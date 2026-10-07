# Application architecture

The supported command is `python fastapi_webui_v2.py`. HTTP behavior and model
workflows remain in `fastapi_webui_v2_impl.py`; the application package owns
server assembly, configuration, backend adapters, and shared infrastructure.

## Structure

| Path | Responsibility |
| --- | --- |
| `fastapi_webui_v2.py` | Stable command/import entry point |
| `indextts_web/main.py` | Uvicorn startup |
| `indextts_web/app.py` | Sole FastAPI factory, static assets, health route, lifespan |
| `indextts_web/api/__init__.py` / `route_groups.py` | Group and validate the production route inventory |
| `indextts_web/config.py` / `gpu_profiles.py` | Typed CLI settings and hardware-aware runtime configuration |
| `indextts_web/runtime.py` | Runtime settings, backend registry, concurrency, production-module reference |
| `indextts_web/services/tts/` | IndexTTS 2.0, IndexTTS 2.5, and Confucius request/lifecycle adapters |
| `indextts_web/services/translation/` | Subtitle parsing, isolated Qwen ASR worker, MOSS client/runtime |
| `indextts_web/infrastructure/` | GPU coordination/memory, Modal runtime, atomic JSON persistence, concurrency, callable compatibility |
| `fastapi_webui_v2_impl.py` | Production endpoint inventory, inference orchestration, translation sessions/artifacts |
| `indextts/` | Model inference implementations and upstream model components |
| `index_new.html` / `static/` | Modern WebUI markup, CSS, JavaScript |
| `tests/` | CPU behavior, deployment, route, streaming, and frontend contracts |

Importing configuration and service contracts does not initialize CUDA or
start services. App assembly loads the production module; model initialization
and mutable runtime setup occur in FastAPI lifespan.

## HTTP assembly and runtime

The production module's `app` is an `APIRouter` containing endpoint
definitions. `create_app()` constructs the single FastAPI application, adds
`/health` and static assets, groups endpoint routes, and installs lifespan.

`route_groups.py` classifies internal snapshot operations, TTS, translation,
speakers/voice design, video/cookies, Stable Audio, model management, utilities,
and UI routes. Assembly rejects unclassified paths and duplicate method/path
pairs. Route tests verify the public inventory and real application assembly.

`app.state.runtime` exposes the objects actually used by the application.
Translation sessions, persisted manifests, and media artifacts remain owned by
the production workflow; there is no separate unused repository/orchestrator
layer.

## TTS and GPU lifecycle

The backend registry selects one `SynthesisRequest` interface for IndexTTS
2.0, IndexTTS 2.5 vLLM-Omni, and Confucius. `index25_manager.py` owns the 2.5
subprocess, OpenAI-compatible upstream requests, sentence batching, and WAV
assembly without importing vLLM-Omni into the WebUI environment.

GPU profiles resolve engine budgets, batch limits, and active synthesis
capacity from reported VRAM. Shared admission coordinates synthesis and model
sleep/wake operations. Qwen ASR uses a separate interpreter; MOSS has an
independent service lifecycle. Optional models can retain allocations after
TTS engines sleep. See [GPU_DEPLOYMENT.md](GPU_DEPLOYMENT.md) for configuration
and the hardware validation requirements.

## Frontend

`index_new.html` contains markup; styles live in `static/css/app.css`.
Ordered deferred scripts under `static/js/` separate core, Stable Audio,
video, speakers, synthesis, translation, and bootstrap behavior. Classic
scripts share their top-level declarations, so order is part of the contract.

The server fills the `chunk-split-min-silence-ms` HTML meta value; static
JavaScript reads it instead of embedding a server template expression.
Completed one-shot extraction scripts are removed; edit the maintained assets
directly.

## Retained legacy interfaces

| File | Purpose |
| --- | --- |
| `api_server.py` | IndexTTS 1.x API, including `/audio/speech` and `/audio/voices`; Docker `APP_SERVER=legacy-api` |
| `api_example.py` | Manual client example for the legacy API |
| `simple_test.py` | Manual HTTP concurrency benchmark |
| `convert_hf_format.sh` / model conversion scripts | Checkpoint conversion for supported model formats |

The retired Gradio launchers and older unserved HTML template are removed.
Use the modern WebUI for speaker presets, synthesis, and translation. Git
history retains the removed prototypes and completed migration scripts.

## Compatibility and verification

Preserve or explicitly version route methods/paths, status codes/JSON fields,
`CHUNK`/`KEEPALIVE` binary frames, translation manifests/artifact names,
supported CLI flags, and Docker/Modal launch commands. Changes to sampling,
duration matching, concurrency, or encoding need behavior-specific validation.

```bash
pip install -e '.[dev]'
python -m unittest discover -s tests -v
ruff check indextts_web tests fastapi_webui_v2.py
python -m compileall -q indextts_web tests fastapi_webui_v2.py fastapi_webui_v2_impl.py
for script in static/js/*.js; do node --check "$script"; done
```

CPU checks use fake models/services; they do not establish GPU inference or
snapshot correctness. Run real synthesis, streaming, backend switching, and
the deployment acceptance matrix in the model environment before release.
