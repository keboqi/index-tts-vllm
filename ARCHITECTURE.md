# Application architecture

The WebUI/API uses a compatibility-preserving modular shell. Existing HTTP
contracts remain implemented by `fastapi_webui_v2_impl.py` while the public
entry point, settings, runtime services, routers, and frontend assets live in
small modules that can be migrated independently.

## Entry points

- `fastapi_webui_v2.py` is the stable command and import path. It contains no
  feature logic.
- `indextts_web.main` owns Uvicorn startup.
- `indextts_web.app.create_app()` assembles the application, feature routers,
  static assets, health endpoint, runtime container, and lifespan.
- `fastapi_webui_v2_impl.py` is the production compatibility implementation. New code
  should not add endpoints or infrastructure helpers to this file.

Importing `indextts_web`, `indextts_web.config`, or the service contracts does
not initialize CUDA, create output directories, or start managed services.
Mutable directories and models initialize inside FastAPI lifespan.

## HTTP routing

`indextts_web/api/` groups the public route inventory into health, internal
snapshot operations, TTS, translation, speakers, video/cookies, Stable Audio,
model management, utilities, and the UI.

The compatibility selectors fail application assembly if a legacy route is
unclassified or assigned to multiple feature groups.
`tests/test_route_contract.py` locks the original 60-route inventory.

## Services

`RuntimeContainer` exposes services through `app.state.runtime`.

The TTS registry owns backend selection. IndexTTS 2.0, the isolated IndexTTS
2.5 vLLM-Omni service, and Confucius implement one `SynthesisRequest` contract
and publish backend capabilities. `index25_manager.py` owns the 2.5 subprocess,
OpenAI-compatible speech payloads, sentence batching, and WAV assembly without
importing the vLLM-Omni environment into the WebUI process.
The production shared synthesis function delegates through this registry.

The translation package defines a stage-oriented `TranslationOrchestrator`, a
`SessionRepository` protocol, an isolated in-memory implementation, and a
contained `ArtifactStore`. The current orchestrator delegates the established
media pipeline while that large workflow remains compatibility-sensitive.

Infrastructure adapters isolate concurrency limits, filesystem containment,
atomic JSON writes, subprocess execution, and FFmpeg command construction.

## Frontend

`index_new.html` contains markup only. Styles live in `static/css/app.css`.
Deferred scripts under `static/js/` preserve the previous execution order while
separating core, Stable Audio, video, speakers, synthesis, translation
state/chunks/media/segments/speakers/requests, and bootstrap code.

The HTML meta value for `CHUNK_SPLIT_MIN_SILENCE_MS` is populated by the server;
application JavaScript reads that value instead of embedding a template
expression in a static file.

## Compatibility rules

Changes must preserve, or explicitly version:

- route methods and paths;
- status codes and JSON field names;
- `CHUNK` and `KEEPALIVE` binary framing;
- translation session manifests and artifact names;
- CLI flags consumed by `fastapi_webui_v2.py`;
- Docker and Modal launch commands.

Do not combine mechanical extraction with changes to model sampling, duration
matching, concurrency defaults, or audio encoding.

## Verification

CPU-only checks do not need FastAPI, CUDA, or model checkpoints:

```bash
python -m unittest discover -s tests -v
ruff check indextts_web tests fastapi_webui_v2.py tools/extract_*.py tools/split_translation_asset.py
python -m compileall -q indextts_web tests fastapi_webui_v2.py fastapi_webui_v2_impl.py
```

Run `node --check` over `static/js/*.js`, then run the GPU smoke suite in the
production model environment before deployment.
