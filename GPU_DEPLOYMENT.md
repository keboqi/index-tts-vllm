# GPU-aware deployment

## Modal

In `deploy_vllm_indextts_v2.py`, edit the `gpu=` argument on
`IndexTTSVllmServer` to one of these values:

```python
gpu="L4",            # 24 GB
gpu="L40S",          # 48 GB
gpu="RTX-PRO-6000",  # 96 GB
```

Use **one** value, then run the usual command from this repository:

```bash
modal deploy deploy_vllm_indextts_v2.py
```

The Modal app is named `audio-studio`, with persistent volumes
`audio-studio-app` and `audio-studio-cache`. Provision these volumes and
`custom-secret` using the deployment's normal setup process. Changing the
volume names does not migrate data from previously named volumes.
Model preparation is manual. For fresh volumes, run this before deployment:

```bash
modal run deploy_vllm_indextts_v2.py::prepare_model
```

Changing GPU does not require changing
vLLM fractions, compilation flags, or concurrency arguments. Detection runs
inside the allocated container before model loading. The `[GPU profile]` log
shows the detected device, usable capacity, and resolved settings.

Application code is copied from the local repository into the Modal image.
At startup, a container-local copy links to the existing persistent checkpoints,
outputs, presets, and optional backend checkouts. Redeploy application changes;
updating only the app Volume is insufficient. GPU/configuration/image changes
invalidate Modal snapshots, while Volume changes do not. See
[Modal's snapshot lifecycle](https://modal.com/docs/guide/memory-snapshots#when-are-memory-snapshots-updated).

The snapshot path requires `/health` to report a loaded model and a warmup to
produce nonempty audio before taking the snapshot. Restore verifies the GPU
architecture/capacity, wakes the engines, and checks readiness without repeating
warmup inference. Compiler caches and generated Omni
configs use separate paths for each resolved GPU profile and Modal image.

## Automatic settings

Fractions are calculated from **reported total VRAM**, not free memory or the
nominal GPU label. For example, an L4 reporting 22.49 GiB still receives a 6 GiB
GPT budget. These are initial memory budgets, not measured peak allocations or
performance guarantees.

| Setting | L4 / 24 GB class | L40S / 48 GB class | RTX PRO 6000 / 96 GB class |
| --- | --- | --- | --- |
| Reported capacity | 20–<32 GiB | 32–<64 GiB | ≥64 GiB |
| IndexTTS GPT budget | 6 GiB | 10 GiB | 15% |
| QwenEmotion budget | 3 GiB | 4 GiB | 5% |
| GPT / emotion max sequences | 4 / 1 | 16 / 4 | Existing vLLM defaults |
| GPT batched tokens | 2560* | 4096* | Existing vLLM default |
| vLLM eager mode | Enabled | Existing default | Existing default |
| S2Mel / Confucius compilation | Enabled | Enabled | Enabled |
| Active IndexTTS / translation synthesis | 1 / 1 | 4 / 4 | 100 / 100 |
| IndexTTS conditioning cache entries | 2 | 4 | 8 |
| Confucius T2S budget | 6 GiB | 10 GiB | 20% in Modal; 15% locally |
| Confucius max sequences | 4 | 16 | Existing default |
| IndexTTS 2.5 AR / S2Mel budgets | 4 / 7 GiB | 8 / 14 GiB | 30% / 30% |
| IndexTTS 2.5 AR / S2Mel max sequences | 4 / 1 | 16 / 4 | 32 / 16 |
| IndexTTS 2.5 parallel segments | 1 | 4 | 100 |

*Raised to the checkpoint context capacity when needed to support prefill;
the automatic profile does not shorten the GPT context. Emotion retains its
2048-token context and subtracts prompt length from the output-token budget.

All profiles enable Torch compilation by default, including Omni S2Mel DiT and
vocoder compilation. The smaller profiles reduce the CFM batch size.
Other stage settings, model contexts, attention backends, and sampling
parameters are retained. Modal's Confucius command uses this repository's
launcher adapter to forward engine options into its isolated environment.
Custom non-Modal Confucius commands must use the adapter if they need the
additional scheduler/eager options; the standard sibling script receives the
resolved memory fraction.

On the smaller profiles, normal synthesis and managed startup/wake operations
coordinate backend access: a backend transition waits for active synthesis,
then sleeps the other TTS engines. This does not unload all auxiliary PyTorch
models. HTTP admission remains independent of active GPU synthesis capacity.

## Overrides and local startup

The modern FastAPI launcher, `quickstart.sh`, and Docker WebUI entry point use
the same core resolver. Leave tuning settings unset for automatic behavior.
The legacy IndexTTS 1.x API remains on its existing startup path.

Explicit memory CLI arguments override environment settings; environment
settings override profile defaults:

| Engine | Memory environment variable | Advanced option prefix |
| --- | --- | --- |
| IndexTTS GPT | `GPU_MEMORY_UTILIZATION` | `INDEXTTS_VLLM_` |
| QwenEmotion | `QWENEMO_GPU_MEMORY_UTILIZATION` | `QWENEMO_VLLM_` |
| Confucius | `CONFUCIUS_VLLM_GPU_MEMORY_UTILIZATION` | `CONFUCIUS_VLLM_` |

Advanced suffixes are `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`,
`MAX_MODEL_LEN`, and `ENFORCE_EAGER`. Memory fractions must be finite and
strictly between zero and one; counts must be positive integers. These map to
[vLLM engine arguments](https://docs.vllm.ai/en/v0.10.2/configuration/engine_args.html).

Use `INDEXTTS_USE_TORCH_COMPILE=0/1`, `--use_torch_compile`, or
`--no-use_torch_compile` for explicit compilation control.
`INDEXTTS_GPU_WORK_CONCURRENCY`, `TRANSLATION_TTS_CONCURRENCY`,
`INDEXTTS_CONDITIONING_CACHE_SIZE`, and `INDEXTTS25_MAX_PARALLEL_SEGMENTS`
override application limits. Translation concurrency cannot exceed IndexTTS
concurrency. `INDEXTTS25_DEPLOY_CONFIG` selects an explicit Omni config, which
is left unchanged.

For Modal, set advanced overrides in the **remote container environment**, for
example through the configured Modal secret or image environment. Local shell
variables are not automatically transferred. No override is needed when
changing the `gpu=` declaration.

Inspect the allocated GPU without loading model weights:

```bash
python -m indextts_web.gpu_profiles
# Include --modal to inspect the Modal defaults for the large profile.
```

The probe respects `CUDA_VISIBLE_DEVICES`; the normal deployment uses logical
`cuda:0`. GPUs below 20 GiB fail with an explicit unsupported-capacity error.
Initial startup also checks free memory against both core engine budgets, an
estimated 8 GiB allowance for other models, and headroom. Advanced users can
adjust that allowance with `INDEXTTS_NON_VLLM_RESERVE_GIB`. This preflight is an
estimate and does not certify that every workload fits.

## Qwen3-ASR + OmniVAD

The Modal image installs `qwen-asr==0.0.6` and its required
`transformers==4.57.6` in `/opt/qwen-asr-venv`. That venv shares the image's
CUDA/audio/diarization packages while keeping its conflicting Transformers
version separate from TTS and MOSS. The image build checks the ASR imports.
This follows the package's requirement to use a separate environment when
dependencies conflict; see [Qwen3-ASR installation guidance](https://github.com/QwenLM/Qwen3-ASR#environment-setup).

`QWEN_OMNIVAD_PYTHON` selects the worker interpreter. The WebUI dispatches the
complete existing pipeline to that process and preserves segment timestamps,
speaker profiles, translation controls, and cache metadata. Logs still appear
in Modal. One ASR worker runs at a time per container. On L4/L40S profiles it
waits for active TTS, sleeps the managed TTS engines, and exits before another
TTS backend can wake. Cancellation and timeout also stop the worker. Models
are loaded again for each job; downloaded files remain cached.

The Modal ASR batch-size defaults are 1 / 4 / 20 for L4 / L40S / RTX PRO 6000.
Override with `QWEN_ASR_MAX_BATCH_SIZE`. `QWEN_OMNIVAD_WORKER_TIMEOUT` defaults
to 7200 seconds. This worker uses the Transformers ASR backend.

Qwen ASR and aligner downloads use `/persistent_app/checkpoints/qwen_omnivad`;
pipeline result caching uses `/persistent_cache/qwen_omnivad`. Existing
on-demand model downloads continue on first use. To apply the dependency fix,
pull the latest checkout and redeploy so Modal builds the updated image.
Rerunning `prepare_model` alone cannot install this interpreter into a running
deployment.

## MOSS model controls

The dedicated MOSS service used by Modal reports its state in Model Manager,
including before the model has been loaded. **Sleep** moves its weights to CPU
and clears its CUDA allocator cache; **Wake** moves them back to the configured
device. **Unload** drops the model and processor and clears the CUDA cache.
Unloaded MOSS remains listed with a **Load** button. Transcription also wakes
or reloads it automatically when needed. **Unload All** includes MOSS.

Inference and lifecycle changes share a lock in the MOSS process, so sleep or
unload waits for active transcription. Status remains responsive while that
work runs. These controls apply to `moss_transcribe_server.py`; an external
SGLang/OpenAI-compatible server without this lifecycle API is not managed.
Sleep/unload releases model allocations; a small CUDA context may remain in
the running service. Redeploy to install the updated service and UI; existing
checkpoints do not need another `prepare_model` run.

## Validation status

CPU tests cover capacity selection, override precedence, engine construction,
Modal command construction for all three GPUs, generated Omni configurations,
source isolation, readiness, and request coordination. Qwen worker tests run
real child processes with a fake pipeline to check dispatch, data transfer,
errors, cancellation, and timeouts without model downloads. They do not verify
ASR inference on a GPU. MOSS tests cover lazy loading, CPU offload, wake,
reference release, inference/lifecycle locking, Model Manager routing and UI
controls, using fake models without CUDA. Install the lightweight
test dependencies with `pip install -e '.[dev]'`, then run:

```bash
python -m unittest discover -s tests -v
ruff check indextts_web tests fastapi_webui_v2.py tools/extract_*.py tools/split_translation_asset.py
```

Actual L4/L40S/RTX PRO 6000 Modal cold starts, inference peaks, and snapshot
restores have **not yet been measured for this change**. Before production
rollout, test each GPU on a separate staging app with isolated writable
volumes: deploy, synthesize with and without emotion text, queue concurrent
requests, scale down, confirm snapshot restore, and synthesize again. Exercise
Confucius and IndexTTS 2.5 separately and while switching backends. Record peak
VRAM, startup/restore time, and valid nonempty output audio.

MOSS/ASR, Stable Audio, enhancement, and other optional model combinations need
separate memory measurements. A successful TTS deployment does not establish
that all optional models fit simultaneously on L4.
