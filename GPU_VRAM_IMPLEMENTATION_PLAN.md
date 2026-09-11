**Implementation plan: VRAM-aware vLLM startup**

Status: runtime implementation and CPU/Modal command tests are in place. Actual GPU deployment and snapshot validation remain pending. See [GPU_DEPLOYMENT.md](GPU_DEPLOYMENT.md) for the implemented settings and operator workflow. The design below records the original investigation and release acceptance criteria.

Implementation details that refine the original plan: the Modal image now ships this checkout's application source and uses container-local code linked to persistent data; Confucius engine options are forwarded by a repository-owned launcher adapter rather than a companion checkout patch. Small/medium profiles coordinate backend transitions and disable compilation. Numeric budgets remain candidates until the hardware acceptance matrix passes.

Investigated application revision: `84f82f1`. The local IndexTTS 2.5 checkout matches the deployment's pinned revision, `0a7d9aaeb9a0516c124669966aeed907e29b811d`.

The primary release target is the existing `deploy_vllm_indextts_v2.py` Modal deployment on L4 (24 GB), L40S (48 GB), and RTX PRO 6000 (96 GB). The operator manually edits the existing `gpu=` value in `@app.cls` and runs the normal Modal deployment command. Once the container starts, it detects the allocated GPU's VRAM and automatically chooses the corresponding engine, batch, concurrency, and compilation settings. Changing GPU must not require manually changing those startup arguments or setting a separate GPU-selection environment variable. Preserve the current RTX PRO 6000 behavior. Modal is the main production path and its complete deployment lifecycle on all three GPUs is a release gate. FastAPI, quickstart, and Docker must use the same profiles, but passing local startup tests alone does not establish completion. Validate IndexTTS 2.0 first, then certify the optional IndexTTS 2.5 and Confucius backends individually and during switching. Lower-memory cards and arbitrary combinations of optional models need separate measurements.

**1. Findings that determine the design**

| Area | Current behavior | Consequence |
| --- | --- | --- |
| `indextts/infer_vllm_v2.py:154,215,224` | GPT fraction defaults to `0.15`, QwenEmotion to `0.05`; VRAM detection only selects initialization order. Even the larger-GPU branch waits for GPT before submitting Qwen. | No actual VRAM-based engine sizing exists. Keep engine profiling sequential. |
| `indextts/infer_vllm_v2.py:1163` | QwenEmotion sets `max_model_len=2048`; neither engine explicitly limits sequences, batched tokens, or CUDA graphs. | vLLM chooses scheduler/graph defaults independently of application GPU capacity. |
| `indextts/infer_vllm_v2.py:271–373` | PyTorch GPT, W2V-BERT, codec, S2Mel, CAMPPlus, BigVGAN, and conditioning tensors load after the vLLM engines. | vLLM budgets must leave room for these weights and peak synthesis activations. |
| `indextts_web/infrastructure/concurrency.py:12` and `.env.example` | GPU work and translation concurrency default to 100. | A successful engine startup does not establish safe concurrent inference on 24 GB. |
| `indextts/infer_vllm_v2.py:329` | S2Mel cache setup receives a nominal batch size of 100, but `diffusion_transformer.py:184` calls it with `use_kv_cache=False`. | Do not claim that reducing this value frees a 100-request KV cache. Measure activation peaks and control real work admission. |
| `deploy_vllm_indextts_v2.py:237,1376,1460,1763` | Fixed memory fractions, fixed `RTX-PRO-6000`, forced Torch compilation, and 100 parallel 2.5 segments. | Automatic settings in the application would be overridden by deployment-generated arguments. |
| `deploy_vllm_indextts_v2.py:347,428,1607` | The image clones the application from GitHub; `prepare_model()` updates persistent code from a branch; runtime executes `/persistent_app`. | Editing/redeploying the local deployment script does not establish that the matching application/profile implementation reaches the container. |
| `deploy_vllm_indextts_v2.py:1309` | `_wait_ready()` accepts a response from `/server_info`, whose handler does not require TTS readiness. | Modal can accept application liveness before successful model initialization; use actual readiness and inference checks. |
| Pinned sibling `vllm_omni/deploy/indextts2_5.yaml` | Both stages use `0.3`; stage 0 allows 32 sequences; stage 1 allows 16, with `s2mel_cfm_batch_size=16` and compilation enabled. Its comment explicitly targets 96 GB. | IndexTTS 2.5 needs stage-specific configuration, including the physical S2Mel batch. |
| `fastapi_webui_v2_impl.py:4605,11349` | Existing switching sleeps selected vLLM engines. The main PyTorch models remain resident, and IndexTTS 2.0 initializes at application startup. | Sleeping vLLM alone does not free the entire backend footprint. Account for retained models when sizing another backend. |

Using nominal capacities for illustration, GPT's current budget falls from 14.4 GB on 96 GB to 3.6 GB on 24 GB; QwenEmotion falls from 4.8 GB to 1.2 GB. These are configured budgets, not measured usage. The models do not shrink with the GPU.

vLLM's utilization parameter is per engine; it is not a shared allocation coordinator. Lower sequence/token limits and reduced CUDA graph capture address different memory costs. Eager execution disables CUDA graphs. See the pinned [vLLM 0.10.2 engine arguments](https://docs.vllm.ai/en/v0.10.2/configuration/engine_args.html#cacheconfig) and [memory guide](https://docs.vllm.ai/en/v0.10.2/configuration/conserving_memory.html).

**2. Introduce one profile resolver**

Add an import-safe `indextts_web/gpu_profiles.py` containing hardware facts, requested overrides, resolved per-engine settings, and pure profile-selection/budget functions. Put runtime device probing in `indextts_web/infrastructure/gpu.py`.

- Probe the device actually used by the backend: total/free bytes, device identity, and compute capability. Respect `CUDA_VISIBLE_DEVICES`, including remapped indices/UUIDs; never add memory across unrelated GPUs. The current detector always examines logical device 0, even when the model constructor is given a different device.
- Resolve once during runtime startup before model allocation and before constructing the effective concurrency budget. Preserve CUDA-free imports of the settings/profile modules. Prefer NVML or `nvidia-smi` with verified device mapping; any Torch-based fallback must avoid initializing CUDA in a parent that later forks vLLM workers.
- Choose a capacity profile from total memory. Use free memory as an admission check, not as the denominator for `gpu_memory_utilization`. Recheck it before managed backend startup/wake; do not dynamically resize live engines.
- Automatic profile selection is the default: 20–<32 GiB selects the L4/24 GB class, 32–<64 GiB selects the L40S/48 GB class, and >=64 GiB selects the RTX PRO 6000/96 GB class. These are capacity classes, not claims that every GPU in a band has been validated. Devices below 20 GiB need explicit experimental settings; report an actionable result instead of selecting a high-memory default. Test actual reported capacities, which can be below nominal labels. Any optional profile override is for advanced tuning/testing; the normal deployment requires only the manual edit to `gpu=`.
- Preserve the existing memory flags and environment variables. Change unspecified values to `None`/auto so the resolver can distinguish a default from an override. Precedence: explicit CLI value > explicit environment value > selected profile. Add typed per-engine overrides for `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, and `enforce_eager`; keep GPT and emotion settings independent.
- Make compilation tri-state as well: explicit enable/disable > profile. Keep the current `--use_torch_compile` spelling and add an explicit disable form.
- Reject invalid/nonfinite fractions and incompatible scheduler settings clearly. A requested override remains authoritative, but an infeasible combination fails with the computed budget; it is not silently rewritten.
- Log device, selected profile, each setting's source, budgets in GiB and fractions, concurrency, graph/compile mode, and reserved headroom. Expose a resolved-config diagnostic without changing existing health fields or route contracts.

Budget model for each simultaneously resident set:

```text
sum(vLLM executor budgets)
  + retained non-vLLM weights and caches
  + measured peak synthesis activation allowance
  + safety headroom
  <= total device memory minus unrelated allocations

engine utilization = engine budget bytes / total device bytes
```

Measure the non-vLLM allowance first. Avoid double counting weights already included in an executor budget or allocations already represented in the free-memory observation. Use vLLM profiling results to verify that each engine still has enough KV cache for a supported request. A fraction is not a hard cap on every CUDA allocation in the application.

**3. Starting profile candidates for IndexTTS 2.0**

These are benchmark inputs, not validated L4 recommendations. Final budgets depend on the installed checkpoints and measured non-vLLM peaks.

| Setting | L4 / 24 GB class | L40S / 48 GB class | RTX PRO 6000 / 96 GB class |
| --- | --- | --- | --- |
| GPT executor budget | 6 GiB | 10 GiB | Existing `0.15` fraction |
| QwenEmotion executor budget | 3 GiB | 4 GiB | Existing `0.05` fraction |
| GPT `max_num_seqs` | 4 | 16 | Preserve current effective default |
| Emotion `max_num_seqs` | 1 | 4 | Preserve current effective default |
| GPT batched-token limit | At least its validated full context; initially `max(context, 2560)` | Initially `max(context, 4096)` | Preserve current effective default |
| Emotion batched-token limit | 2048 | 2048 | Preserve current effective default |
| GPT context | Preserve checkpoint-derived supported context | Same | Same |
| Emotion context | Keep 2048 | Keep 2048 | Keep 2048 |
| vLLM eager execution | On initially | Benchmark limited graph capture | Preserve current behavior |
| S2Mel Torch compilation | Off initially | Benchmark before enabling by default | Preserve launch-path behavior |
| Application GPU work / translation concurrency | 1 / 1 initially | 4 / 4 initially | Existing 100 / 100 |
| Conditioning cache entries | 2 | 4 | Existing 8 |

For an exactly 24 GiB device, the candidate budgets correspond to `0.25` and `0.125`. Compute actual fractions from detected bytes. At least 2 GiB or 10% of total memory, whichever is larger, is the initial safety-headroom target, in addition to the measured non-vLLM allowance. If this does not fit, revise measured budgets or residency; do not squeeze a model below its minimum viable allocation.

Do not shorten audio generation limits merely to fit a profile. GPT currently permits 2048 generated tokens, so a blanket `max_model_len=2048` could eliminate room for conditioning and text. Derive the required context from the checkpoint and actual prompt format. Validate Qwen's prompt-plus-output handling too: its current generation limit equals its entire 2048-token context. Any output-limit correction must be explicit and covered by long-input tests.

Keep model precision, sampling, attention behavior, and audio encoding unchanged initially. Quantization, FP8 cache, and CPU offload are separate experiments if measurements show they are necessary.

**4. Wire the resolver through every supported launch path**

| Files | Planned change |
| --- | --- |
| `indextts_web/config.py`, new profile/probe modules | Parse optional overrides; detect hardware and produce a validated runtime configuration. |
| `indextts/infer_vllm_v2.py` | Pass resolved engine arguments to both `AsyncEngineArgs` constructors; retain a profile fallback for direct constructor callers; keep sequential profiling and sleep hooks. |
| `indextts_web/infrastructure/concurrency.py`, `fastapi_webui_v2_impl.py` | Construct semaphores and derived limits from the resolved profile during startup. Update the legacy aliases together; avoid leaving import-time limits at 100. Keep new policy logic in small modules. |
| `entrypoint.sh`, `.env.example`, `docker-compose.yaml` | Stop forcing default numbers into CLI arguments; omit unset optional arguments. Comment out example overrides so copied examples select auto. Preserve the legacy API's existing behavior and avoid passing unsupported new flags to it. |
| `quickstart.sh` | Remove unconditional compilation enablement and let the runtime profile decide unless the operator specifies it. |
| `deploy_vllm_indextts_v2.py` | Keep the existing manually editable `gpu="RTX-PRO-6000"` declaration and document `"L4"` and `"L40S"` alternatives. Detect VRAM and resolve settings on the allocated GPU at container startup. Remove forced memory/compile/segment values that mask auto settings. |
| READMEs and tests | Document auto mode, explicit overrides, supported profiles, and the performance tradeoff. |

Modal's HTTP input limit and Uvicorn's connection limit are distinct from physical GPU concurrency. Preserve useful HTTP admission/keepalive capacity while the application bounds active GPU work; do not reduce every concurrency setting to one.

Validate the existing CUDA/PyTorch/vLLM/attention-kernel stack on Ada as well as Blackwell. The repo uses separate vLLM versions: 0.10.2 for the main app, 0.16.0 in the inspected Confucius checkout, and 0.27.0 for 2.5. Build version-specific adapters and verify argument support in each environment. Do not change CUDA versions simply because VRAM is smaller.

Keep snapshot sleep/wake behavior. Test cold start and snapshot restore on each GPU profile. GPU/configuration changes require a fresh snapshot; Modal documents this in its [snapshot lifecycle guidance](https://modal.com/docs/guide/memory-snapshots#when-are-memory-snapshots-updated).

**4a. Modal deployment contract and required changes**

The operator edits just the existing GPU declaration in `deploy_vllm_indextts_v2.py`:

```python
@app.cls(
    image=image,
    gpu="L4",  # Manually choose "L4", "L40S", or "RTX-PRO-6000".
    # Existing resource, volume, secret, and snapshot configuration follows.
)
```

Then run `modal deploy deploy_vllm_indextts_v2.py` as usual. This excerpt illustrates the intended workflow; it is not a standalone replacement for the full decorator. Automatic startup tuning is the implementation being planned. Leave `gpu="RTX-PRO-6000"` as the checked-in default. Keep one manually selected GPU type per deployment.

| Operator's `gpu=` value | Nominal capacity | Automatic runtime profile |
| --- | --- | --- |
| `"L4"` | 24 GB | Small-memory budgets and batches; conservative compilation policy |
| `"L40S"` | 48 GB | Intermediate budgets, batches, and concurrency |
| `"RTX-PRO-6000"` | 96 GB | Preserve the current high-memory deployment behavior |

Use detected total VRAM for budget calculations; the table's capacity labels are not a substitute for probing the allocated device. No profile or utilization override is required for any of these three normal deployment choices.

| Deployment point | Required behavior |
| --- | --- |
| Module configuration and `@app.cls` | Preserve the manually edited `gpu=` resource definition. Keep remote GPU detection out of import/image-build code. Auto tuning requires no additional operator configuration. If advanced engine overrides are supplied, explicitly transport only those options into the container; do not assume local shell variables become remote environment variables or copy the full local environment. |
| Image and `prepare_model()` | Include a matching application revision and profile implementation in the image/persistent runtime. Pin the application revision used for this rollout, verify it after preparation, and make it part of deployment configuration. Replace the branch-head update for this path with the selected revision. Fail early if the mounted application is stale. For testing unpublished changes, package an explicit source snapshot into an isolated staging deployment. Preserve weights, outputs, and speaker data. |
| `_configure_persistent_runtime()` | Prepare paths and verify the code/config schema before importing the resolver from the deployed application. Remove unconditional Confucius utilization/compile assignments; they must not overwrite supplied overrides. |
| `IndexTTSVllmServer.start()` | After runtime preparation and before model allocation, probe the allocated GPU and resolve one configuration. Pass it explicitly to the WebUI and both managed backend command builders. Include a schema version and fingerprint in any serialized config passed to subprocesses. The WebUI must consume this result instead of independently choosing potentially different defaults. |
| `_build_webui_command()` | Consume the resolved GPU/translation/cache limits and both engine settings; remove unconditional `--use_torch_compile`, fixed fractions, and `--indextts25_max_parallel_segments 100`. Preserve current 96 GB Modal compile behavior through its profile. |
| `_build_confucius_start_command()` | Use the resolved Confucius budget, supported engine arguments, and compile policy. The current generated custom command has its own fixed values and must be updated together with the environment. |
| `_build_indextts25_start_command()` | Pass the generated profile-specific YAML to `--deploy-config`, using the isolated 2.5 executable. Verify the file exists and matches the selected profile before launching. An environment-only override cannot replace the currently hardcoded CLI path. |
| `_wait_ready()` and snapshot warmup | Require `/health` to report `ready=true`, and require real warmup success before sleep/snapshot. Propagate failed initialization/inference instead of returning an apparently successful warmup response. Preserve existing public response fields; add strict internal validation as needed. |
| `wake_up()` | Confirm the restored profile/code fingerprint and compatible device capacity, wake engines, then verify model readiness and a small synthesis. Do not reuse the pre-start free-memory formula against already-loaded resident weights or silently change engine settings in restored state. |
| Cache/config paths | Key generated configs and compiled artifacts by backend, GPU architecture, profile, relevant library versions, and code revision. Keep compatible downloaded weights shared. Avoid one mutable YAML/cache location being overwritten by L4, L40S, and RTX PRO 6000 deployments. |
| `legacy_serve_without_snapshot()` | Keep its internal helper using the same resolver and command builders. It is currently undecorated; do not describe it as an available deployed Modal endpoint. |

Use explicit Modal environment configuration for transported options, as supported by [Modal's image environment API](https://modal.com/docs/guide/environment_variables#container-image-environment-variables). Include the code revision/profile schema in the deployed configuration so updates are visible to snapshot invalidation. Changes only to a mounted Volume do not invalidate existing snapshots; see [Modal's snapshot update rules](https://modal.com/docs/guide/memory-snapshots#when-are-memory-snapshots-updated).

Add `tests/test_modal_gpu_profiles.py` with CPU-only tests of actual command/environment builders and lifecycle decisions using fake hardware and HTTP/subprocess adapters. Verify L4, L40S, and RTX PRO 6000 outputs with no tuning environment variables set, plus explicit overrides, absent optional arguments, unchanged service paths/ports, supported per-environment arguments, generated-config selection, stale-code rejection, and failed-readiness/warmup propagation. Extend the existing deployment tests beyond string-presence checks. Local import tests must not need CUDA, weights, or access to persistent volumes.

Run the complete acceptance matrix on separate staging Modal deployments with isolated writable app/runtime state. A staging `prepare_model()` must not update the production app Volume. For each of L4, L40S, and RTX PRO 6000, change only the `gpu=` declaration as the hardware/tuning input, then test image provisioning, preparation of the selected revision, `modal deploy`, first cold request, snapshot creation, scale-down and restore into a new container, another successful synthesis, and a queued concurrent workload. Run both engines and each certified optional backend. Save the deployment revision, detected GPU, resolved settings, peak VRAM, audio validity, and timings. A successful local process or `modal run` helper is not evidence that deployed snapshot restore works.

**5. Extend profiles to the optional backends**

For IndexTTS 2.5, generate a runtime deployment YAML from the pinned base config, preserving model-specific keys, connectors, attention backend, sampling, and context lengths. Write it atomically to the backend runtime directory; do not edit the sibling checkout's base YAML. Use a profile/config fingerprint in the generated filename.

- For the 24 GB benchmark, start with stage 0 `max_num_seqs=4`; stage 1 `max_num_seqs=1` and `s2mel_cfm_batch_size=1`; stage 0 eager execution; and stage 1 DiT/vocoder compilation disabled. Keep stage 1's existing eager mode.
- Include a separate L40S/48 GB benchmark profile: initially stage 0 `max_num_seqs=16`, stage 1 `max_num_seqs=4`, and `s2mel_cfm_batch_size=4`. Measure stage budgets and compilation overhead before promoting these candidates to defaults. Preserve the pinned 32/16/16 values for the RTX PRO 6000 profile.
- Start frontend segment concurrency at one. Choose each stage's memory budget from its own profiling results, including any retained IndexTTS 2.0 models. Do not assume the two stages need equal fractions.
- Preserve stage 0's `TRITON_ATTN`, disabled chunked prefill, and 2560 context/token budget. Preserve stage 1's audio-length semantics; reducing 32768 arbitrarily can truncate valid output. Inspect stage-specific scheduling before changing its 8192 batched-token value.
- In `index25_manager.py`, set `INDEXTTS25_DEPLOY_CONFIG` for the standard sibling launcher. It already supports this environment variable. Make the Modal builder use the same generated file; its custom command currently hardcodes `--deploy-config` and would otherwise bypass the override.
- Preserve explicitly supplied custom startup commands/configs. Report when those commands own engine tuning rather than claiming the profile controls them.

For Confucius, pass the resolved memory fraction through both supported environment variable names. Its low-level `Text2SemanticVLLM` accepts sequence/context limits and engine kwargs, but the inspected public FastAPI/inference layer does not forward them. Full profile support therefore needs a small companion change in `Confucius4-TTS`: typed CLI/environment options forwarded through `fastapi_app.py` and `confuciustts/cli/inference.py`. Include eager/batched-token settings and remove Modal's unconditional S2A compilation in the low-memory profile. Record and test the companion revision; do not inject unsupported CLI flags.

**6. Shared-GPU lifecycle is part of certification**

First measure backend switching with the existing sleep hooks and retained PyTorch models. If a 24 GB backend cannot fit, add controlled release/offload of inactive backend components or selected-backend lazy initialization. Audit speaker presets and emotion extraction before making IndexTTS 2.0 optional at startup, since they use its models.

Any automatic transition must wait for active GPU work to finish and serialize backend startup/sleep/wake. Sleeping an engine while another request uses it is not a safe memory policy. Reuse the existing managers, adding a shared transition/admission coordinator if needed; do not terminate externally managed services.

Certify optional translation/audio workflows separately. The Modal MOSS server is started early but loads its model only on the first transcription request, then retains it. Its `/v1/models` readiness response does not measure that footprint. SGLang's separate MOSS memory fraction also needs independent sizing. A TTS-only test cannot establish that MOSS, Stable Audio, enhancement models, and TTS will fit together. Use existing explicit unload paths or stage handoffs where feasible; report unsupported combinations clearly until tested.

**7. Delivery order and acceptance checks**

1. Add the pure resolver, runtime probe, override precedence, diagnostics, and CPU-only tests. Preserve manual Modal GPU selection, wire automatic runtime tuning, and verify the deployed-code revision contract. Add a config-inspection command that does not load weights.
2. Integrate IndexTTS 2.0 engines, effective concurrency, and Modal's runtime/command builders in the same milestone. Benchmark actual L4 and L40S Modal cold deployments and snapshot restores, then tune candidate budgets. Run the existing RTX PRO 6000 Modal configuration as the regression baseline.
3. Add generated 2.5 stage configs and the Confucius forwarding change through their Modal custom commands. Measure startup, inference, and switching in Modal separately; add residency handling only where measurements require it.
4. Apply launcher parity to quickstart/Docker, validate optional translation handoffs and overloaded request queues, and complete the Modal acceptance matrix. Publish measured defaults and profile-specific support results in both READMEs, with Modal examples first.

CPU checks should cover profile boundaries (including approximately 22–23 GiB L4 reports), explicit overrides, invalid values, unavailable/misidentified devices, free-memory insufficiency, multiple visible GPUs, argument construction for both engines, launch-path parity, generated YAML preservation, and bounded request admission. Verify import safety and existing route/streaming contracts. Use the existing unittest suite and project lint/compile checks after implementation.

GPU checks should include:

- Cold startup followed by the first real inference, both with and without text emotion; reference-audio conditioning and bounded cache churn.
- Streaming and nonstreaming synthesis, long/multisentence text, duration control, and the longest supported audio cases.
- Requests at the profile concurrency and above it: excess requests queue; no CUDA OOM, deadlock, or broken keepalive framing.
- Per-process/whole-device VRAM measurements at engine load, non-vLLM load, graph/compile warmup, inference peak, sleep, and wake. Record latency/RTF and sustained throughput.
- A repeated mixed-request run of at least 100 requests, plus switching among certified backends and Modal snapshot restore.
- RTX PRO 6000 regression comparison against the unchanged configuration. Initial performance gate: no more than 10% median latency/throughput regression under the same workload, subject to measured run-to-run variance.

L4 and L40S acceptance requires successful cold load, warmup, and sustained supported workloads without OOM, with measured safety headroom and complete audio output. Startup failures must surface the effective profile and original error promptly. The current initializer catches failures and returns `False`, and warmup can swallow errors; validation must check readiness and actual synthesis results, not merely process survival or a successful warmup HTTP response.

Implementation is complete when resolved settings reach every intended engine and deployment path, including Modal's custom backend commands. Release acceptance requires actual Modal cold deployment and snapshot restore on L4, L40S, and RTX PRO 6000, with the existing RTX PRO 6000 performance retained. For each GPU, the operator must need only the manual edit to `gpu=`; startup settings adapt automatically. Support is certified only after the corresponding GPU workload passes; the proposed numeric profiles above are not a substitute for those measurements.
