# Colab setup and deployment parity

Colab follows the standalone quick-start lifecycle: prepare the main application,
then start optional backends when requested. Modal installs component environments
while building its image; its CPU setup manager prepares checkpoint bundles.
All three use the same application, model managers, and worker interfaces.

## Colab defaults and preparation controls

| Component | Default setup | Optional preparation before use |
| --- | --- | --- |
| IndexTTS 2.0 / WebUI | Main environment, checkpoints, and service | Required |
| HY-MT translation | Download checkpoints, matching quick start | Turn `DOWNLOAD_HY_MT` off for normal Hub fallback |
| Confucius4-TTS | Clone source only; existing standalone launcher installs and downloads on first use | `PREPARE_CONFUCIUS_SERVICE` starts the manager, prepares weights, then sleeps its engine |
| IndexTTS 2.5 | Clone source only; existing standalone launcher installs and downloads on first use | `PREPARE_INDEXTTS25_SERVICE` starts the manager, prepares weights, then sleeps its engine |
| ClearVoice | Isolated worker environment and default enhancement/SR weights install on first use | `PREPARE_CLEARVOICE` and `DOWNLOAD_CLEARVOICE_MODELS` |
| MOSS transcription | Isolated environment and local Transformers service start on demand; weights load on first transcription | `PREPARE_MOSS` and `DOWNLOAD_MOSS_MODEL` |
| Qwen3-ASR + OmniVAD | Isolated worker environment; weights download on first use | `PREPARE_QWEN_ASR` and `DOWNLOAD_QWEN_MODELS` (ASR, aligner, OmniVAD, Sortformer) |
| Qwen3 Voice Design | Runtime dependencies installed; Hub weights download on first use | `DOWNLOAD_VOICE_DESIGN` |
| Stable Audio 3 | Runtime dependencies installed; gated weights require preparation | Select a variant in `STABLE_AUDIO_DOWNLOAD` |
| WhisperX / Parakeet / audio separation | Runtime dependencies installed; selected weights download on first use | Select the required feature in the WebUI |
| Video downloads | yt-dlp and a verified Node.js 22 runtime | Site-specific authentication may still be needed |

Every `PREPARE_*` control and extra model download control defaults off.
Source checkout controls default on, like quick start; cloning a source repository
does not install that backend's environment or download its checkpoints.
Selecting a model download for Qwen ASR or ClearVoice also prepares its worker
environment so the installed package's normal download routines can run.

Preparation does not send synthesis requests. Services can load models while
preparing, and release or sleep their engines afterward. `ENABLE_WARMUP` alone
controls startup inference warmup and S2Mel compilation together, both off by
default. Enabling it can speed up generation but costs startup warmup time.
Google Drive persistence also defaults off.

## Shared code and runtime isolation

Confucius and IndexTTS 2.5 use their existing standalone launchers and the WebUI's
normal model wake/unload routes. Colab checks out their sources beside this
repository and forwards their progress to the notebook logs.

ClearVoice needs a different rotary-embedding version from the main separation
stack. Qwen ASR and MOSS also need different Transformers versions from TTS.
The Colab wrappers prepare separate environments and then execute the existing
ClearVoice/Qwen workers or MOSS HTTP server. A successful installation is reused;
failed installations remain retryable. Checked preparation controls perform the
same setup ahead of launch.

The reported Colab failure was a missing Docker command. Installing the CLI alone
would not provide a working Docker daemon. The notebook uses the existing local
MOSS Transformers HTTP service, as Modal does, rather than assuming a daemon is
available. Standalone retains its existing managed SGLang/Docker default. These
engines use the same MOSS model but can differ in throughput and latency.

The only shared WebUI configuration change is an optional
`QWEN3_TTS_USE_FLASH_ATTENTION` override. Its default remains enabled for standalone
and Modal; Colab aligns it with `INSTALL_FLASH_ATTENTION`.

## What still depends on the user or runtime

- Stable Audio requires accepting its model access terms and an authorized
  `HF_TOKEN`. Enable `USE_HF_SECRET` to read the token from Colab Secrets.
- Selecting pyannote diarization can require gated model access. The Qwen model
  preparation option prefetches the ungated Sortformer alternative; it does not
  grant pyannote access or prefetch every language-specific WhisperX aligner.
- Cloud translation providers require their API credentials. Local HY-MT can
  run without a cloud translation key.
- Some video sources require cookies or a working proof-of-origin token provider.
  Installing its yt-dlp plugin does not start a token-provider HTTP service.
- Downloads and first-use installation require network access, sufficient disk,
  and a compatible CUDA driver. The notebook checks its main environment and
  validates isolated environments when they are prepared.

CPU regressions cover notebook defaults, setup reuse/failure handling, worker
argument forwarding, process ownership, model download validation, and the shared
service preparation routes. They do not establish successful inference on a live
Colab G4 GPU; that remains a runtime validation step.
