#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
MODEL_DIR="${MODEL_DIR:-checkpoints}"
HY_MT_TRANSLATION_MODEL="${HY_MT_TRANSLATION_MODEL:-tencent/Hy-MT2-1.8B}"
HY_MT_TRANSLATION_LOCAL_DIR="${HY_MT_TRANSLATION_LOCAL_DIR:-${MODEL_DIR}/hy-mt}"
export HY_MT_TRANSLATION_MODEL
export HY_MT_TRANSLATION_LOCAL_DIR
CONFUCIUS_REPO_DIR="${CONFUCIUS_REPO_DIR:-../Confucius4-TTS}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
INSTALL_CONFUCIUS="${INSTALL_CONFUCIUS:-1}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"
DOWNLOAD_HY_MT_MODEL="${DOWNLOAD_HY_MT_MODEL:-1}"
RUN_SERVER="${RUN_SERVER:-1}"
EXPORT_TUNNEL="${EXPORT_TUNNEL:-1}"
SERVER_PORT="${SERVER_PORT:-8000}"
UV_BIN="${UV_BIN:-uv}"

if [[ "${1:-}" == "--setup-only" ]]; then
    RUN_SERVER=0
    shift
fi

log() {
    printf '[index-tts-vllm] %s\n' "$*"
}

run_as_root() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        printf 'sudo is required to install system packages. Install ffmpeg, sox, and libstdc++6 manually.\n' >&2
        exit 1
    fi
}

ensure_system_dependencies() {
    if command -v ffmpeg >/dev/null 2>&1 && command -v sox >/dev/null 2>&1; then
        return
    fi

    if [[ "${INSTALL_SYSTEM_DEPS}" != "1" ]]; then
        log "System dependency installation disabled; ffmpeg and/or sox are missing"
        return
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        printf 'Install ffmpeg, sox, and the libstdc++ runtime before continuing.\n' >&2
        exit 1
    fi

    log "Installing system audio dependencies"
    run_as_root apt-get update
    run_as_root apt-get install -y ffmpeg sox libstdc++6
}

ensure_uv() {
    if command -v "${UV_BIN}" >/dev/null 2>&1; then
        return
    fi
    if ! command -v curl >/dev/null 2>&1; then
        printf 'uv is missing. Install uv first: https://docs.astral.sh/uv/getting-started/installation/\n' >&2
        exit 1
    fi

    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    UV_BIN="$(command -v uv || true)"
    if [[ -z "${UV_BIN}" ]]; then
        printf 'uv was installed but is not available on PATH. Open a new shell and rerun this script.\n' >&2
        exit 1
    fi
}

ensure_confucius_checkout() {
    if [[ "${INSTALL_CONFUCIUS}" != "1" || -d "${CONFUCIUS_REPO_DIR}" ]]; then
        return
    fi
    if ! command -v git >/dev/null 2>&1; then
        printf 'git is required to clone the optional Confucius4-TTS backend.\n' >&2
        exit 1
    fi

    log "Cloning optional Confucius4-TTS backend"
    git clone https://github.com/keboqi/Confucius4-TTS "${CONFUCIUS_REPO_DIR}"
}

ensure_venv() {
    if [[ -x "${VENV_DIR}/bin/python" ]]; then
        log "Using existing virtual environment: ${VENV_DIR}"
        return
    fi

    log "Creating isolated uv environment: ${VENV_DIR} (Python ${PYTHON_VERSION})"
    "${UV_BIN}" venv --seed --python "${PYTHON_VERSION}" "${VENV_DIR}"
}

activate_venv() {
    VIRTUAL_ENV="$(cd -- "${VENV_DIR}" && pwd)"
    export VIRTUAL_ENV
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
}

install_python_dependencies() {
    local python="${VENV_DIR}/bin/python"

    log "Installing the CUDA 13.0 PyTorch stack"
    "${UV_BIN}" pip install --python "${python}" \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu130

    log "Installing core requirements"
    "${UV_BIN}" pip install --python "${python}" -r requirements.txt

    log "Installing WebUI, audio, and download integrations"
    "${UV_BIN}" pip install --python "${python}" \
        alias-free-torch dill einops-exts huggingface_hub importlib-resources \
        nnAudio PyWavelets safetensors scipy soxr torchsde tqdm transformers \
        v-diffusion-pytorch vector-quantize-pytorch \
        'audio-separator[gpu]' clearvoice google-genai whisperx pydub \
        'nemo_toolkit[asr]' json-repair \
        'yt-dlp[default]' yt-dlp-ejs bgutil-ytdlp-pot-provider \
        sentencepiece 'numpy<2'

    # This source currently declares an incompatible Python range. Its dependencies
    # are installed explicitly above, so pip is used only for this no-deps install.
    if ! "${python}" -m pip show stable-audio-tools >/dev/null 2>&1; then
        log "Installing Stable Audio Tools"
        "${python}" -m pip install --upgrade --no-deps --ignore-requires-python \
            'git+https://github.com/Stability-AI/stable-audio-tools.git'
    fi

    log "Installing optimized GPU inference extensions"
    "${UV_BIN}" pip install --python "${python}" flashinfer-python
    if ! "${python}" -c 'import flash_attn' >/dev/null 2>&1; then
        "${UV_BIN}" pip install --python "${python}" \
            --no-build-isolation flash-attn
    fi

    log "MOSS Transcribe+Diarize is the default translation ASR pipeline; the Docker manager starts it on first use."
    # Qwen3-ASR + OmniVAD remains available as an alternate local pipeline.
    # The README quick start installs these directly and is battle-tested.
    log "Installing Qwen3-ASR and OmniVAD for alternate translation pipeline"
    "${UV_BIN}" pip install --python "${python}" qwen-asr omnivad sentencepiece
}

download_model() {
    if [[ "${DOWNLOAD_MODEL}" != "1" ]]; then
        log "Model download disabled"
        return
    fi

    log "Downloading/resuming IndexTTS2 vLLM weights"
    "${VENV_DIR}/bin/hf" download \
        garyswansrs/index_tts_2_vllm \
        --local-dir "${MODEL_DIR}"

    if [[ "${DOWNLOAD_HY_MT_MODEL}" == "1" ]]; then
        log "Downloading/resuming HY-MT translation model"
        "${VENV_DIR}/bin/hf" download \
            "${HY_MT_TRANSLATION_MODEL}" \
            --local-dir "${HY_MT_TRANSLATION_LOCAL_DIR}"
    fi
}

ensure_system_dependencies
ensure_uv
ensure_confucius_checkout
ensure_venv
activate_venv
install_python_dependencies
download_model

if command -v node >/dev/null 2>&1; then
    export YTDLP_NODE_PATH="$(node -p 'process.execPath')"
else
    log "Node.js is unavailable; yt-dlp JavaScript challenge support may be limited"
fi

if [[ "${RUN_SERVER}" != "1" ]]; then
    log "Setup complete"
    exit 0
fi

# Export port 8000 via Cloudflare tunnel (runs in background)
EXPORT_SCRIPT="${SCRIPT_DIR}/export_services.sh"
if [[ "${EXPORT_TUNNEL}" == "1" ]] && [[ -x "${EXPORT_SCRIPT}" ]]; then
    log "Exporting port ${SERVER_PORT} via Cloudflare tunnel"
    bash "${EXPORT_SCRIPT}" start "${SERVER_PORT}" &
    TUNNEL_PID=$!
    log "Cloudflare tunnel starting in background (PID: ${TUNNEL_PID})"
elif [[ "${EXPORT_TUNNEL}" == "1" ]]; then
    log "export_services.sh not found or not executable at ${EXPORT_SCRIPT}; skipping tunnel"
fi

log "Starting FastAPI WebUI on port ${SERVER_PORT}"
exec "${VENV_DIR}/bin/python" fastapi_webui_v2.py \
    --use_torch_compile \
    --model_dir "${MODEL_DIR}" \
    --port "${SERVER_PORT}" \
    --confucius_repo_dir "${CONFUCIUS_REPO_DIR}" \
    "$@"
