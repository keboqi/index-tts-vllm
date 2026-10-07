#!/usr/bin/env bash
set -Eeuo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

APP_SERVER="${APP_SERVER:-web}"
MODEL_DIR="${MODEL_DIR:-checkpoints}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"

case "$APP_SERVER" in
    web)
        PORT="${PORT:-8000}"
        MODEL="${MODEL:-garyswansrs/index_tts_2_vllm}"
        VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-0}"
        CONVERT_MODEL="${CONVERT_MODEL:-0}"
        if [[ "$CONVERT_MODEL" == "1" ]]; then
            printf 'The WebUI needs the preconverted IndexTTS2 bundle; set CONVERT_MODEL=0.\n' >&2
            exit 1
        fi
        ;;
    legacy-api)
        PORT="${PORT:-8001}"
        MODEL="${MODEL:-IndexTeam/IndexTTS-1.5}"
        VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-1}"
        CONVERT_MODEL="${CONVERT_MODEL:-1}"
        ;;
    *)
        printf 'Unknown APP_SERVER: %s (expected web or legacy-api).\n' "$APP_SERVER" >&2
        exit 1
        ;;
esac
export VLLM_USE_MODELSCOPE

printf 'Starting %s with %s in %s on port %s\n' "$APP_SERVER" "$MODEL" "$MODEL_DIR" "$PORT"

check_model_exists() {
    local required=(config.yaml gpt.pth bpe.model)
    local file
    if [[ "$APP_SERVER" == "web" ]]; then
        required+=(
            s2mel.pth wav2vec2bert_stats.pt feat1.pt feat2.pt
            gpt/config.json
            qwen0.6bemo4-merge/config.json qwen0.6bemo4-merge/model.safetensors
            w2v-bert-2.0/config.json w2v-bert-2.0/model.safetensors
            w2v-bert-2.0/preprocessor_config.json
            semantic_codec/model.safetensors campplus/campplus_cn_common.bin
            bigvgan/config.json bigvgan/bigvgan_generator.pt
        )
    else
        required+=(bigvgan_generator.pth)
    fi
    for file in "${required[@]}"; do
        if [[ ! -f "$MODEL_DIR/$file" ]]; then
            printf 'Missing model asset: %s/%s\n' "$MODEL_DIR" "$file" >&2
            return 1
        fi
    done
    if [[ "$APP_SERVER" == "web" ]] && \
       [[ ! -f "$MODEL_DIR/gpt/model.safetensors" && ! -f "$MODEL_DIR/gpt/pytorch_model.bin" ]]; then
        printf 'Missing preconverted GPT weights in %s/gpt\n' "$MODEL_DIR" >&2
        return 1
    fi
}

if ! check_model_exists; then
    if [[ "$DOWNLOAD_MODEL" != "1" ]]; then
        printf 'Model assets are missing and DOWNLOAD_MODEL=0.\n' >&2
        exit 1
    fi
    mkdir -p -- "$MODEL_DIR"
    if [[ "$VLLM_USE_MODELSCOPE" == "1" ]]; then
        modelscope download --model "$MODEL" --local_dir "$MODEL_DIR"
    else
        python3 - "$MODEL" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PY
    fi
    check_model_exists
fi

if [[ "$APP_SERVER" == "legacy-api" ]]; then
    # convert_hf_format.sh writes the GPT bundle consumed by infer_vllm.py.
    if [[ ! -f "$MODEL_DIR/gpt/config.json" || ! -f "$MODEL_DIR/gpt/tokenizer.json" ]] || \
       [[ ! -f "$MODEL_DIR/gpt/pytorch_model.bin" && ! -f "$MODEL_DIR/gpt/model.safetensors" ]]; then
        if [[ "$CONVERT_MODEL" != "1" ]]; then
            printf 'Legacy GPT conversion is missing and CONVERT_MODEL=0.\n' >&2
            exit 1
        fi
        bash convert_hf_format.sh "$MODEL_DIR"
        if [[ ! -f "$MODEL_DIR/gpt/config.json" || ! -f "$MODEL_DIR/gpt/pytorch_model.bin" ]]; then
            printf 'Legacy GPT conversion did not produce its expected files.\n' >&2
            exit 1
        fi
    fi
    exec env VLLM_USE_V1=0 python3 api_server.py \
        --model_dir "$MODEL_DIR" \
        --port "$PORT" \
        --gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.25}" \
        "$@"
fi

exec python3 fastapi_webui_v2.py --model_dir "$MODEL_DIR" --port "$PORT" "$@"
