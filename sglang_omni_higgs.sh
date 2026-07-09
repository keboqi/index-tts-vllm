#!/usr/bin/env bash
set -Eeuo pipefail

# SGLang-Omni Higgs TTS Docker manager
# Usage:
#   bash sglang_omni_higgs.sh deploy
#   bash sglang_omni_higgs.sh start
#   bash sglang_omni_higgs.sh stop
#   bash sglang_omni_higgs.sh status
#   bash sglang_omni_higgs.sh logs
#   bash sglang_omni_higgs.sh test output.wav
#
# Optional env overrides:
#   NAME=index-tts-higgs-sglang
#   IMAGE=lmsysorg/sglang-omni:dev
#   MODEL=bosonai/higgs-audio-v3-tts-4b
#   PORT=8002
#   MEM_FRACTION_STATIC=0.30
#   MAX_RUNNING_REQUESTS=100
#   DTYPE=bfloat16
#   ALLOWED_LOCAL_MEDIA_PATH=/workspace/higgs_tts_reference_audio
#   PROJECT_ROOT=$HOME/index-tts-higgs-sglang-workspace
#   HF_CACHE=$HOME/.cache/huggingface
#   EXTRA_ARGS="--some-sgl-omni-arg value"

NAME="${NAME:-index-tts-higgs-sglang}"
IMAGE="${IMAGE:-lmsysorg/sglang-omni:dev}"
MODEL="${MODEL:-bosonai/higgs-audio-v3-tts-4b}"
PORT="${PORT:-8002}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.30}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-100}"
DTYPE="${DTYPE:-bfloat16}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/workspace/higgs_tts_reference_audio}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/index-tts-higgs-sglang-workspace}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LINES="${LINES:-200}"

REPO_IN="/workspace/sglang-omni"
LOG_DIR_IN="/workspace/logs"
PID_FILE_IN="/workspace/higgs_tts_${PORT}.pid"
PGID_FILE_IN="/workspace/higgs_tts_${PORT}.pgid"
LOG_FILE_IN="/workspace/logs/higgs_tts_${PORT}.log"
LIFECYCLE_LOCK_FILE="$PROJECT_ROOT/.sglang_omni_higgs_${PORT}.lifecycle.lock"
LIFECYCLE_LOCK_DIR="${LIFECYCLE_LOCK_FILE}.d"
LIFECYCLE_LOCK_TIMEOUT="${LIFECYCLE_LOCK_TIMEOUT:-3600}"
_LIFECYCLE_LOCK_KIND=""

usage() {
  cat <<USAGE
SGLang-Omni Higgs TTS manager

Commands:
  deploy          Pull Docker image, create persistent container, clone/install sglang-omni, download model
  start           Start the Higgs TTS server inside the container
  stop            Stop only the Higgs TTS server process, keep container alive
  restart         Stop then start the Higgs TTS server
  status          Show container/server status and URLs
  logs            Tail server logs; set LINES=500 for more history
  shell           Open an interactive shell inside the container
  test [file.wav] Send a test TTS request to /v1/audio/speech
  container-stop  Stop the Docker container
  container-rm    Remove the Docker container, keeping host cache/repo files

Env overrides:
  NAME=${NAME}
  IMAGE=${IMAGE}
  MODEL=${MODEL}
  PORT=${PORT}
  MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC}
  MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}
  DTYPE=${DTYPE}
  ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH}
  PROJECT_ROOT=${PROJECT_ROOT}
  HF_CACHE=${HF_CACHE}
  EXTRA_ARGS=${EXTRA_ARGS}

Examples:
  bash $0 deploy
  bash $0 start
  bash $0 logs
  PORT=8010 bash $0 start
  MEM_FRACTION_STATIC=0.35 bash $0 start
  MAX_RUNNING_REQUESTS=100 DTYPE=bfloat16 bash $0 start
USAGE
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

release_lifecycle_lock() {
  if [[ "$_LIFECYCLE_LOCK_KIND" == "mkdir" ]]; then
    rm -rf "$LIFECYCLE_LOCK_DIR"
  fi
}

acquire_lifecycle_lock() {
  mkdir -p "$PROJECT_ROOT"
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$LIFECYCLE_LOCK_FILE"
    flock -w "$LIFECYCLE_LOCK_TIMEOUT" 9 || {
      echo "Timed out waiting for backend lifecycle lock: $LIFECYCLE_LOCK_FILE" >&2
      exit 1
    }
    _LIFECYCLE_LOCK_KIND="flock"
    return
  fi

  local deadline=$((SECONDS + LIFECYCLE_LOCK_TIMEOUT))
  while ! mkdir "$LIFECYCLE_LOCK_DIR" 2>/dev/null; do
    if [[ -f "$LIFECYCLE_LOCK_DIR/pid" ]]; then
      local owner
      owner=$(cat "$LIFECYCLE_LOCK_DIR/pid" 2>/dev/null || true)
      if [[ -n "$owner" ]] && ! kill -0 "$owner" 2>/dev/null; then
        rm -rf "$LIFECYCLE_LOCK_DIR"
        continue
      fi
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for backend lifecycle lock: $LIFECYCLE_LOCK_DIR" >&2
      exit 1
    fi
    sleep 1
  done
  echo "$$" > "$LIFECYCLE_LOCK_DIR/pid"
  _LIFECYCLE_LOCK_KIND="mkdir"
  trap release_lifecycle_lock EXIT INT TERM
}

container_exists() {
  docker container inspect "$NAME" >/dev/null 2>&1
}

container_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || true)" == "true" ]]
}

print_urls() {
  echo "Local URL:      http://localhost:${PORT}"
  if [[ -n "${LIGHTNING_CLOUDSPACE_HOST:-}" ]]; then
    echo "Lightning URL:  https://${PORT}-${LIGHTNING_CLOUDSPACE_HOST}"
  fi
}

ensure_docker() {
  need_cmd docker
  docker info >/dev/null 2>&1 || {
    echo "Docker is not running or current user cannot access Docker." >&2
    exit 1
  }
}

ensure_container() {
  ensure_docker
  mkdir -p "$PROJECT_ROOT" "$HF_CACHE"

  if container_exists; then
    if ! container_running; then
      echo "Starting existing container: $NAME"
      docker start "$NAME" >/dev/null
    fi
    docker exec -i -e ALLOWED_LOCAL_MEDIA_PATH="$ALLOWED_LOCAL_MEDIA_PATH" "$NAME" /bin/zsh -lc \
      'mkdir -p "$ALLOWED_LOCAL_MEDIA_PATH" && chmod 0777 "$ALLOWED_LOCAL_MEDIA_PATH"'
    return 0
  fi

  echo "Creating container: $NAME"
  docker run -dit \
    --name "$NAME" \
    --restart unless-stopped \
    --shm-size 32g \
    --gpus all \
    --ipc host \
    --network host \
    --privileged \
    -v "$PROJECT_ROOT:/workspace" \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    "$IMAGE" \
    /bin/zsh -lc "sleep infinity" >/dev/null

  docker exec -i -e ALLOWED_LOCAL_MEDIA_PATH="$ALLOWED_LOCAL_MEDIA_PATH" "$NAME" /bin/zsh -lc \
    'mkdir -p "$ALLOWED_LOCAL_MEDIA_PATH" && chmod 0777 "$ALLOWED_LOCAL_MEDIA_PATH"'
}

deploy() {
  ensure_docker
  echo "Pulling image: $IMAGE"
  docker pull "$IMAGE"

  ensure_container

  echo "Bootstrapping sglang-omni and downloading model inside container..."
  docker exec -i \
    -e MODEL="$MODEL" \
    "$NAME" \
    /bin/zsh -lc '
set -Eeuo pipefail
mkdir -p /workspace/logs

if [ ! -d /workspace/sglang-omni/.git ]; then
  echo "Cloning sglang-omni repo..."
  rm -rf /workspace/sglang-omni
  git clone https://github.com/sgl-project/sglang-omni.git /workspace/sglang-omni
else
  echo "Repo already exists; updating..."
  cd /workspace/sglang-omni
  git fetch --all --prune || true
  git pull --ff-only || true
fi

cd /workspace/sglang-omni

if [ ! -x .venv/bin/python ]; then
  echo "Creating Python 3.12 venv..."
  uv venv .venv -p 3.12
else
  echo "Python venv already exists; skipping venv creation."
fi

source .venv/bin/activate

if python -c "import sglang_omni" >/dev/null 2>&1 && [ -x .venv/bin/sgl-omni ]; then
  echo "sglang-omni import and CLI work; skipping reinstall."
else
  echo "Installing sglang-omni editable package..."
  uv pip install -v -e .
fi

if [ ! -x .venv/bin/sgl-omni ]; then
  echo "Installation failed: .venv/bin/sgl-omni was not created." >&2
  exit 1
fi

echo "Downloading model: ${MODEL}"
if command -v hf >/dev/null 2>&1; then
  hf download "${MODEL}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${MODEL}"
else
  python - <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(os.environ["MODEL"])
PY
fi

echo "Deploy finished."
'

  status
}

server_running_in_container() {
  container_running || return 1
  docker exec -i \
    -e MODEL="$MODEL" \
    -e PORT="$PORT" \
    -e PID_FILE_IN="$PID_FILE_IN" \
    "$NAME" /bin/zsh -lc '
find_server_pids() {
  local proc pid cmdline
  for proc in /proc/[0-9]*; do
    pid=${proc##*/}
    [[ -r "$proc/cmdline" ]] || continue
    cmdline=$(tr "\0" " " < "$proc/cmdline" 2>/dev/null || true)
    if [[ "$cmdline" == *"sgl-omni serve"* && "$cmdline" == *"--model-path $MODEL"* && "$cmdline" == *"--port $PORT"* ]]; then
      echo "$pid"
    fi
  done
}

if [ -f "$PID_FILE_IN" ]; then
  PID=$(cat "$PID_FILE_IN" 2>/dev/null || true)
  if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null && [ -r "/proc/$PID/cmdline" ]; then
    CMDLINE=$(tr "\0" " " < "/proc/$PID/cmdline" 2>/dev/null || true)
    # The console launcher may re-exec after startup, so trust the persisted
    # live PID when it still belongs to an SGLang-Omni process.
    if [[ "$CMDLINE" == *sgl-omni* || "$CMDLINE" == *"$MODEL"* ]]; then
      exit 0
    fi
  fi
fi

PID=$(find_server_pids | head -n 1)
if [[ -n "${PID:-}" ]]; then
  echo "$PID" > "$PID_FILE_IN"
  exit 0
fi
rm -f "$PID_FILE_IN"
exit 1
' >/dev/null 2>&1
}

start_server() {
  ensure_container

  if ! docker exec -i "$NAME" /bin/zsh -lc "test -x ${REPO_IN}/.venv/bin/sgl-omni" >/dev/null 2>&1; then
    echo "sglang-omni is not deployed yet; running deploy first."
    deploy
  fi

  if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 3 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "Higgs TTS is already healthy on port ${PORT}; reusing it."
    return 0
  fi

  if server_running_in_container; then
    echo "Higgs TTS server is already running."
    status
    return 0
  fi

  echo "Starting Higgs TTS server on port ${PORT}..."
  docker exec -i \
    -e MODEL="$MODEL" \
    -e PORT="$PORT" \
    -e MEM_FRACTION_STATIC="$MEM_FRACTION_STATIC" \
    -e MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS" \
    -e DTYPE="$DTYPE" \
    -e ALLOWED_LOCAL_MEDIA_PATH="$ALLOWED_LOCAL_MEDIA_PATH" \
    -e EXTRA_ARGS="$EXTRA_ARGS" \
    -e REPO_IN="$REPO_IN" \
    -e PID_FILE_IN="$PID_FILE_IN" \
    -e PGID_FILE_IN="$PGID_FILE_IN" \
    -e LOG_FILE_IN="$LOG_FILE_IN" \
    "$NAME" \
    /bin/zsh -lc '
set -Eeuo pipefail
mkdir -p "$(dirname "$LOG_FILE_IN")"
mkdir -p "$ALLOWED_LOCAL_MEDIA_PATH"
chmod 0777 "$ALLOWED_LOCAL_MEDIA_PATH"
cd "$REPO_IN"
source .venv/bin/activate

# Start a dedicated process session so every pipeline stage can be stopped as
# one process group. Intentionally allow EXTRA_ARGS word splitting for flags.
SESSION_PREFIX=""
if command -v setsid >/dev/null 2>&1; then
  SESSION_PREFIX="setsid"
fi
nohup $SESSION_PREFIX .venv/bin/sgl-omni serve \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --stages.2.runtime.sglang_server_args.mem_fraction_static="$MEM_FRACTION_STATIC" \
  --stages.2.runtime.sglang_server_args.max_running_requests="$MAX_RUNNING_REQUESTS" \
  --stages.2.runtime.sglang_server_args.dtype="$DTYPE" \
  $EXTRA_ARGS \
  > "$LOG_FILE_IN" 2>&1 &

echo $! > "$PID_FILE_IN"
PID=$(cat "$PID_FILE_IN")
if [[ -n "$SESSION_PREFIX" ]]; then
  echo "$PID" > "$PGID_FILE_IN"
else
  ps -o pgid= -p "$PID" 2>/dev/null | tr -d " " > "$PGID_FILE_IN" || true
fi
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  rm -f "$PID_FILE_IN"
  rm -f "$PGID_FILE_IN"
  echo "Higgs TTS server failed to start. Recent log output:" >&2
  tail -n 80 "$LOG_FILE_IN" >&2 || true
  exit 1
fi
echo "Started PID $PID"
echo "Log file: $LOG_FILE_IN"
'

  print_urls
  echo "Use: bash $0 logs"
}

stop_server() {
  if ! container_exists; then
    echo "Container does not exist: $NAME"
    return 0
  fi
  if ! container_running; then
    echo "Container is not running: $NAME"
    return 0
  fi

  echo "Stopping Higgs TTS server if running..."
  docker exec -i \
    -e MODEL="$MODEL" \
    -e REPO_IN="$REPO_IN" \
    -e PID_FILE_IN="$PID_FILE_IN" \
    -e PGID_FILE_IN="$PGID_FILE_IN" \
    "$NAME" \
    /bin/bash -lc '
set -Eeuo pipefail
STOPPED=0

find_owned_pids() {
  local proc pid cmdline process_env
  for proc in /proc/[0-9]*; do
    pid=${proc##*/}
    [[ "$pid" == "$$" ]] && continue
    [[ -r "$proc/cmdline" ]] || continue
    cmdline=$(tr "\0" " " < "$proc/cmdline" 2>/dev/null || true)
    process_env=$(tr "\0" "\n" < "$proc/environ" 2>/dev/null || true)
    if [[ "$cmdline" == *"$REPO_IN/.venv/bin/python"* \
      || "$cmdline" == *"sgl-omni serve"* \
      || "$process_env" == *"VIRTUAL_ENV=$REPO_IN/.venv"* ]]; then
      echo "$pid"
    fi
  done
}

# New launches have their own session/process group.
if [[ -f "$PGID_FILE_IN" ]]; then
  PGID=$(cat "$PGID_FILE_IN" 2>/dev/null || true)
  if [[ "$PGID" =~ ^[0-9]+$ ]] && (( PGID > 1 )); then
    kill -TERM -- "-$PGID" 2>/dev/null || true
    STOPPED=1
  fi
fi

# Sweep legacy or orphaned workers from launches made before process-group
# tracking was added. The container is dedicated to this SGLang environment.
mapfile -t OWNED_PIDS < <(find_owned_pids)
if (( ${#OWNED_PIDS[@]} > 0 )); then
  for pid in "${OWNED_PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  STOPPED=1
fi

for _ in $(seq 1 30); do
  mapfile -t OWNED_PIDS < <(find_owned_pids)
  (( ${#OWNED_PIDS[@]} == 0 )) && break
  sleep 0.5
done

mapfile -t OWNED_PIDS < <(find_owned_pids)
for pid in "${OWNED_PIDS[@]}"; do
  kill -KILL "$pid" 2>/dev/null || true
done

rm -f "$PID_FILE_IN" "$PGID_FILE_IN"

if [ "$STOPPED" = "1" ]; then
  echo "Server process group and SGLang worker processes stopped."
else
  echo "No matching Higgs TTS server process found."
fi
'
}

status() {
  ensure_docker
  echo "Container: $NAME"
  if ! container_exists; then
    echo "  exists:  no"
    return 0
  fi

  if container_running; then
    echo "  running: yes"
  else
    echo "  running: no"
  fi

  if server_running_in_container; then
    echo "Server:    running"
    docker exec -i -e PID_FILE_IN="$PID_FILE_IN" "$NAME" /bin/zsh -lc 'echo "  pid:     $(cat "$PID_FILE_IN")"' || true
  else
    echo "Server:    stopped"
  fi

  echo "Model:     $MODEL"
  echo "Port:      $PORT"
  echo "Repo:      $PROJECT_ROOT/sglang-omni"
  echo "HF cache:  $HF_CACHE"
  echo "Log:       $PROJECT_ROOT/logs/higgs_tts_${PORT}.log"
  print_urls

  if command -v curl >/dev/null 2>&1; then
    if curl -fsS "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "HTTP:      /v1/models responds"
    else
      echo "HTTP:      not responding yet"
    fi
  fi
}

logs() {
  ensure_container
  docker exec -it \
    -e LOG_FILE_IN="$LOG_FILE_IN" \
    -e LINES="$LINES" \
    "$NAME" \
    /bin/zsh -lc '
mkdir -p "$(dirname "$LOG_FILE_IN")"
touch "$LOG_FILE_IN"
tail -n "$LINES" -f "$LOG_FILE_IN"
'
}

shell_into_container() {
  ensure_container
  docker exec -it "$NAME" /bin/zsh
}

test_request() {
  local out="${1:-output.wav}"
  need_cmd curl
  echo "Sending test request to http://localhost:${PORT}/v1/audio/speech -> ${out}"
  curl -fsS -X POST "http://localhost:${PORT}/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{"input":"Hello, this is a Higgs TTS test from SGLang Omni."}' \
    --output "$out"
  echo "Wrote: $out"
}

container_stop() {
  if container_exists; then
    docker stop "$NAME" >/dev/null
    echo "Stopped container: $NAME"
  else
    echo "Container does not exist: $NAME"
  fi
}

container_rm() {
  if container_exists; then
    stop_server || true
    docker rm -f "$NAME" >/dev/null
    echo "Removed container: $NAME"
    echo "Kept host files under: $PROJECT_ROOT"
    echo "Kept HF cache under:    $HF_CACHE"
  else
    echo "Container does not exist: $NAME"
  fi
}

cmd="${1:-}"
shift || true

case "$cmd" in
  deploy|start|stop|restart|container-stop|container-rm)
    acquire_lifecycle_lock
    ;;
esac

case "$cmd" in
  deploy) deploy "$@" ;;
  start) start_server "$@" ;;
  stop) stop_server "$@" ;;
  restart) stop_server; start_server ;;
  status) status "$@" ;;
  logs) logs "$@" ;;
  shell) shell_into_container "$@" ;;
  test) test_request "$@" ;;
  container-stop) container_stop "$@" ;;
  container-rm) container_rm "$@" ;;
  ""|-h|--help|help) usage ;;
  *) echo "Unknown command: $cmd" >&2; usage; exit 1 ;;
esac
