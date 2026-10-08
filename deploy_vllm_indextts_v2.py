import modal
import os
import json
import socket
import shlex
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import List, Dict, Optional

# The image supports Ada (L4/L40S) and RTX PRO 6000 Blackwell.
cuda_version = "13.0.0"
flavor = "devel" 
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

INDEXTTS_REPO_URL = "https://github.com/keboqi/index-tts-vllm.git"
CONFUCIUS_REPO_URL = "https://github.com/keboqi/Confucius4-TTS.git"
CONFUCIUS_IMAGE_DIR = "/app/Confucius4-TTS"
CONFUCIUS_APP_SUBDIR = "Confucius4-TTS"
CONFUCIUS_VENV_DIR = "/opt/confucius4tts-venv"
CONFUCIUS_PYTHON = f"{CONFUCIUS_VENV_DIR}/bin/python"
MOSS_TRANSCRIBE_VENV_DIR = "/opt/moss-transcribe-venv"
MOSS_TRANSCRIBE_PYTHON = f"{MOSS_TRANSCRIBE_VENV_DIR}/bin/python"
QWEN_ASR_VENV_DIR = "/opt/qwen-asr-venv"
QWEN_ASR_PYTHON = f"{QWEN_ASR_VENV_DIR}/bin/python"
CLEARVOICE_VENV_DIR = "/opt/clearvoice-venv"
CLEARVOICE_PYTHON = f"{CLEARVOICE_VENV_DIR}/bin/python"
CONFUCIUS_MODEL_REPO_ID = "netease-youdao/Confucius4-TTS"
CONFUCIUS_W2V_REPO_ID = "facebook/w2v-bert-2.0"
CONFUCIUS_BIGVGAN_REPO_ID = "nvidia/bigvgan_v2_22khz_80band_256x"
CONFUCIUS_CAMPPLUS_REPO_ID = "funasr/campplus"
CONFUCIUS_CAMPPLUS_FILENAME = "campplus_cn_common.bin"
CONFUCIUS_FASTAPI_CONFIG = "config/inference_config.modal.yaml"
INDEXTTS25_REPO_URL = "https://github.com/keboqi/indextts-2.5-vllm-omni-experiment.git"
# Pin the backend checkout so Modal's image cache is invalidated deliberately
# whenever the tested IndexTTS 2.5 integration revision changes.
INDEXTTS25_REPO_REF = "0a7d9aaeb9a0516c124669966aeed907e29b811d"
INDEXTTS25_IMAGE_DIR = "/app/index-tts-2.5-vllm-omni-experiment"
INDEXTTS25_APP_SUBDIR = "index-tts-2.5-vllm-omni-experiment"
INDEXTTS25_VENV_DIR = "/opt/indextts25-venv"
INDEXTTS25_PYTHON = f"{INDEXTTS25_VENV_DIR}/bin/python"
INDEXTTS25_VLLM = f"{INDEXTTS25_VENV_DIR}/bin/vllm"
INDEXTTS25_TORCH_BACKEND = "cu130"
INDEXTTS25_MODEL_REPO_ID = "IndexTeam/IndexTTS-2.5"
INDEXTTS25_W2V_REPO_ID = "facebook/w2v-bert-2.0"
INDEXTTS25_CAMPPLUS_REPO_ID = "funasr/campplus"
INDEXTTS25_BIGVGAN_REPO_ID = "nvidia/bigvgan_v2_22khz_80band_256x"
STABLE_AUDIO3_REPOS = {
    "medium": "stabilityai/stable-audio-3-medium",
    "small-music": "stabilityai/stable-audio-3-small-music",
    "small-sfx": "stabilityai/stable-audio-3-small-sfx",
}
MOSS_TRANSCRIBE_REPO_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
HY_MT_TRANSLATION_REPO_ID = "tencent/Hy-MT2-1.8B"
RUNTIME_SOURCE_DIR = "/opt/indextts-runtime-source"
DEPLOY_SOURCE_ROOT = Path(__file__).resolve().parent
# Increment only when applying repository/model changes made in the web UI.
# Ordinary code/image changes already invalidate Modal's snapshots.
SNAPSHOT_REVISION = "1"


def _ignore_runtime_source(path) -> bool:
    """Ship application source/assets, never checkpoints, credentials or outputs."""
    relative = Path(path).resolve().relative_to(DEPLOY_SOURCE_ROOT)
    if not relative.parts:
        return False
    if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
        return True
    directories = {"indextts", "indextts_web", "static", "tools", "fonts", "examples", "assets"}
    if relative.parts[0] in directories:
        return False
    return not (len(relative.parts) == 1 and relative.suffix in {".py", ".html", ".sh"}
                and not relative.name.startswith("deploy_"))

# Only the deploying process has the checkout and dependency manifests.
# Modal containers import this module to register functions in the existing image.
image = None
if modal.is_local():
    from indextts_web.infrastructure.modal_dependencies import install_main_dependencies

    # Create Modal image for IndexTTS v2 with vLLM optimization
    image = (
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
        .apt_install(
            "ffmpeg",
            "git",
            "wget",
            "build-essential",
            "gcc",
            "g++",
            "cmake",
            "sox",
            "libsox-fmt-all",
            "libgl1",
            "libglib2.0-0",
            "nodejs",
            "npm",
        )
        .env({
            "CUDA_HOME": "/usr/local/cuda",
            "CUDA_PATH": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0;12.0",
            "FORCE_CUDA": "1",
            "CXX": "g++",
            "CC": "gcc",
        })
        .run_commands("pip install --upgrade pip setuptools wheel")
        .run_commands(
            f"git clone {INDEXTTS_REPO_URL} /app/index-tts-vllm"
        )
    )

    # The local manifests are copied into a build layer before installation. A
    # dependency edit therefore invalidates the cache even when git clone is cached.
    # Resolve TTS and main-process ASR together, including vLLM's exact Torch ABI.
    image = install_main_dependencies(image, DEPLOY_SOURCE_ROOT)
    image = (
        image
        .run_commands(
            "python -m pip check",
            # Qwen-TTS requires the CPU distribution and audio-separator requires
            # the GPU distribution. They share import paths; put GPU bindings last.
            "python -m pip install --force-reinstall --no-deps "
            "-c /app/index-tts-vllm/constraints-main.txt onnxruntime-gpu",
            "python -c \"import torch; "
            "assert torch.version.cuda, 'IndexTTS installed CPU-only Torch'; "
            "print('IndexTTS CUDA Torch:', torch.__version__, torch.version.cuda); "
            "import onnxruntime; assert 'CUDAExecutionProvider' in "
            "onnxruntime.get_available_providers(), 'Audio separator installed CPU-only ONNX Runtime'\""
        )
        .run_commands(
            f"git clone {CONFUCIUS_REPO_URL} {CONFUCIUS_IMAGE_DIR}"
        )
        .run_commands(
            f"python -m venv {CONFUCIUS_VENV_DIR}",
            f"{CONFUCIUS_PYTHON} -m pip install --upgrade pip setuptools wheel",
            f"cd {CONFUCIUS_IMAGE_DIR} && {CONFUCIUS_PYTHON} -m pip install -r requirements.txt",
            f"cd {CONFUCIUS_IMAGE_DIR} && {CONFUCIUS_PYTHON} -m pip install --force-reinstall -r requirements-cu128.txt",
            f"cd {CONFUCIUS_IMAGE_DIR} && {CONFUCIUS_PYTHON} -m pip install -r requirements-vllm.txt",
            f"{CONFUCIUS_PYTHON} -m pip install \"numpy<2\" \"torchcodec==0.9.*\"",
        )
        .run_commands(
            "pip install uv",
            f"git clone {INDEXTTS25_REPO_URL} {INDEXTTS25_IMAGE_DIR}",
            f"git -C {INDEXTTS25_IMAGE_DIR} fetch origin {INDEXTTS25_REPO_REF}",
            f"git -C {INDEXTTS25_IMAGE_DIR} checkout --detach {INDEXTTS25_REPO_REF}",
            "uv python install 3.11",
            f"uv venv --python 3.11 --seed {INDEXTTS25_VENV_DIR}",
            # Image builders have no GPU, so auto detection installs CPU-only
            # PyTorch. Select the CUDA wheel explicitly for the CUDA 13 image.
            f"uv pip install --python {INDEXTTS25_PYTHON} 'vllm==0.27.0' "
            f"--torch-backend={INDEXTTS25_TORCH_BACKEND}",
            f"uv pip install --python {INDEXTTS25_PYTHON} -e '{INDEXTTS25_IMAGE_DIR}[indextts2]'",
            f"uv pip install --python {INDEXTTS25_PYTHON} -e "
            f"'{INDEXTTS25_IMAGE_DIR}/experiments/indextts25_backend_compat'",
            f"uv pip install --python {INDEXTTS25_PYTHON} 'huggingface_hub[cli]'",
            f"{INDEXTTS25_PYTHON} "
            f"{INDEXTTS25_IMAGE_DIR}/experiments/indextts25_backend_compat/src/"
            "indextts25_compat/patch_flashinfer.py",
            # Importing flashinfer initializes CUDA and cannot run in Modal's
            # GPU-less image builder. The patcher above validates its target; here
            # verify that the installed Torch wheel has CUDA support without
            # initializing a CUDA device.
            f"{INDEXTTS25_PYTHON} -c \"import importlib.metadata, torch; "
            "assert torch.version.cuda, 'IndexTTS 2.5 installed CPU-only Torch'; "
            "print('IndexTTS 2.5 CUDA Torch:', torch.version.cuda, "
            "'FlashInfer:', importlib.metadata.version('flashinfer-python'))\"",
        )
        .run_commands(
            # The PyPI stable-audio-tools wheel is too old for Stable Audio 3
            # configs and also pins older torch builds. Use current source and
            # install its runtime dependencies in the constrained main environment.
            "pip install --force-reinstall --no-deps --ignore-requires-python "
            "git+https://github.com/Stability-AI/stable-audio-tools.git",
        )
        .run_commands(
            # These official wheels match Torch 2.8/Python 3.12. Select its actual
            # C++ ABI and avoid an incompatible CUDA 13 source-build fallback.
            "python -c \"import subprocess, sys, torch; "
            "assert torch.__version__.split('+')[0] == '2.8.0'; "
            "assert torch.version.cuda and torch.version.cuda.startswith('12.'); "
            "abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper(); "
            "wheel = 'https://github.com/Dao-AILab/flash-attention/releases/download/' "
            "+ 'v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abi' "
            "+ abi + '-cp312-cp312-linux_x86_64.whl'; "
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', wheel], check=True)\""
        )
        .add_local_file(DEPLOY_SOURCE_ROOT / "requirements-clearvoice.txt",
                        "/opt/requirements-clearvoice.txt", copy=True)
        .run_commands(
            # This environment must not see the incompatible Roformer packages.
            f"python -m venv {CLEARVOICE_VENV_DIR}",
            f"{CLEARVOICE_PYTHON} -m pip install --upgrade pip setuptools wheel",
            f"{CLEARVOICE_PYTHON} -m pip install -r /opt/requirements-clearvoice.txt",
            f"{CLEARVOICE_PYTHON} -c \"from clearvoice import ClearVoice; import torch; "
            "assert torch.version.cuda, 'ClearVoice installed CPU-only Torch'; "
            "print('ClearVoice environment ready')\"",
        )
        .run_commands(
            f"python -m venv --system-site-packages {MOSS_TRANSCRIBE_VENV_DIR}",
            f"{MOSS_TRANSCRIBE_PYTHON} -m pip install --upgrade pip setuptools wheel",
            f"{MOSS_TRANSCRIBE_PYTHON} -m pip install "
            "'transformers>=5.6.0,<6' av librosa soundfile soxr "
            "fastapi uvicorn python-multipart",
            f"{MOSS_TRANSCRIBE_PYTHON} -m pip install --no-deps "
            "git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git",
        )
        .run_commands(
            "npm install -g n",
            "n 22",
            "node --version",
        )
        .run_commands(
            # Reuse the CUDA/audio/diarization stack, but install Qwen ASR's exact
            # Transformers dependency inside its own venv, never in the TTS env.
            f"python -m venv --system-site-packages {QWEN_ASR_VENV_DIR}",
            f"{QWEN_ASR_PYTHON} -m pip install --upgrade pip setuptools wheel",
            f"{QWEN_ASR_PYTHON} -m pip install 'qwen-asr==0.0.6' 'transformers==4.57.6'",
            f"{QWEN_ASR_PYTHON} -c \"from qwen_asr import Qwen3ASRModel; "
            "from omnivad import OmniVAD; import litai, transformers, torch; "
            "assert transformers.__version__ == '4.57.6'; "
            "assert torch.version.cuda, 'Qwen ASR requires CUDA Torch'; "
            "print('Qwen3-ASR environment ready')\"",
        )
        # copy=True makes source changes part of the image/snapshot identity. Use
        # these exact local sources at startup, not the code in the model Volume.
        .add_local_dir(str(DEPLOY_SOURCE_ROOT), RUNTIME_SOURCE_DIR,
                       copy=True, ignore=_ignore_runtime_source)
        .run_commands(
            f"cd {RUNTIME_SOURCE_DIR} && python -c "
            "\"from indextts.utils.maskgct.models.tts.maskgct.llama_nar "
            "import DiffLlama; print('IndexTTS Transformers compatibility check passed')\""
        )
    )

    # Runtime settings come after every build step so changes reuse dependency layers.
    image = image.env({
        # vLLM sleep mode uses its CUDA memory pool; PyTorch expandable
        # segments are incompatible with that allocator.
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",

        # Cache directories for faster subsequent runs
        "HF_HOME": "/persistent_cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/persistent_cache/huggingface/hub",
        "TORCH_HOME": "/persistent_cache/torch",
        "TRANSFORMERS_CACHE": "/persistent_cache/transformers",
        "CUDA_CACHE_PATH": "/persistent_cache/cuda_cache",
        "VLLM_CACHE": "/persistent_cache/vllm_cache",
        "TRITON_CACHE_DIR": "/persistent_cache/triton",
        "VLLM_SERVER_DEV_MODE": "1",
        # Modal containers do not provide a Docker daemon. Run MOSS directly
        # through Transformers.
        "MOSS_TRANSCRIBE_MANAGE_BACKEND": "0",
        "MOSS_TRANSCRIBE_BACKEND": "http",
        "MOSS_TRANSCRIBE_DEVICE": "cuda:0",
        "MOSS_TRANSCRIBE_MODEL": "/persistent_app/checkpoints/MOSS-Transcribe-Diarize",
        "MOSS_TRANSCRIBE_SGLANG_URL": "http://127.0.0.1:8003",
        "HY_MT_TRANSLATION_LOCAL_DIR": "/persistent_app/checkpoints/hy-mt",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCH_NCCL_ENABLE_MONITORING": "0",
        "TORCH_CPP_LOG_LEVEL": "ERROR",
        # Disable ONNX Runtime telemetry before application imports.
        "ORT_DISABLE_TELEMETRY": "1",
        "CLEARVOICE_PYTHON": CLEARVOICE_PYTHON,
        "QWEN_OMNIVAD_PYTHON": QWEN_ASR_PYTHON,
        "QWEN_OMNIVAD_MODEL_DIR": "/persistent_app/checkpoints/qwen_omnivad",
        "QWEN_OMNIVAD_CACHE_DIR": "/persistent_cache/qwen_omnivad",
        "INDEXTTS_SETUP_SNAPSHOT_REVISION": SNAPSHOT_REVISION,
    })

app = modal.App("audio-studio", image=image)

# Create persistent storage volumes
app_storage = modal.Volume.from_name("audio-studio-app", create_if_missing=True)
cache_storage = modal.Volume.from_name("audio-studio-cache", create_if_missing=True)

# Configuration
PERSISTENT_APP_DIR = "/persistent_app"
PERSISTENT_CACHE_DIR = "/persistent_cache"
MOSS_TRANSCRIBE_PERSISTENT_DIR = (
    f"{PERSISTENT_APP_DIR}/checkpoints/MOSS-Transcribe-Diarize"
)
HY_MT_TRANSLATION_PERSISTENT_DIR = f"{PERSISTENT_APP_DIR}/checkpoints/hy-mt"
CONFUCIUS_PERSISTENT_REPO_DIR = f"{PERSISTENT_APP_DIR}/{CONFUCIUS_APP_SUBDIR}"
INDEXTTS25_PERSISTENT_REPO_DIR = f"{PERSISTENT_APP_DIR}/{INDEXTTS25_APP_SUBDIR}"
INDEXTTS25_PERSISTENT_MODEL_DIR = f"{PERSISTENT_APP_DIR}/checkpoints/IndexTTS-2.5"
INDEXTTS25_PERSISTENT_DATA_DIR = f"{PERSISTENT_CACHE_DIR}/indextts25"
VLLM_PORT = 8000
CONFUCIUS_PORT = 8001
INDEXTTS25_PORT = 8092
MOSS_TRANSCRIBE_PORT = 8003
SNAPSHOT_STARTUP_TIMEOUT = 1800
SNAPSHOT_REQUEST_TIMEOUT = 900
INTERNAL_TOKEN_ENV = "INDEXTTS_INTERNAL_TOKEN"
DEFAULT_TTS_BACKEND = "index"
CONFUCIUS_STARTUP_TIMEOUT = 1200
CONFUCIUS_REQUEST_TIMEOUT = 900
INDEXTTS25_STARTUP_TIMEOUT = 1800
INDEXTTS25_REQUEST_TIMEOUT = 900

STABLE_AUDIO3_VARIANTS = tuple(STABLE_AUDIO3_REPOS)


def _ensure_confucius_vllm_patch_compatibility(confucius_repo_path: Path) -> Dict[str, object]:
    """Keep the persistent Confucius checkout compatible with current vLLM."""
    patch_path = confucius_repo_path / "confuciustts" / "llm" / "vllm_patch.py"
    status: Dict[str, object] = {
        "success": False,
        "changed": False,
        "path": str(patch_path),
        "message": "",
    }
    if not patch_path.exists():
        status["message"] = f"Confucius vLLM patch file not found: {patch_path}"
        return status

    old_signature = """    def _prepare_inputs_with_confucius_positions(
        self,
        scheduler_output,
        num_scheduled_tokens,
        *args,
        **kwargs,
    ):
        result = current_prepare(
            self,
            scheduler_output,
            num_scheduled_tokens,
            *args,
            **kwargs,
        )
"""
    new_signature = """    def _prepare_inputs_with_confucius_positions(
        self,
        scheduler_output,
        *args,
        **kwargs,
    ):
        result = current_prepare(
            self,
            scheduler_output,
            *args,
            **kwargs,
        )
"""
    compatibility_marker = "        num_scheduled_tokens = None\n        if args:\n"
    req_indices_line = "        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)\n"
    compatibility_block = """        num_scheduled_tokens = None
        if args:
            candidate = args[0]
            if isinstance(candidate, np.ndarray):
                num_scheduled_tokens = candidate
        if num_scheduled_tokens is None and isinstance(result, tuple) and len(result) >= 4:
            candidate = result[3]
            if isinstance(candidate, np.ndarray):
                num_scheduled_tokens = candidate
        if num_scheduled_tokens is None:
            req_ids_for_tokens = list(self.input_batch.req_ids[:num_reqs])
            num_scheduled_tokens = np.array(
                [scheduler_output.num_scheduled_tokens[req_id] for req_id in req_ids_for_tokens],
                dtype=np.int32,
            )

"""

    text = patch_path.read_text(encoding="utf-8")
    updated_text = text
    changed = False

    if old_signature in updated_text:
        updated_text = updated_text.replace(old_signature, new_signature, 1)
        changed = True
    elif "        num_scheduled_tokens,\n        *args,\n" in updated_text:
        status["message"] = "Confucius vLLM wrapper signature did not match the expected patch shape"
        return status

    if compatibility_marker not in updated_text:
        if req_indices_line not in updated_text:
            status["message"] = "Confucius vLLM patch insertion point was not found"
            return status
        updated_text = updated_text.replace(req_indices_line, compatibility_block + req_indices_line, 1)
        changed = True

    if changed:
        patch_path.write_text(updated_text, encoding="utf-8")
        status["changed"] = True
        status["message"] = "Confucius vLLM patch compatibility update applied"
    else:
        status["message"] = "Confucius vLLM patch is already compatible"
    status["success"] = True
    return status

@app.function(
    image=image, timeout=86400, cpu=4.0, memory=32768, max_containers=1,
    volumes={PERSISTENT_APP_DIR: app_storage, PERSISTENT_CACHE_DIR: cache_storage},
    secrets=[modal.Secret.from_name("custom-secret")],
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def prepare_model():
    """Web UI around the existing CPU model preparation flow."""
    import sys
    sys.path.insert(0, RUNTIME_SOURCE_DIR)
    from indextts_web.infrastructure.model_manager import create_manager_app
    from indextts_web.infrastructure.model_setup import ModelSetup

    def get_state():
        app_storage.reload()
        cache_storage.reload()
        return ModelSetup(Path(PERSISTENT_APP_DIR), Path(RUNTIME_SOURCE_DIR)).status()

    def execute_operation(action, target, emit):
        app_storage.reload()
        cache_storage.reload()
        setup = ModelSetup(Path(PERSISTENT_APP_DIR), Path(RUNTIME_SOURCE_DIR),
                           emit=emit, patch_confucius=_ensure_confucius_vllm_patch_compatibility)
        try:
            setup.execute(action, target)
        finally:
            # Keep partial downloads for resume, including after a failure.
            cache_storage.commit()
            app_storage.commit()

    return create_manager_app(get_state=get_state, execute_operation=execute_operation)


def legacy_serve_without_snapshot():
    """
    Serve the IndexTTS v2 FastAPI application by running python fastapi_webui_v2.py directly.
    """
    import os
    from pathlib import Path
    import subprocess
    
    print("🚀 Starting IndexTTS v2 vLLM FastAPI WebUI...")
    
    # ========================================================================
    # STEP 1: Setup Persistent Cache System
    # ========================================================================
    print("\n💾 Configuring persistent cache system...")
    print("   📌 CUDA kernels will compile on FIRST startup (needs GPU)")
    print("   📌 Subsequent startups will reuse cached artifacts from persistent volume\n")
    
    # 1.1: Set cache environment variables (before any Python imports that use them)
    cache_env_vars = {
        "HF_HOME": "/persistent_cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/persistent_cache/huggingface/hub",
        "TORCH_HOME": "/persistent_cache/torch",
        "TRANSFORMERS_CACHE": "/persistent_cache/transformers",
        "CUDA_CACHE_PATH": "/persistent_cache/cuda_cache",
        "VLLM_CACHE": "/persistent_cache/vllm_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/persistent_cache/torch_compile_cache",
        "TRITON_CACHE_DIR": "/persistent_cache/triton",
        "XDG_CACHE_HOME": "/persistent_cache",
        "ORT_DISABLE_TELEMETRY": "1",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
    }
    
    print("   Setting environment variables:")
    for key, value in cache_env_vars.items():
        os.environ[key] = value
        print(f"      ✅ {key}={value}")
    
    # 1.2: Create cache directories in persistent volume
    cache_dirs = [
        "/persistent_cache/huggingface",
        "/persistent_cache/torch", 
        "/persistent_cache/transformers",
        "/persistent_cache/cuda_cache",
        "/persistent_cache/vllm_cache",
        "/persistent_cache/torch_compile_cache",
        "/persistent_cache/confucius",
        "/persistent_cache/triton",
        INDEXTTS25_PERSISTENT_DATA_DIR,
    ]
    
    print("\n   Creating cache directories:")
    for cache_dir in cache_dirs:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"      📁 {cache_dir}")
    
    # 1.3: Create symlinks from standard cache locations to persistent volume
    local_cache_map = {
        "/root/.cache/huggingface": "/persistent_cache/huggingface",
        "/root/.cache/torch": "/persistent_cache/torch",
        "/root/.cache/transformers": "/persistent_cache/transformers",
        "/root/.cache/vllm": "/persistent_cache/vllm_cache"
    }
    
    print("\n   Creating cache symlinks:")
    for local_path, persistent_path in local_cache_map.items():
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        if os.path.exists(local_path):
            if os.path.islink(local_path):
                os.unlink(local_path)
            else:
                import shutil
                shutil.rmtree(local_path)
        
        os.symlink(persistent_path, local_path)
        print(f"      🔗 {local_path} -> {persistent_path}")
    
    # ========================================================================
    # STEP 2: Verify Application and Models
    # ========================================================================
    print("\n📂 Verifying application and models...")
    
    # 2.1: Verify persistent application exists  
    persistent_app_path = Path(PERSISTENT_APP_DIR)
    if not persistent_app_path.exists():
        print("❌ Persistent application not found! Open the prepare_model web manager first.")
        raise FileNotFoundError(f"Application not found at {persistent_app_path}")
    
    print(f"   ✅ Application: {persistent_app_path}")
    
    # 2.2: Verify model files exist
    checkpoints_dir = persistent_app_path / "checkpoints"
    if not checkpoints_dir.exists():
        print("❌ Model checkpoints not found!")
        raise FileNotFoundError(f"Checkpoints missing at {checkpoints_dir}")
    
    print(f"   ✅ Checkpoints: {checkpoints_dir}")
    
    # ========================================================================
    # STEP 3: Setup Application Environment
    # ========================================================================
    print("\n🔧 Configuring application environment...")
    
    # 3.1: Change to persistent app directory
    os.chdir(str(persistent_app_path))
    print(f"   📁 Working directory: {os.getcwd()}")
    
    # 3.2: Setup Python path for vLLM worker processes
    os.environ["PYTHONPATH"] = str(persistent_app_path)
    os.environ["PYTHONUNBUFFERED"] = "1"
    print(f"   🐍 PYTHONPATH: {os.environ['PYTHONPATH']}")
    
    # 3.3: Setup Qwen3-TTS Voice Design model path (use local pre-downloaded model)
    voice_design_model_path = persistent_app_path / "checkpoints" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    if voice_design_model_path.exists():
        os.environ["QWEN3_VOICE_DESIGN_MODEL"] = str(voice_design_model_path)
        print(f"   🎤 QWEN3_VOICE_DESIGN_MODEL: {voice_design_model_path}")
    else:
        print(f"   ⚠️ Voice Design model not found at {voice_design_model_path}, will use HuggingFace download")
    
    # ========================================================================
    stable_audio_root = checkpoints_dir / "stable-audio-3"
    print(f"   Stable Audio 3 root: {stable_audio_root}")
    for key in STABLE_AUDIO3_VARIANTS:
        path = stable_audio_root / key
        ready = (
            (path / "model_config.json").exists()
            and ((path / "model.safetensors").exists() or (path / "model.ckpt").exists())
        )
        print(f"      {key}: {'ready' if ready else 'missing'} ({path})")

    # STEP 4: Start FastAPI Server
    # ========================================================================
    print("\n🚀 Starting FastAPI server...")
    
    persistent_app_path = _configure_gpu_runtime(persistent_app_path)
    cmd = _build_webui_command(persistent_app_path)
    
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Working dir: {os.getcwd()}\n")
    print("="*80)
    print("🎉 IndexTTS v2 vLLM initialization complete!")
    print("="*80 + "\n")
    
    # Start the FastAPI server (this will keep running)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.Popen(cmd, cwd=str(persistent_app_path), env=env)


def _local_url(path: str) -> str:
    return f"http://127.0.0.1:{VLLM_PORT}{path}"


def _call_local_json(
    path: str,
    *,
    method: str = "GET",
    timeout: int = 30,
    payload: Optional[Dict] = None,
    internal: bool = False,
) -> Dict:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if internal:
        headers["X-IndexTTS-Internal-Token"] = os.environ[INTERNAL_TOKEN_ENV]

    request = urllib.request.Request(
        _local_url(path),
        data=body,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_body = response.read().decode("utf-8")
    return json.loads(response_body) if response_body else {}


def _wait_ready(proc: subprocess.Popen, *, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = None

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"FastAPI server exited with code {proc.returncode}")

        try:
            socket.create_connection(("127.0.0.1", VLLM_PORT), timeout=1).close()
            health = _call_local_json("/health", timeout=10)
            if health.get("ready") is not True:
                raise urllib.error.URLError("TTS model is not ready")
            print("FastAPI server is ready.")
            return
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(2)

    raise TimeoutError(
        f"Timed out waiting {timeout_seconds}s for FastAPI server readiness. "
        f"Last error: {last_error}"
    )


def _start_moss_transcribe_server(persistent_app_path: Path) -> subprocess.Popen:
    """Start the isolated pure-Python MOSS service (no Docker daemon)."""
    server_script = persistent_app_path / "moss_transcribe_server.py"
    if not server_script.exists():
        raise FileNotFoundError(f"MOSS server script not found: {server_script}")
    if not Path(MOSS_TRANSCRIBE_PYTHON).exists():
        raise FileNotFoundError(
            f"MOSS virtual environment Python not found: {MOSS_TRANSCRIBE_PYTHON}"
        )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(persistent_app_path)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        MOSS_TRANSCRIBE_PYTHON,
        "-m",
        "uvicorn",
        "moss_transcribe_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(MOSS_TRANSCRIBE_PORT),
    ]
    print(f"Starting isolated MOSS transcription server: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=str(persistent_app_path), env=env)


def _wait_moss_ready(proc: subprocess.Popen, *, timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{MOSS_TRANSCRIBE_PORT}/v1/models"
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"MOSS transcription server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if 200 <= response.status < 300:
                    print("MOSS transcription server is ready.")
                    return
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for MOSS server. Last error: {last_error}")


def _build_confucius_start_command(confucius_repo_path: Path, gpu_profile) -> str:
    confucius_config_path = confucius_repo_path / CONFUCIUS_FASTAPI_CONFIG
    confucius_vllm_dir = confucius_repo_path / "checkpoints" / "t2s-vllm"
    confucius_output_dir = confucius_repo_path / "outputs" / "api"
    confucius_compile_cache_dir = (
        Path(PERSISTENT_CACHE_DIR) / "confucius" / gpu_profile.cache_key / "torchinductor"
    )
    confucius_warmup_voice = confucius_repo_path / "resources" / "voice.mp3"

    parts = [
        "env",
        f"PYTHONPATH={confucius_repo_path}{os.pathsep}{confucius_repo_path.parent}",
        CONFUCIUS_PYTHON,
        "-u",
        "-m",
        "indextts_web.services.tts.confucius_launcher",
        "--host",
        "127.0.0.1",
        "--port",
        str(CONFUCIUS_PORT),
        "--config",
        str(confucius_config_path),
        "--vllm-model-dir",
        str(confucius_vllm_dir),
        "--vllm-gpu-memory-utilization",
        str(gpu_profile.confucius.gpu_memory_utilization),
        "--vllm-attention-backend",
        "FLASHINFER",
        "--vllm-prefix-mode",
        "auto",
        "--vllm-latent-mode",
        "auto",
        "--output-dir",
        str(confucius_output_dir),
        "--compile-cache-dir",
        str(confucius_compile_cache_dir),
        "--warmup",
        "--warmup-mode",
        "background" if gpu_profile.name == "96gb" else "foreground",
        "--warmup-prompt-wav",
        str(confucius_warmup_voice),
        "--compile-s2a" if gpu_profile.use_torch_compile else "--no-compile-s2a",
        "--gpu-stage-concurrency",
        "1",
        "--postprocess-concurrency",
        "2",
        "--inference-workers",
        "1",
    ]
    return " ".join(shlex.quote(str(part)) for part in parts)


def _build_indextts25_start_command(indextts25_repo_path: Path, gpu_profile) -> str:
    from indextts_web.gpu_profiles import write_omni_deploy_config

    deploy_config = Path(os.environ["INDEXTTS25_DEPLOY_CONFIG"]) if os.environ.get("INDEXTTS25_DEPLOY_CONFIG") else (
        write_omni_deploy_config(
            indextts25_repo_path / "vllm_omni" / "deploy" / "indextts2_5.yaml",
            Path(INDEXTTS25_PERSISTENT_DATA_DIR) / "deploy", gpu_profile,
        )
    )
    if not deploy_config.is_file():
        raise FileNotFoundError(f"IndexTTS 2.5 deployment config missing: {deploy_config}")
    data_dir = Path(INDEXTTS25_PERSISTENT_DATA_DIR)
    parts = [
        "env",
        "FLASHINFER_DISABLE_VERSION_CHECK=1",
        f"PYTHONPATH={indextts25_repo_path}",
        f"HF_HOME={data_dir / 'cache' / 'huggingface'}",
        f"SPEAKER_SAMPLES_DIR={data_dir / 'speakers'}",
        f"SPEAKER_CACHE_DIR={data_dir / 'cache' / 'speaker-conditioning'}",
        "SPEAKER_CACHE_VERSION=indextts25-v1",
        f"TORCHINDUCTOR_CACHE_DIR={data_dir / 'cache' / gpu_profile.cache_key / 'torchinductor'}",
        f"TRITON_CACHE_DIR={data_dir / 'cache' / gpu_profile.cache_key / 'triton'}",
        f"CUDA_CACHE_PATH={data_dir / 'cache' / gpu_profile.cache_key / 'cuda'}",
        INDEXTTS25_VLLM,
        "serve",
        INDEXTTS25_PERSISTENT_MODEL_DIR,
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(INDEXTTS25_PORT),
        "--served-model-name",
        INDEXTTS25_MODEL_REPO_ID,
        "--trust-remote-code",
        "--enable-sleep-mode",
        "--log-stats",
        "--deploy-config",
        str(deploy_config),
    ]
    return " ".join(shlex.quote(str(part)) for part in parts)


def _build_webui_command(persistent_app_path: Path, gpu_profile=None) -> List[str]:
    """Build one launch command shared by snapshot and legacy entry points."""
    if gpu_profile is None:
        from indextts_web.gpu_profiles import runtime_gpu_profile
        gpu_profile = runtime_gpu_profile(modal=True)
    return [
        "python",
        "-u",
        "fastapi_webui_v2.py",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--model_dir",
        "checkpoints",
        "--gpu_memory_utilization",
        str(gpu_profile.index.gpu_memory_utilization),
        "--qwenemo_gpu_memory_utilization",
        str(gpu_profile.emotion.gpu_memory_utilization),
        "--tts_backend",
        DEFAULT_TTS_BACKEND,
        "--confucius_repo_dir",
        str(persistent_app_path / CONFUCIUS_APP_SUBDIR),
        "--confucius_host",
        "127.0.0.1",
        "--confucius_port",
        str(CONFUCIUS_PORT),
        "--confucius_start_command",
        _build_confucius_start_command(persistent_app_path / CONFUCIUS_APP_SUBDIR, gpu_profile),
        "--confucius_start_timeout",
        str(CONFUCIUS_STARTUP_TIMEOUT),
        "--confucius_request_timeout",
        str(CONFUCIUS_REQUEST_TIMEOUT),
        "--confucius_vllm_gpu_memory_utilization",
        str(gpu_profile.confucius.gpu_memory_utilization),
        "--confucius_attach_stdio",
        "--confucius_keepalive_interval",
        "60",
        "--confucius_unhealthy_grace",
        "30",
        "--indextts25_repo_dir",
        str(persistent_app_path / INDEXTTS25_APP_SUBDIR),
        "--indextts25_model_dir",
        INDEXTTS25_PERSISTENT_MODEL_DIR,
        "--indextts25_data_dir",
        INDEXTTS25_PERSISTENT_DATA_DIR,
        "--indextts25_host",
        "127.0.0.1",
        "--indextts25_port",
        str(INDEXTTS25_PORT),
        "--indextts25_served_model_name",
        INDEXTTS25_MODEL_REPO_ID,
        "--indextts25_start_command",
        _build_indextts25_start_command(persistent_app_path / INDEXTTS25_APP_SUBDIR, gpu_profile),
        "--indextts25_start_timeout",
        str(INDEXTTS25_STARTUP_TIMEOUT),
        "--indextts25_request_timeout",
        str(INDEXTTS25_REQUEST_TIMEOUT),
        "--indextts25_attach_stdio",
        "--indextts25_keepalive_interval",
        "60",
        "--indextts25_unhealthy_grace",
        "30",
        "--indextts25_max_parallel_segments",
        str(gpu_profile.parallel_segments),
        "--use_torch_compile" if gpu_profile.use_torch_compile else "--no-use_torch_compile",
    ]


def _updated_runtime_source(persistent_app_path: Path) -> Path:
    import sys
    sys.path.insert(0, RUNTIME_SOURCE_DIR)
    from indextts_web.infrastructure.model_setup import repository_source

    return repository_source(persistent_app_path, Path(RUNTIME_SOURCE_DIR))


def _configure_persistent_runtime():
    from pathlib import Path

    print("Starting IndexTTS v2 vLLM FastAPI WebUI with Modal snapshots...")

    cache_env_vars = {
        "HF_HOME": "/persistent_cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/persistent_cache/huggingface/hub",
        "TORCH_HOME": "/persistent_cache/torch",
        "TRANSFORMERS_CACHE": "/persistent_cache/transformers",
        "CUDA_CACHE_PATH": "/persistent_cache/cuda_cache",
        "VLLM_CACHE": "/persistent_cache/vllm_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/persistent_cache/torch_compile_cache",
        "TRITON_CACHE_DIR": "/persistent_cache/triton",
        "XDG_CACHE_HOME": "/persistent_cache",
        # Set before imports/subprocesses; the Python telemetry API is too late
        # to prevent ORT from creating its device ID and offline cache files.
        "ORT_DISABLE_TELEMETRY": "1",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "INDEXTTS_ENABLE_VLLM_SLEEP_MODE": "1",
        "MOSS_TRANSCRIBE_MANAGE_BACKEND": os.environ.get(
            "MOSS_TRANSCRIBE_MANAGE_BACKEND", "0"
        ),
        "MOSS_TRANSCRIBE_BACKEND": os.environ.get(
            "MOSS_TRANSCRIBE_BACKEND", "http"
        ),
        "MOSS_TRANSCRIBE_DEVICE": os.environ.get(
            "MOSS_TRANSCRIBE_DEVICE", "cuda:0"
        ),
        "MOSS_TRANSCRIBE_MODEL": os.environ.get(
            "MOSS_TRANSCRIBE_MODEL", MOSS_TRANSCRIBE_PERSISTENT_DIR
        ),
        "MOSS_TRANSCRIBE_SGLANG_URL": os.environ.get(
            "MOSS_TRANSCRIBE_SGLANG_URL",
            f"http://127.0.0.1:{MOSS_TRANSCRIBE_PORT}",
        ),
        "HY_MT_TRANSLATION_LOCAL_DIR": os.environ.get(
            "HY_MT_TRANSLATION_LOCAL_DIR", HY_MT_TRANSLATION_PERSISTENT_DIR
        ),
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "TORCH_NCCL_ENABLE_MONITORING": "0",
        "TORCH_CPP_LOG_LEVEL": "ERROR",
    }

    print("Configuring persistent cache environment:")
    for key, value in cache_env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")

    cache_dirs = [
        "/persistent_cache/huggingface",
        "/persistent_cache/torch",
        "/persistent_cache/transformers",
        "/persistent_cache/cuda_cache",
        "/persistent_cache/vllm_cache",
        "/persistent_cache/torch_compile_cache",
        "/persistent_cache/confucius",
        "/persistent_cache/triton",
        INDEXTTS25_PERSISTENT_DATA_DIR,
    ]
    for cache_dir in cache_dirs:
        os.makedirs(cache_dir, exist_ok=True)

    local_cache_map = {
        "/root/.cache/huggingface": "/persistent_cache/huggingface",
        "/root/.cache/torch": "/persistent_cache/torch",
        "/root/.cache/transformers": "/persistent_cache/transformers",
        "/root/.cache/vllm": "/persistent_cache/vllm_cache",
    }
    for local_path, persistent_path in local_cache_map.items():
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.exists(local_path):
            if os.path.islink(local_path) or os.path.isfile(local_path):
                os.unlink(local_path)
            else:
                import shutil

                shutil.rmtree(local_path)
        os.symlink(persistent_path, local_path)
        print(f"  cache link: {local_path} -> {persistent_path}")

    persistent_app_path = Path(PERSISTENT_APP_DIR)
    if not persistent_app_path.exists():
        raise FileNotFoundError(
            f"Application not found at {persistent_app_path}. Open the prepare_model web manager first."
        )

    checkpoints_dir = persistent_app_path / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints missing at {checkpoints_dir}")

    os.chdir(str(persistent_app_path))
    os.environ["PYTHONPATH"] = str(persistent_app_path)
    os.environ["PYTHONUNBUFFERED"] = "1"

    voice_design_model_path = checkpoints_dir / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    if voice_design_model_path.exists():
        os.environ["QWEN3_VOICE_DESIGN_MODEL"] = str(voice_design_model_path)

    stable_audio_root = checkpoints_dir / "stable-audio-3"
    stable_audio_ready = {}
    for key in STABLE_AUDIO3_VARIANTS:
        path = stable_audio_root / key
        stable_audio_ready[key] = (
            (path / "model_config.json").exists()
            and ((path / "model.safetensors").exists() or (path / "model.ckpt").exists())
        )
    print(f"Stable Audio 3 root: {stable_audio_root}")
    print(f"Stable Audio 3 readiness: {stable_audio_ready}")

    confucius_repo_path = persistent_app_path / CONFUCIUS_APP_SUBDIR
    confucius_config_path = confucius_repo_path / CONFUCIUS_FASTAPI_CONFIG
    confucius_vllm_dir = confucius_repo_path / "checkpoints" / "t2s-vllm"
    confucius_output_dir = confucius_repo_path / "outputs" / "api"
    confucius_compile_cache_dir = Path(PERSISTENT_CACHE_DIR) / "confucius" / "torchinductor"
    confucius_profile_dir = confucius_repo_path / "outputs" / "profiles"
    confucius_warmup_voice = confucius_repo_path / "resources" / "voice.mp3"

    required_confucius_paths = {
        "repo": confucius_repo_path,
        "fastapi_app": confucius_repo_path / "fastapi_app.py",
        "config": confucius_config_path,
        "vllm_model": confucius_vllm_dir / "model.safetensors",
        "vllm_config": confucius_vllm_dir / "config.json",
        "venv_python": Path(CONFUCIUS_PYTHON),
    }
    missing_confucius_paths = [
        f"{name}: {path}"
        for name, path in required_confucius_paths.items()
        if not path.exists()
    ]
    if missing_confucius_paths:
        raise FileNotFoundError(
            "Confucius4-TTS persistent setup is incomplete. Open the prepare_model web manager first. "
            + "; ".join(missing_confucius_paths)
        )

    indextts25_repo_path = persistent_app_path / INDEXTTS25_APP_SUBDIR
    indextts25_model_path = Path(INDEXTTS25_PERSISTENT_MODEL_DIR)
    required_indextts25_paths = {
        "repo": indextts25_repo_path,
        "deploy_config": indextts25_repo_path / "vllm_omni" / "deploy" / "indextts2_5.yaml",
        "omni_model_package": (
            indextts25_repo_path / "vllm_omni" / "model_executor" / "models" / "__init__.py"
        ),
        "venv_python": Path(INDEXTTS25_PYTHON),
        "vllm": Path(INDEXTTS25_VLLM),
        "config": indextts25_model_path / "config.yaml",
        "gpt": indextts25_model_path / "gpt.pth",
        "codec": indextts25_model_path / "codec.pth",
        "s2mel": indextts25_model_path / "s2mel.pth",
        "wav2vec": indextts25_model_path / "w2v-bert-2.0" / "model.safetensors",
        "wav2vec_preprocessor": (
            indextts25_model_path / "w2v-bert-2.0" / "preprocessor_config.json"
        ),
        "campplus": indextts25_model_path / "campplus_cn_common.bin",
        "bigvgan": indextts25_model_path / "bigvgan" / "bigvgan_generator.pt",
    }
    missing_indextts25_paths = [
        f"{name}: {path}"
        for name, path in required_indextts25_paths.items()
        if not path.exists()
    ]
    if missing_indextts25_paths:
        raise FileNotFoundError(
            "IndexTTS 2.5 vLLM-Omni persistent setup is incomplete. Open the prepare_model web manager first. "
            + "; ".join(missing_indextts25_paths)
        )

    indextts25_data_path = Path(INDEXTTS25_PERSISTENT_DATA_DIR)
    for relative_path in (
        "logs",
        "speakers",
        "cache/huggingface",
        "cache/speaker-conditioning",
        "cache/torchinductor",
        "cache/triton",
        "cache/cuda",
    ):
        (indextts25_data_path / relative_path).mkdir(parents=True, exist_ok=True)
    print(f"IndexTTS 2.5 vLLM-Omni repo: {indextts25_repo_path}")
    print(f"IndexTTS 2.5 model: {indextts25_model_path}")
    print(f"IndexTTS 2.5 isolated environment: {INDEXTTS25_VENV_DIR}")

    for path in (
        confucius_output_dir,
        confucius_compile_cache_dir,
        confucius_profile_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    confucius_env_vars = {
        "CONFUCIUS_TTS_CONFIG": str(confucius_config_path),
        "CONFUCIUS_T2S_VLLM_DIR": str(confucius_vllm_dir),
        "CONFUCIUS_API_OUTPUT_DIR": str(confucius_output_dir),
        "CONFUCIUS_COMPILE_CACHE_DIR": str(confucius_compile_cache_dir),
        "CONFUCIUS_PROFILE_DIR": str(confucius_profile_dir),
        "CONFUCIUS_WARMUP_PROMPT_WAV": str(confucius_warmup_voice),
        "CONFUCIUS_WARMUP": "1",
        "CONFUCIUS_WARMUP_MODE": "background",
        "CONFUCIUS_VLLM_ATTENTION_BACKEND": "FLASHINFER",
        "CONFUCIUS_VLLM_PREFIX_MODE": "auto",
        "CONFUCIUS_VLLM_LATENT_MODE": "auto",
        "CONFUCIUS_GPU_STAGE_CONCURRENCY": "1",
        "CONFUCIUS_POSTPROCESS_CONCURRENCY": "2",
        "CONFUCIUS_API_INFERENCE_WORKERS": "1",
    }
    print("Configuring Confucius4-TTS runtime environment:")
    for key, value in confucius_env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")

    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath_parts = [str(persistent_app_path)]
    if existing_pythonpath:
        confucius_pythonpath = str(confucius_repo_path).rstrip(os.sep)
        for part in existing_pythonpath.split(os.pathsep):
            part = part.strip()
            if not part or part in pythonpath_parts:
                continue
            if part.rstrip(os.sep) == confucius_pythonpath:
                continue
            pythonpath_parts.append(part)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    if not os.environ.get(INTERNAL_TOKEN_ENV):
        os.environ[INTERNAL_TOKEN_ENV] = uuid.uuid4().hex

    return _configure_gpu_runtime(persistent_app_path)


def _configure_gpu_runtime(persistent_app_path: Path) -> Path:
    """Resolve on the allocated GPU before any model is loaded or snapshotted."""
    import sys
    import tempfile

    sys.path.insert(0, RUNTIME_SOURCE_DIR)
    from indextts_web.gpu_profiles import PROFILE_ENV, runtime_gpu_profile
    from indextts_web.infrastructure.modal_runtime import prepare_runtime_code

    source = _updated_runtime_source(persistent_app_path)
    runtime_path = prepare_runtime_code(
        source, persistent_app_path,
        Path(tempfile.mkdtemp(prefix="indextts-runtime-")) / "app",
        managed_source=source != Path(RUNTIME_SOURCE_DIR),
    )
    profile = runtime_gpu_profile(modal=True)
    profile.check_startup_memory(non_vllm_gib=float(os.environ.get("INDEXTTS_NON_VLLM_RESERVE_GIB", "8")))
    os.environ[PROFILE_ENV] = profile.to_json()
    os.environ.setdefault("CONFUCIUS_REFERENCE_CACHE_SIZE", "2" if profile.name == "24gb" else "100")
    os.environ.setdefault("QWEN_ASR_MAX_BATCH_SIZE", {"24gb": "1", "48gb": "4", "96gb": "20"}[profile.name])
    os.environ["PYTHONPATH"] = str(runtime_path)
    cache_root = Path(PERSISTENT_CACHE_DIR) / "gpu-profiles" / profile.cache_key
    for variable, subdir in (("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
                             ("TRITON_CACHE_DIR", "triton"), ("CUDA_CACHE_PATH", "cuda"),
                             ("VLLM_CACHE_ROOT", "vllm")):
        directory = cache_root / subdir
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    os.environ["VLLM_CACHE"] = str(cache_root / "vllm")
    os.chdir(runtime_path)
    print(f"[GPU profile] {profile.to_json()}")
    print(f"Using deployed source: {runtime_path}; persistent data: {persistent_app_path}")
    return runtime_path


def _commit_snapshot_volumes(phase: str) -> None:
    """Persist paths referenced by the snapshot before another container restores it."""
    # Snapshot restore walks Volume paths before Python restore hooks can run.
    # Background/shutdown commits are not a persistence barrier during startup.
    # Do not reload: model workers may still hold open files on these mounts.
    for name, volume in (("audio-studio-cache", cache_storage), ("audio-studio-app", app_storage)):
        print(f"[Snapshot] Committing {name} ({phase})...", flush=True)
        try:
            volume.commit()
        except Exception as exc:
            raise RuntimeError(f"Snapshot Volume commit failed for {name} ({phase})") from exc
    print(f"[Snapshot] Persistent Volumes committed ({phase}).", flush=True)


@app.cls(
    image=image,
    gpu="RTX-PRO-6000",  # Manually choose "L4", "L40S", or "RTX-PRO-6000"; VRAM tuning is automatic.
    cpu=2.0,
    memory=8192,
    timeout=3600,
    scaledown_window=300,
    volumes={
        PERSISTENT_APP_DIR: app_storage,
        PERSISTENT_CACHE_DIR: cache_storage,
    },
    min_containers=0,
    max_containers=1,
    secrets=[modal.Secret.from_name("custom-secret")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=100)  # 100 concurrent requests
class IndexTTSVllmServer:
    """
    Serve IndexTTS v2 through FastAPI, with Modal CPU+GPU memory snapshots.
    """

    @modal.enter(snap=True)
    def start(self):
        persistent_app_path = _configure_persistent_runtime()
        cmd = _build_webui_command(persistent_app_path)
        _commit_snapshot_volumes("before model startup")

        self.moss_server_proc = _start_moss_transcribe_server(persistent_app_path)
        _wait_moss_ready(self.moss_server_proc)

        print(f"Starting FastAPI server: {' '.join(cmd)}")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        self.server_proc = subprocess.Popen(cmd, cwd=str(persistent_app_path), env=env)

        _wait_ready(self.server_proc, timeout_seconds=SNAPSHOT_STARTUP_TIMEOUT)

        print("Running snapshot warmup inference...")
        _call_local_json(
            "/internal/snapshot/warmup",
            method="POST",
            timeout=SNAPSHOT_REQUEST_TIMEOUT,
            internal=True,
        )

        print("Putting vLLM engines into sleep mode before snapshot...")
        _call_local_json(
            "/internal/snapshot/sleep?level=1",
            method="POST",
            timeout=SNAPSHOT_REQUEST_TIMEOUT,
            internal=True,
        )
        # Persist compiler caches and application data before snapshot capture.
        # Returning from snap=True permits capture, so commit synchronously here.
        _commit_snapshot_volumes("before snapshot capture")

    @modal.enter(snap=False)
    def wake_up(self):
        from indextts_web.gpu_profiles import runtime_gpu_profile
        from indextts_web.infrastructure.gpu import probe_gpu

        profile = runtime_gpu_profile(modal=True)
        actual = probe_gpu()
        if actual.capability != profile.gpu.capability or actual.total_bytes < profile.gpu.total_bytes:
            raise RuntimeError("Snapshot GPU does not match its startup profile; redeploy to rebuild the snapshot")
        _wait_moss_ready(self.moss_server_proc)
        print("Waking vLLM engines after memory snapshot restore...")
        _call_local_json(
            "/internal/snapshot/wake",
            method="POST",
            timeout=SNAPSHOT_REQUEST_TIMEOUT,
            internal=True,
        )
        _wait_ready(self.server_proc, timeout_seconds=SNAPSHOT_STARTUP_TIMEOUT)

    @modal.web_server(port=VLLM_PORT, startup_timeout=SNAPSHOT_STARTUP_TIMEOUT)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        for proc_name in ("server_proc", "moss_server_proc"):
            proc = getattr(self, proc_name, None)
            if proc is None or proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
