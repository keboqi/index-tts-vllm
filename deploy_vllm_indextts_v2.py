import modal
import os
import asyncio
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

# Use CUDA 13.0 for RTX Pro 6000 Blackwell.
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
CONFUCIUS_MODEL_REPO_ID = "netease-youdao/Confucius4-TTS"
CONFUCIUS_W2V_REPO_ID = "facebook/w2v-bert-2.0"
CONFUCIUS_BIGVGAN_REPO_ID = "nvidia/bigvgan_v2_22khz_80band_256x"
CONFUCIUS_CAMPPLUS_REPO_ID = "funasr/campplus"
CONFUCIUS_CAMPPLUS_FILENAME = "campplus_cn_common.bin"
CONFUCIUS_FASTAPI_CONFIG = "config/inference_config.modal.yaml"

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
        
        # vLLM sleep mode uses its CUDA memory pool; PyTorch expandable
        # segments are incompatible with that allocator.
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "TORCH_CUDNN_BENCHMARK": "1",  # Enable cuDNN autotuning
        "TORCH_COMPILE_MODE": "reduce-overhead",  # Optimize for TTS workloads

        # Cache directories for faster subsequent runs
        "HF_HOME": "/persistent_cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/persistent_cache/huggingface/hub",
        "TORCH_HOME": "/persistent_cache/torch",
        "TRANSFORMERS_CACHE": "/persistent_cache/transformers",
        "CUDA_CACHE_PATH": "/persistent_cache/cuda_cache",
        "VLLM_CACHE": "/persistent_cache/vllm_cache",
        "TRITON_CACHE_DIR": "/persistent_cache/triton",
        "VLLM_SERVER_DEV_MODE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCH_NCCL_ENABLE_MONITORING": "0",
        "TORCH_CPP_LOG_LEVEL": "ERROR"
    })
    .run_commands("pip install --upgrade pip setuptools wheel")
    .pip_install(
        "torch", 
        "torchaudio",
        extra_options="--index-url https://download.pytorch.org/whl/cu130"
    )
    .pip_install(
        "litai",
        "whisperx",
        "nemo_toolkit[asr]",
        "json-repair"
    )
    .run_commands(
        f"git clone {INDEXTTS_REPO_URL} /app/index-tts-vllm"
    )
    .run_commands(
        "cd /app/index-tts-vllm && pip install -r requirements.txt"
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
        # The PyPI stable-audio-tools wheel is too old for Stable Audio 3
        # configs and also pins older torch builds. Use current source and
        # keep dependencies explicit so CUDA 13 torch stays installed.
        "pip install --force-reinstall --no-deps --ignore-requires-python "
        "git+https://github.com/Stability-AI/stable-audio-tools.git",
        "pip install alias-free-torch dill einops-exts huggingface_hub "
        "importlib-resources nnAudio PyWavelets safetensors scipy soxr "
        "torchsde tqdm transformers v-diffusion-pytorch "
        "vector-quantize-pytorch",
    )
    .pip_install("alias_free_torch")
    .pip_install(
        "pydub",
        "flashinfer-python"
    )
    .run_commands("pip install flash-attn --no-build-isolation")
    .run_commands("pip install audio-separator")
    .run_commands("pip install clearvoice google-genai")
    .run_commands("pip install qwen-asr omnivad")
    .pip_install(
        "yt-dlp[default]",
        "yt-dlp-ejs",
        "bgutil-ytdlp-pot-provider",
    )
    .pip_install("numpy<2")
    .run_commands(
        "npm install -g n",
        "n 22",
        "node --version",
    )
    .pip_install("pedalboard")
)

app = modal.App("vllm-indextts-v2", image=image)

# Create persistent storage volumes
app_storage = modal.Volume.from_name("indextts-v2-app", create_if_missing=True)
cache_storage = modal.Volume.from_name("indextts-v2-cache", create_if_missing=True)

# Configuration
PERSISTENT_APP_DIR = "/persistent_app"
PERSISTENT_CACHE_DIR = "/persistent_cache"
CONFUCIUS_PERSISTENT_REPO_DIR = f"{PERSISTENT_APP_DIR}/{CONFUCIUS_APP_SUBDIR}"
VLLM_PORT = 8000
CONFUCIUS_PORT = 8001
SNAPSHOT_STARTUP_TIMEOUT = 1800
SNAPSHOT_REQUEST_TIMEOUT = 900
INTERNAL_TOKEN_ENV = "INDEXTTS_INTERNAL_TOKEN"
DEFAULT_TTS_BACKEND = "index"
GPU_MEMORY_UTILIZATION = 0.15
QWENEMO_GPU_MEMORY_UTILIZATION = 0.05
CONFUCIUS_GPU_MEMORY_UTILIZATION = 0.20
CONFUCIUS_STARTUP_TIMEOUT = 1200
CONFUCIUS_REQUEST_TIMEOUT = 900

STABLE_AUDIO3_VARIANTS = ("medium", "small-music", "small-sfx")


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
    image=image,
    timeout=3600,
    volumes={
        PERSISTENT_APP_DIR: app_storage,
        PERSISTENT_CACHE_DIR: cache_storage
    },
    cpu=4.0,
    memory=32768
)
def prepare_model():
    """
    CPU function to:
    1. Copy IndexTTS and Confucius4-TTS code into persistent storage
    2. Update both applications from their integration remotes
    3. Download IndexTTS, Qwen3 Voice Design, Stable Audio 3, and
       Confucius4-TTS assets into persistent storage/cache
    4. Convert the Confucius T2S checkpoint into a vLLM-loadable directory
    
    This is a one-time setup that creates a fully self-contained persistent app.
    """
    import subprocess
    import shutil
    from pathlib import Path
    
    print("🚀 Preparing IndexTTS v2 application and model...")
    
    # Step 1: Copy the entire application to persistent storage
    persistent_app_path = Path(PERSISTENT_APP_DIR)
    source_app_path = Path("/app/index-tts-vllm")
    confucius_source_path = Path(CONFUCIUS_IMAGE_DIR)
    confucius_persistent_path = persistent_app_path / CONFUCIUS_APP_SUBDIR
    
    if not persistent_app_path.exists() or len(list(persistent_app_path.iterdir())) == 0:
        print("📂 Copying application to persistent storage...")
        persistent_app_path.mkdir(exist_ok=True)
        
        # Copy all files from source to persistent storage
        for item in source_app_path.iterdir():
            dest_item = persistent_app_path / item.name
            if item.is_dir():
                if dest_item.exists():
                    shutil.rmtree(dest_item)
                shutil.copytree(item, dest_item)
                print(f"   📁 Copied directory: {item.name}")
            else:
                shutil.copy2(item, dest_item)
                print(f"   📄 Copied file: {item.name}")
        
        print("✅ Application copied to persistent storage successfully!")
    else:
        print("✅ Application already exists in persistent storage")
    
    if not confucius_persistent_path.exists() or len(list(confucius_persistent_path.iterdir())) == 0:
        print(f"Copying Confucius4-TTS app to persistent storage: {confucius_persistent_path}")
        confucius_persistent_path.mkdir(parents=True, exist_ok=True)
        for item in confucius_source_path.iterdir():
            dest_item = confucius_persistent_path / item.name
            if item.is_dir():
                if dest_item.exists():
                    shutil.rmtree(dest_item)
                shutil.copytree(item, dest_item)
                print(f"   Copied Confucius directory: {item.name}")
            else:
                shutil.copy2(item, dest_item)
                print(f"   Copied Confucius file: {item.name}")
    else:
        print(f"Confucius4-TTS already exists in persistent storage: {confucius_persistent_path}")

    # Step 2: Update the application with latest code from git (force override local changes)
    print("\n📥 Step 2: Updating application from git repository (force override)...")
    repo_update_status = {
        "success": False,
        "message": "",
        "output": ""
    }
    try:
        # Change to persistent app directory
        os.chdir(str(persistent_app_path))
        subprocess.run(
            ["git", "remote", "set-url", "origin", INDEXTTS_REPO_URL],
            capture_output=True,
            text=True,
            cwd=str(persistent_app_path),
        )
        print(f"   📁 Changed to directory: {persistent_app_path}")
        
        # Step 2a: Fetch latest from all remotes
        print("   📥 Fetching latest from all remotes...")
        fetch_result = subprocess.run(
            ["git", "fetch", "--all"],
            capture_output=True,
            text=True,
            cwd=str(persistent_app_path)
        )
        
        if fetch_result.returncode != 0:
            print(f"⚠️ Git fetch failed: {fetch_result.stderr.strip()}")
            repo_update_status["message"] = f"Git fetch failed: {fetch_result.stderr.strip()}"
            repo_update_status["output"] = fetch_result.stderr.strip()
        else:
            print("   ✅ Fetch completed")
            
            # Step 2b: Get the default branch name
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(persistent_app_path)
            )
            current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "main"
            print(f"   📌 Current branch: {current_branch}")
            
            # Step 2c: Hard reset to origin/branch (force override local changes)
            print(f"   🔄 Resetting to origin/{current_branch} (force override local changes)...")
            reset_result = subprocess.run(
                ["git", "reset", "--hard", f"origin/{current_branch}"],
                capture_output=True,
                text=True,
                cwd=str(persistent_app_path)
            )
            
            if reset_result.returncode == 0:
                print(f"✅ Git reset successful!")
                repo_update_status["success"] = True
                repo_update_status["message"] = f"Repository updated to origin/{current_branch}"
                repo_update_status["output"] = reset_result.stdout.strip()
                if reset_result.stdout.strip():
                    print(f"   📋 Output: {reset_result.stdout.strip()}")
            else:
                print(f"⚠️ Git reset failed with exit code: {reset_result.returncode}")
                repo_update_status["message"] = f"Git reset failed with exit code {reset_result.returncode}"
                repo_update_status["output"] = reset_result.stderr.strip() or reset_result.stdout.strip()
                if reset_result.stderr.strip():
                    print(f"   ⚠️ Stderr: {reset_result.stderr.strip()}")
    except Exception as e:
        print(f"⚠️ Git update failed (non-fatal): {str(e)}")
        print("   Continuing with existing code...")
        repo_update_status["message"] = f"Git update exception: {str(e)}"
    
    confucius_repo_update_status = {
        "success": False,
        "message": "",
        "output": "",
        "repo": str(confucius_persistent_path),
        "remote": CONFUCIUS_REPO_URL,
    }
    try:
        if not (confucius_persistent_path / ".git").exists():
            confucius_repo_update_status["message"] = (
                f"Skipped git update because {confucius_persistent_path} has no .git directory"
            )
        else:
            subprocess.run(
                ["git", "remote", "set-url", "origin", CONFUCIUS_REPO_URL],
                capture_output=True,
                text=True,
                cwd=str(confucius_persistent_path),
            )
            fetch_result = subprocess.run(
                ["git", "fetch", "origin"],
                capture_output=True,
                text=True,
                cwd=str(confucius_persistent_path),
            )
            if fetch_result.returncode != 0:
                confucius_repo_update_status["message"] = (
                    f"Confucius git fetch failed: {fetch_result.stderr.strip()}"
                )
                confucius_repo_update_status["output"] = fetch_result.stderr.strip()
            else:
                branch_result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=str(confucius_persistent_path),
                )
                current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "main"
                if not current_branch or current_branch == "HEAD":
                    current_branch = "main"
                origin_ref = f"origin/{current_branch}"
                reset_result = subprocess.run(
                    ["git", "reset", "--hard", origin_ref],
                    capture_output=True,
                    text=True,
                    cwd=str(confucius_persistent_path),
                )
                if reset_result.returncode == 0:
                    confucius_repo_update_status["success"] = True
                    confucius_repo_update_status["message"] = f"Confucius repository updated to {origin_ref}"
                    confucius_repo_update_status["output"] = reset_result.stdout.strip()
                else:
                    confucius_repo_update_status["message"] = (
                        f"Confucius git reset failed: {reset_result.stderr.strip()}"
                    )
                    confucius_repo_update_status["output"] = (
                        reset_result.stderr.strip() or reset_result.stdout.strip()
                    )
    except Exception as exc:
        confucius_repo_update_status["message"] = f"Confucius git update exception: {exc}"
    print(confucius_repo_update_status["message"])

    try:
        confucius_vllm_patch_status = _ensure_confucius_vllm_patch_compatibility(
            confucius_persistent_path
        )
    except Exception as exc:
        confucius_vllm_patch_status = {
            "success": False,
            "changed": False,
            "path": str(
                confucius_persistent_path
                / "confuciustts"
                / "llm"
                / "vllm_patch.py"
            ),
            "message": f"Confucius vLLM patch compatibility update failed: {exc}",
        }
    print(confucius_vllm_patch_status["message"])

    # Step 3: Download model repo directly into persistent checkpoints folder
    checkpoints_dir = persistent_app_path / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    print(f"📦 Downloading IndexTTS v2 model to: {checkpoints_dir}")
    
    try:
        # Use huggingface-hub to download the model directly into checkpoints
        result = subprocess.run([
            "python", "-c", 
            f"""
from huggingface_hub import snapshot_download
import os

print("Downloading main IndexTTS v2 / Stable Audio 3 model repo...")
snapshot_download(
    repo_id="garyswansrs/index_tts_2_vllm",
    local_dir="{checkpoints_dir}",
    local_dir_use_symlinks=False
)
print("Model download completed!")
"""
        ], check=True, capture_output=True, text=True, cwd=str(persistent_app_path))
        
        print("✅ Main model repo download completed successfully!")
        
        # Step 3b: Voice Design and Stable Audio 3 models are included in checkpoints
        voice_design_dir = checkpoints_dir / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        stable_audio_root = checkpoints_dir / "stable-audio-3"
        stable_audio_dirs = {
            key: stable_audio_root / key
            for key in STABLE_AUDIO3_VARIANTS
        }
        
        # Step 4: List downloaded files for verification
        print("🔍 Listing downloaded model files...")
        for file_path in checkpoints_dir.rglob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
                relative_path = file_path.relative_to(checkpoints_dir)
                print(f"   📄 {relative_path} ({file_size:.1f} MB)")
        
        # Check for vLLM directory (should exist in pre-converted model)
        vllm_dir = checkpoints_dir / "gpt"
        if vllm_dir.exists():
            print(f"   ✅ vLLM model directory: {vllm_dir}")
            vllm_files = list(vllm_dir.iterdir())
            print(f"   📁 vLLM files: {[f.name for f in vllm_files]}")
        else:
            print(f"   ⚠️ vLLM directory not found: {vllm_dir}")
        
        print("   Stable Audio 3 checkpoint directories:")
        for key, path in stable_audio_dirs.items():
            config_path = path / "model_config.json"
            ckpt_ready = (path / "model.safetensors").exists() or (path / "model.ckpt").exists()
            status = "ready" if config_path.exists() and ckpt_ready else "missing"
            print(f"      {key}: {path} ({status})")

        confucius_checkpoints_dir = confucius_persistent_path / "checkpoints"
        confucius_pretrained_dir = confucius_persistent_path / "pretrained"
        confucius_outputs_dir = confucius_persistent_path / "outputs" / "api"
        confucius_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        confucius_pretrained_dir.mkdir(parents=True, exist_ok=True)
        confucius_outputs_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading Confucius4-TTS assets to: {confucius_persistent_path}")
        confucius_download_code = f"""
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

checkpoints = Path({str(confucius_checkpoints_dir)!r})
pretrained = Path({str(confucius_pretrained_dir)!r})
checkpoints.mkdir(parents=True, exist_ok=True)
pretrained.mkdir(parents=True, exist_ok=True)

model_files = [
    "t2s_model.safetensors",
    "s2a_model.pt",
    "wav2vec2bert_stats.pt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]
for filename in model_files:
    print("Downloading Confucius model file: " + filename)
    hf_hub_download(
        repo_id={CONFUCIUS_MODEL_REPO_ID!r},
        filename=filename,
        local_dir=str(checkpoints),
        local_dir_use_symlinks=False,
    )

print("Downloading Wav2Vec2-BERT speaker/semantic model...")
snapshot_download(
    repo_id={CONFUCIUS_W2V_REPO_ID!r},
    local_dir=str(pretrained / "w2v-bert-2.0"),
    local_dir_use_symlinks=False,
)

print("Downloading BigVGAN vocoder...")
snapshot_download(
    repo_id={CONFUCIUS_BIGVGAN_REPO_ID!r},
    local_dir=str(pretrained / "bigvgan_v2_22khz_80band_256x"),
    local_dir_use_symlinks=False,
)

print("Downloading CAMPPlus speaker encoder...")
hf_hub_download(
    repo_id={CONFUCIUS_CAMPPLUS_REPO_ID!r},
    filename={CONFUCIUS_CAMPPLUS_FILENAME!r},
    local_dir=str(pretrained / "campplus"),
    local_dir_use_symlinks=False,
)
print("Confucius asset download completed.")
"""
        confucius_download_result = subprocess.run(
            ["python", "-c", confucius_download_code],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(confucius_persistent_path),
        )
        if confucius_download_result.stdout.strip():
            print(confucius_download_result.stdout.strip())
        if confucius_download_result.stderr.strip():
            print(confucius_download_result.stderr.strip())

        confucius_base_config = confucius_persistent_path / "config" / "inference_config.yaml"
        confucius_modal_config = confucius_persistent_path / CONFUCIUS_FASTAPI_CONFIG
        config_text = confucius_base_config.read_text(encoding="utf-8")
        config_text = config_text.replace(
            "  w2v_bert_path: facebook/w2v-bert-2.0",
            "  w2v_bert_path: ./pretrained/w2v-bert-2.0",
        )
        config_text = config_text.replace(
            "  vocoder_path: nvidia/bigvgan_v2_22khz_80band_256x",
            "  vocoder_path: ./pretrained/bigvgan_v2_22khz_80band_256x",
        )
        confucius_modal_config.write_text(config_text, encoding="utf-8")
        print(f"Wrote Confucius Modal config: {confucius_modal_config}")

        confucius_vllm_dir = confucius_checkpoints_dir / "t2s-vllm"
        confucius_vllm_ready = (
            (confucius_vllm_dir / "model.safetensors").exists()
            and (confucius_vllm_dir / "config.json").exists()
        )
        if confucius_vllm_ready:
            print(f"Confucius vLLM directory already ready: {confucius_vllm_dir}")
        else:
            confucius_python = Path(CONFUCIUS_PYTHON)
            if not confucius_python.exists():
                raise FileNotFoundError(f"Confucius image venv Python not found: {confucius_python}")

            convert_cmd = [
                str(confucius_python),
                "tools/convert_t2s_vllm.py",
                "--config",
                str(confucius_modal_config),
                "--output",
                str(confucius_vllm_dir),
            ]
            local_t2s_checkpoint = confucius_checkpoints_dir / "t2s_model.safetensors"
            if local_t2s_checkpoint.exists():
                convert_cmd.extend(["--checkpoint", str(local_t2s_checkpoint)])

            convert_env = dict(os.environ)
            convert_env["PYTHONPATH"] = str(confucius_persistent_path)
            convert_env.setdefault("HF_HOME", f"{PERSISTENT_CACHE_DIR}/huggingface")
            convert_env.setdefault("TORCH_HOME", f"{PERSISTENT_CACHE_DIR}/torch")
            print(f"Converting Confucius T2S model for vLLM: {' '.join(convert_cmd)}")
            convert_result = subprocess.run(
                convert_cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(confucius_persistent_path),
                env=convert_env,
            )
            if convert_result.stdout.strip():
                print(convert_result.stdout.strip())
            if convert_result.stderr.strip():
                print(convert_result.stderr.strip())

        confucius_vllm_ready = (
            (confucius_vllm_dir / "model.safetensors").exists()
            and (confucius_vllm_dir / "config.json").exists()
        )
        confucius_ready = (
            (confucius_persistent_path / "fastapi_app.py").exists()
            and confucius_modal_config.exists()
            and confucius_vllm_ready
            and (confucius_persistent_path / "resources" / "voice.mp3").exists()
        )
        print(f"Confucius4-TTS readiness: {confucius_ready}")
        print(f"  repo: {confucius_persistent_path}")
        print(f"  config: {confucius_modal_config}")
        print(f"  checkpoints: {confucius_checkpoints_dir}")
        print(f"  pretrained: {confucius_pretrained_dir}")
        print(f"  vLLM: {confucius_vllm_dir} ({'ready' if confucius_vllm_ready else 'missing'})")

        # Step 5: List complete application structure for verification
        print("\n📋 Persistent application structure:")
        def show_tree(path, prefix="", max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return
            items = sorted(list(path.iterdir()))
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item.name}")
                if item.is_dir() and current_depth < max_depth - 1:
                    extension = "    " if is_last else "│   "
                    show_tree(item, prefix + extension, max_depth, current_depth + 1)
        
        show_tree(persistent_app_path)
        
        print(f"\n✅ IndexTTS v2 application and models preparation completed!")
        print(f"📁 Persistent app location: {persistent_app_path}")
        print(f"📁 IndexTTS model location: {checkpoints_dir}")
        print(f"📁 Voice Design model location: {voice_design_dir}")
        print("🚀 Ready for inference deployment!")
        
        return {
            "status": "success",
            "message": "IndexTTS v2 application, Stable Audio 3, and models prepared successfully",
            "app_dir": str(persistent_app_path),
            "model_dir": str(checkpoints_dir),
            "voice_design_dir": str(voice_design_dir),
            "stable_audio3_root": str(stable_audio_root),
            "stable_audio3_dirs": {key: str(path) for key, path in stable_audio_dirs.items()},
            "vllm_ready": vllm_dir.exists(),
            "voice_design_ready": voice_design_dir.exists(),
            "stable_audio3_ready": {
                key: (path / "model_config.json").exists()
                and ((path / "model.safetensors").exists() or (path / "model.ckpt").exists())
                for key, path in stable_audio_dirs.items()
            },
            "repo_update": repo_update_status,
            "confucius_repo_update": confucius_repo_update_status,
            "confucius_vllm_patch": confucius_vllm_patch_status,
            "confucius": {
                "repo_dir": str(confucius_persistent_path),
                "config": str(confucius_modal_config),
                "checkpoints_dir": str(confucius_checkpoints_dir),
                "pretrained_dir": str(confucius_pretrained_dir),
                "outputs_dir": str(confucius_outputs_dir),
                "vllm_dir": str(confucius_vllm_dir),
                "ready": confucius_ready,
                "vllm_ready": confucius_vllm_ready,
            },
        }
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to download model: {e.stderr}"
        print(f"❌ {error_msg}")
        return {
            "status": "error", 
            "message": error_msg,
            "stdout": e.stdout,
            "stderr": e.stderr,
            "repo_update": repo_update_status,
            "confucius_repo_update": confucius_repo_update_status,
            "confucius_vllm_patch": locals().get("confucius_vllm_patch_status"),
        }
    except Exception as e:
        error_msg = f"Model preparation failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "status": "error",
            "message": error_msg,
            "repo_update": repo_update_status,
            "confucius_repo_update": confucius_repo_update_status,
            "confucius_vllm_patch": locals().get("confucius_vllm_patch_status"),
        }



@app.function(
    image=image,
    timeout=180,
    volumes={
        PERSISTENT_APP_DIR: app_storage,
        PERSISTENT_CACHE_DIR: cache_storage
    }
)
def clear_cache():
    """
    Function to clear persistent caches while preserving the app and models.
    
    Usage:
        modal run deploy_vllm_indextts_v2.py::clear_cache
    """
    import os
    import shutil
    import glob
    from pathlib import Path
    
    print("🧹 Starting cache clearing process for IndexTTS v2...")
    print("⚠️  This will clear caches but preserve the app and models.")
    
    # Clear general persistent caches
    cache_dirs_to_clear = [
        "/persistent_cache/huggingface",
        "/persistent_cache/torch", 
        "/persistent_cache/transformers",
        "/persistent_cache/cuda_cache",
        "/persistent_cache/vllm_cache",
        "/persistent_cache/torch_compile_cache",  # torch.compile artifacts
        "/persistent_cache/confucius",
        "/persistent_cache/triton",
    ]
    
    # Clear app-specific caches (but keep the app and models)
    persistent_app_path = Path(PERSISTENT_APP_DIR)
    app_cache_dirs_to_clear = []
    if persistent_app_path.exists():
        app_cache_dirs_to_clear = [
            persistent_app_path / "speaker_presets",
            persistent_app_path / "emotion_cache",
            persistent_app_path / "emb_cache",
            persistent_app_path / "outputs",
            persistent_app_path / CONFUCIUS_APP_SUBDIR / "outputs",
        ]
    
    # Calculate total cache size before clearing
    total_size_before = 0
    all_dirs = cache_dirs_to_clear + [str(d) for d in app_cache_dirs_to_clear]
    
    for cache_dir in all_dirs:
        if os.path.exists(cache_dir):
            try:
                for dirpath, dirnames, filenames in os.walk(cache_dir):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size_before += os.path.getsize(filepath)
            except Exception as e:
                print(f"⚠️ Could not calculate size for {cache_dir}: {e}")
    
    print(f"💾 Total cache size before clearing: {total_size_before / (1024 * 1024):.2f} MB")
    
    # Clear cache directories
    cleared_dirs = []
    failed_dirs = []
    
    # Clear general caches
    for cache_dir in cache_dirs_to_clear:
        try:
            if os.path.exists(cache_dir):
                print(f"🗑️ Clearing cache directory: {cache_dir}")
                shutil.rmtree(cache_dir)
                # Recreate empty directory
                os.makedirs(cache_dir, exist_ok=True)
                cleared_dirs.append(cache_dir)
                print(f"✅ Cleared: {cache_dir}")
            else:
                print(f"⏭️ Skipped (doesn't exist): {cache_dir}")
        except Exception as e:
            failed_dirs.append({"dir": cache_dir, "error": str(e)})
            print(f"❌ Failed to clear {cache_dir}: {e}")
    
    # Clear app-specific caches
    for cache_dir in app_cache_dirs_to_clear:
        try:
            if cache_dir.exists():
                print(f"🗑️ Clearing app cache directory: {cache_dir}")
                shutil.rmtree(str(cache_dir))
                # Recreate empty directory
                cache_dir.mkdir(exist_ok=True)
                cleared_dirs.append(str(cache_dir))
                print(f"✅ Cleared: {cache_dir}")
            else:
                print(f"⏭️ Skipped (doesn't exist): {cache_dir}")
        except Exception as e:
            failed_dirs.append({"dir": str(cache_dir), "error": str(e)})
            print(f"❌ Failed to clear {cache_dir}: {e}")
    
    # Also clear any CUDA/vLLM compilation caches
    additional_cache_patterns = [
        "/tmp/nvcc_*",  # CUDA compilation temps
        "/tmp/tmpxft_*",  # More CUDA temps
        "/tmp/cuda_*",  # CUDA runtime temps
        "/tmp/vllm_*",  # vLLM temps
    ]
    
    for pattern in additional_cache_patterns:
        try:
            for path in glob.glob(pattern):
                if os.path.exists(path):
                    if os.path.isfile(path):
                        os.unlink(path)
                    else:
                        shutil.rmtree(path)
                    cleared_dirs.append(path)
                    print(f"✅ Cleared: {path}")
        except Exception as e:
            failed_dirs.append({"pattern": pattern, "error": str(e)})
            print(f"❌ Failed to clear pattern {pattern}: {e}")
    
    # Summary
    print("\n🧹 Cache clearing completed!")
    print(f"✅ Successfully cleared: {len(cleared_dirs)} directories/files")
    print(f"❌ Failed to clear: {len(failed_dirs)} directories/files")
    print(f"💾 Total space freed: {total_size_before / (1024 * 1024):.2f} MB")
    
    if failed_dirs:
        print("\n⚠️ Failed operations:")
        for failed in failed_dirs:
            print(f"  - {failed}")
    
    print("\n📋 What was preserved:")
    print("  ✅ Application code and files")
    print("  ✅ Model weights and checkpoints")
    print("  ✅ Application directory structure")
    
    print("\n📋 What was cleared:")
    print("  🗑️ Speaker preset caches")
    print("  🗑️ Emotion analysis caches")
    print("  🗑️ Embedding caches")
    print("  🗑️ PyTorch/HuggingFace caches")
    print("  🗑️ CUDA compilation caches")
    print("  🗑️ Output files")
    
    print("\n📋 Next steps:")
    print("  1. Caches will rebuild automatically on next use")
    print("  2. No need to re-download models or re-copy application")
    print("  3. Redeploy with: modal deploy deploy_vllm_indextts_v2.py")
    
    return {
        "status": "completed",
        "cleared_count": len(cleared_dirs),
        "failed_count": len(failed_dirs),
        "space_freed_mb": total_size_before / (1024 * 1024)
    }

def legacy_serve_without_snapshot():
    """
    Serve the IndexTTS v2 FastAPI application by running python fastapi_webui_v2.py directly.
    """
    import os
    import sys
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
        print("❌ Persistent application not found! Run prepare_model first.")
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
    
    # Build the command
    cmd = [
        "python",
        "-u",
        "fastapi_webui_v2.py",
        "--gpu_memory_utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--qwenemo_gpu_memory_utilization",
        str(QWENEMO_GPU_MEMORY_UTILIZATION),
        "--tts_backend",
        DEFAULT_TTS_BACKEND,
        "--confucius_repo_dir",
        str(persistent_app_path / CONFUCIUS_APP_SUBDIR),
        "--confucius_host",
        "127.0.0.1",
        "--confucius_port",
        str(CONFUCIUS_PORT),
        "--confucius_start_command",
        _build_confucius_start_command(persistent_app_path / CONFUCIUS_APP_SUBDIR),
        "--confucius_start_timeout",
        str(CONFUCIUS_STARTUP_TIMEOUT),
        "--confucius_request_timeout",
        str(CONFUCIUS_REQUEST_TIMEOUT),
        "--confucius_vllm_gpu_memory_utilization",
        str(CONFUCIUS_GPU_MEMORY_UTILIZATION),
    ]
    
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
            _call_local_json("/server_info", timeout=10)
            print("FastAPI server is ready.")
            return
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(2)

    raise TimeoutError(
        f"Timed out waiting {timeout_seconds}s for FastAPI server readiness. "
        f"Last error: {last_error}"
    )


def _build_confucius_start_command(confucius_repo_path: Path) -> str:
    confucius_config_path = confucius_repo_path / CONFUCIUS_FASTAPI_CONFIG
    confucius_vllm_dir = confucius_repo_path / "checkpoints" / "t2s-vllm"
    confucius_output_dir = confucius_repo_path / "outputs" / "api"
    confucius_compile_cache_dir = (
        Path(PERSISTENT_CACHE_DIR) / "confucius" / "torchinductor"
    )
    confucius_warmup_voice = confucius_repo_path / "resources" / "voice.mp3"

    parts = [
        "env",
        f"PYTHONPATH={confucius_repo_path}{os.pathsep}{PERSISTENT_APP_DIR}",
        CONFUCIUS_PYTHON,
        "-u",
        "fastapi_app.py",
        "--host",
        "127.0.0.1",
        "--port",
        str(CONFUCIUS_PORT),
        "--config",
        str(confucius_config_path),
        "--vllm-model-dir",
        str(confucius_vllm_dir),
        "--vllm-gpu-memory-utilization",
        str(CONFUCIUS_GPU_MEMORY_UTILIZATION),
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
        "background",
        "--warmup-prompt-wav",
        str(confucius_warmup_voice),
        "--compile-s2a",
        "--gpu-stage-concurrency",
        "1",
        "--postprocess-concurrency",
        "2",
        "--inference-workers",
        "1",
    ]
    return " ".join(shlex.quote(str(part)) for part in parts)


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
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "INDEXTTS_ENABLE_VLLM_SLEEP_MODE": "1",
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
            f"Application not found at {persistent_app_path}. Run prepare_model first."
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
            "Confucius4-TTS persistent setup is incomplete. Run prepare_model first. "
            + "; ".join(missing_confucius_paths)
        )

    for path in (
        confucius_output_dir,
        confucius_compile_cache_dir,
        confucius_profile_dir,
        Path(PERSISTENT_CACHE_DIR) / "triton",
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
        "CONFUCIUS_VLLM_GPU_MEMORY_UTILIZATION": str(CONFUCIUS_GPU_MEMORY_UTILIZATION),
        "CONFUCIUS_VLLM_ATTENTION_BACKEND": "FLASHINFER",
        "CONFUCIUS_VLLM_PREFIX_MODE": "auto",
        "CONFUCIUS_VLLM_LATENT_MODE": "auto",
        "CONFUCIUS_USE_TORCH_COMPILE": "1",
        "CONFUCIUS_GPU_STAGE_CONCURRENCY": "1",
        "CONFUCIUS_POSTPROCESS_CONCURRENCY": "2",
        "CONFUCIUS_API_INFERENCE_WORKERS": "1",
        "CONFUCIUS_REFERENCE_CACHE_SIZE": "100",
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

    print(f"Working directory: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")
    return persistent_app_path


@app.cls(
    image=image,
    gpu="RTX-PRO-6000",  # 96GB Blackwell; use "L40S" if you want Ada/L40S instead.
    cpu=4.0,
    memory=8124,
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

        cmd = [
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
            str(GPU_MEMORY_UTILIZATION),
            "--qwenemo_gpu_memory_utilization",
            str(QWENEMO_GPU_MEMORY_UTILIZATION),
            "--tts_backend",
            DEFAULT_TTS_BACKEND,
            "--confucius_repo_dir",
            str(persistent_app_path / CONFUCIUS_APP_SUBDIR),
            "--confucius_host",
            "127.0.0.1",
            "--confucius_port",
            str(CONFUCIUS_PORT),
            "--confucius_start_command",
            _build_confucius_start_command(persistent_app_path / CONFUCIUS_APP_SUBDIR),
            "--confucius_start_timeout",
            str(CONFUCIUS_STARTUP_TIMEOUT),
            "--confucius_request_timeout",
            str(CONFUCIUS_REQUEST_TIMEOUT),
            "--confucius_vllm_gpu_memory_utilization",
            str(CONFUCIUS_GPU_MEMORY_UTILIZATION),
            "--use_torch_compile",
        ]
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

    @modal.enter(snap=False)
    def wake_up(self):
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
        proc = getattr(self, "server_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
