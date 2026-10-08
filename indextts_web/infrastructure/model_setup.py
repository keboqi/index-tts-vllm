"""CPU-only provisioning for the Modal manager; no model imports at startup."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORIES = {
    "index": ("IndexTTS WebUI", "https://github.com/keboqi/index-tts-vllm.git", "repositories/index"),
    "confucius": ("Confucius4-TTS", "https://github.com/keboqi/Confucius4-TTS.git", "Confucius4-TTS"),
    "index25": ("IndexTTS 2.5 Omni", "https://github.com/keboqi/indextts-2.5-vllm-omni-experiment.git",
                "index-tts-2.5-vllm-omni-experiment"),
}
COMPONENTS = {
    "main": ("Main TTS / WebUI", "/usr/local/bin/python"),
    "confucius": ("Confucius4-TTS", "/opt/confucius4tts-venv/bin/python"),
    "index25": ("IndexTTS 2.5 Omni", "/opt/indextts25-venv/bin/python"),
    "moss": ("MOSS transcription", "/opt/moss-transcribe-venv/bin/python"),
    "qwen-asr": ("Qwen3-ASR", "/opt/qwen-asr-venv/bin/python"),
    "clearvoice": ("ClearVoice", "/opt/clearvoice-venv/bin/python"),
}
REPOSITORY_ENTRYPOINTS = {
    "index": "fastapi_webui_v2.py",
    "confucius": "fastapi_app.py",
    "index25": "vllm_omni/deploy/indextts2_5.yaml",
}
IMAGE_REPOSITORIES = {
    "confucius": Path("/app/Confucius4-TTS"),
    "index25": Path("/app/index-tts-2.5-vllm-omni-experiment"),
}

# A bundle can contain multiple Hub repositories. Keep paths aligned with the
# inference launchers, including Confucius's separate checkpoints/pretrained.
MODELS = {
    "index": {
        "name": "IndexTTS 2.0", "downloads": [("garyswansrs/index_tts_2_vllm", "checkpoints", None)],
        "required": ["checkpoints/config.yaml", "checkpoints/gpt/config.json",
                     "checkpoints/gpt/model*.safetensors|checkpoints/gpt/pytorch_model*.bin",
                     "checkpoints/s2mel.pth"],
    },
    "index25": {
        "name": "IndexTTS 2.5",
        "downloads": [
            ("IndexTeam/IndexTTS-2.5", "checkpoints/IndexTTS-2.5", None),
            ("facebook/w2v-bert-2.0", "checkpoints/IndexTTS-2.5/w2v-bert-2.0",
             ["config.json", "model.safetensors", "preprocessor_config.json"]),
            ("funasr/campplus", "checkpoints/IndexTTS-2.5", ["campplus_cn_common.bin"]),
            ("nvidia/bigvgan_v2_22khz_80band_256x", "checkpoints/IndexTTS-2.5/bigvgan",
             ["config.json", "bigvgan_generator.pt"]),
        ],
        "required": [f"checkpoints/IndexTTS-2.5/{name}" for name in (
            "config.yaml", "gpt.pth", "codec.pth", "s2mel.pth", "wav2vec2bert_stats.pt",
            "multilingual_zh_ja_yue_char_del.tiktoken", "qwen0.6bemo4-merge/config.json",
            "qwen0.6bemo4-merge/model.safetensors", "w2v-bert-2.0/config.json",
            "w2v-bert-2.0/model.safetensors", "w2v-bert-2.0/preprocessor_config.json",
            "campplus_cn_common.bin", "bigvgan/config.json", "bigvgan/bigvgan_generator.pt")],
    },
    "voice-design": {
        "name": "Qwen3 Voice Design",
        "downloads": [("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                       "checkpoints/Qwen3-TTS-12Hz-1.7B-VoiceDesign", None)],
        "required": ["checkpoints/Qwen3-TTS-12Hz-1.7B-VoiceDesign/config.json",
                     "checkpoints/Qwen3-TTS-12Hz-1.7B-VoiceDesign/model*.safetensors"],
    },
    "moss": {
        "name": "MOSS Transcribe / Diarize",
        "downloads": [("OpenMOSS-Team/MOSS-Transcribe-Diarize", "checkpoints/MOSS-Transcribe-Diarize", None)],
        "required": ["checkpoints/MOSS-Transcribe-Diarize/config.json",
                     "checkpoints/MOSS-Transcribe-Diarize/*.safetensors"],
    },
    "hy-mt": {
        "name": "HY-MT translation",
        "downloads": [("tencent/Hy-MT2-1.8B", "checkpoints/hy-mt", None)],
        "required": ["checkpoints/hy-mt/config.json", "checkpoints/hy-mt/*.safetensors"],
    },
    "confucius": {
        "name": "Confucius4-TTS + encoders",
        "downloads": [
            ("netease-youdao/Confucius4-TTS", "Confucius4-TTS/checkpoints",
             ["t2s_model.safetensors", "s2a_model.pt", "wav2vec2bert_stats.pt", "special_tokens_map.json",
              "tokenizer.json", "tokenizer.model", "tokenizer_config.json"]),
            ("facebook/w2v-bert-2.0", "Confucius4-TTS/pretrained/w2v-bert-2.0", None),
            ("nvidia/bigvgan_v2_22khz_80band_256x", "Confucius4-TTS/pretrained/bigvgan_v2_22khz_80band_256x", None),
            ("funasr/campplus", "Confucius4-TTS/pretrained/campplus", ["campplus_cn_common.bin"]),
        ],
        "required": ["Confucius4-TTS/checkpoints/t2s-vllm/config.json",
                     "Confucius4-TTS/checkpoints/t2s-vllm/model.safetensors",
                     "Confucius4-TTS/checkpoints/s2a_model.pt",
                     "Confucius4-TTS/checkpoints/wav2vec2bert_stats.pt",
                     "Confucius4-TTS/pretrained/w2v-bert-2.0/model.safetensors",
                     "Confucius4-TTS/pretrained/bigvgan_v2_22khz_80band_256x/bigvgan_generator.pt",
                     "Confucius4-TTS/pretrained/campplus/campplus_cn_common.bin",
                     "Confucius4-TTS/config/inference_config.modal.yaml"],
    },
}
for _variant in ("medium", "small-music", "small-sfx"):
    MODELS[f"stable-audio-{_variant}"] = {
        "name": f"Stable Audio 3 · {_variant}", "gated": True,
        "downloads": [(f"stabilityai/stable-audio-3-{_variant}", f"checkpoints/stable-audio-3/{_variant}", None)],
        "required": [f"checkpoints/stable-audio-3/{_variant}/model_config.json",
                     f"checkpoints/stable-audio-3/{_variant}/model.safetensors|"
                     f"checkpoints/stable-audio-3/{_variant}/model.ckpt"],
    }


def read_manifest(root: Path) -> dict:
    path = root / "manager-runtime.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_manifest(root: Path, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "manager-runtime.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(root / "manager-runtime.json")


def repository_source(root: Path, fallback: Path) -> Path:
    """Use an explicitly updated WebUI checkout, otherwise the deployed source."""
    manifest = read_manifest(root)
    source = root / REPOSITORIES["index"][2] if manifest.get("source") else fallback
    if not (source / "fastapi_webui_v2.py").is_file():
        raise FileNotFoundError("WebUI source is incomplete; update the IndexTTS repository in the manager")
    return source


def validate_operation(action: str, target: str) -> None:
    catalog = {"prepare-models": {"all"}, "update-repo": REPOSITORIES, "download-model": MODELS}
    if action not in catalog or target not in catalog[action]:
        raise ValueError("Unknown manager action or target")


def missing_model_files(root: Path, model: dict) -> list[str]:
    missing = []
    for requirement in model["required"]:
        candidates = [pattern for pattern in requirement.split("|")
                      if any(path.is_file() and path.stat().st_size > 0 for path in root.glob(pattern))]
        if not candidates:
            missing.append(requirement)
            continue
        # Either complete format is usable. A partial safetensors conversion
        # must not hide a valid PyTorch bundle (or vice versa).
        failed_formats = []
        for pattern in candidates:
            problems = []
            extension = next((ext for ext in ("safetensors", "bin") if f".{ext}" in pattern), None)
            indexes = (root / Path(pattern).parent).glob(f"*.{extension}.index.json") if extension else ()
            for index in indexes:
                try:
                    weights = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
                    if not isinstance(weights, dict) or not weights or not all(
                        isinstance(filename, str) for filename in weights.values()
                    ):
                        raise ValueError("Invalid shard index")
                except (ValueError, KeyError, TypeError):
                    problems.append(index.relative_to(root).as_posix())
                    continue
                for filename in set(weights.values()):
                    shard = index.parent / filename
                    if not shard.is_file() or shard.stat().st_size == 0:
                        problems.append(shard.relative_to(root).as_posix())
            if not problems:
                break
            failed_formats.append(problems)
        else:
            missing.extend(min(failed_formats, key=len))
    return missing


class ModelSetup:
    def __init__(self, root: Path, source: Path, *, emit=print, patch_confucius=None):
        self.root, self.source = root, source
        self.emit, self.patch_confucius = emit, patch_confucius

    def run(self, command: list[str], *, cwd: Path | None = None, env: dict | None = None) -> str:
        self.emit("$ " + " ".join(command))
        # Stream progress while pip, Git, Hub and conversion run. Never use a shell.
        lines = []
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                line = line.rstrip()
                self.emit(line)
                lines.append(line)
                lines = lines[-200:]
            code = process.wait()
        if code:
            raise RuntimeError(f"Command failed ({code}): {' '.join(command[:3])}\n" + "\n".join(lines[-10:]))
        return "\n".join(lines)

    def bootstrap(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoints", "outputs", "speaker_presets", "emotion_cache"):
            (self.root / name).mkdir(exist_ok=True)
        # Seed sibling sources at the image's tested revisions. Downloading a
        # model must never silently fetch newer code or reset a managed checkout.
        for component, image_path in (
            ("confucius", Path("/app/Confucius4-TTS")),
            ("index25", Path("/app/index-tts-2.5-vllm-omni-experiment")),
        ):
            destination = self.root / REPOSITORIES[component][2]
            if not (destination / ".git").exists():
                shutil.copytree(image_path, destination, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", ".venv-indextts25"))

    def status(self) -> dict:
        manifest = read_manifest(self.root)
        repositories = []
        for key, (name, url, relative) in REPOSITORIES.items():
            path = self.root / relative
            origin = "persistent"
            if key == "index":
                # The default source is shipped in the image without .git.
                # A managed checkout is used only after explicit activation.
                if not manifest.get("source"):
                    path, origin = self.source, "image"
                else:
                    origin = "managed"
            elif not path.is_dir():
                path, origin = IMAGE_REPOSITORIES[key], "image"
            revision = ""
            if (path / ".git").exists():
                result = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                                        capture_output=True, text=True, check=False)
                revision = result.stdout.strip() if result.returncode == 0 else ""
            available = (path / REPOSITORY_ENTRYPOINTS[key]).is_file()
            repositories.append({"id": key, "name": name, "url": url, "revision": revision,
                                 "ready": available, "origin": origin, "path": str(path),
                                 "status_label": ("Bundled source" if origin == "image" else "Source available")
                                 if available else "Source missing"})
        models = []
        for key, model in MODELS.items():
            missing = missing_model_files(self.root, model)
            policy = "default" if key == "index" else "on-demand"
            # Disk availability and GPU residency are separate. Legacy setup
            # markers may say 'running' after a failed validation or interrupted
            # worker; actual checkpoint files determine download availability.
            models.append({"id": key, "name": model["name"], "ready": not missing,
                           "missing": missing, "gated": model.get("gated", False),
                           "downloaded": not missing, "load_policy": policy,
                           "status_label": ("Downloaded" if not missing else "Download needed")
                           + (" · default backend" if policy == "default" else " · loads on demand"),
                           "repos": [download[0] for download in model["downloads"]]})
        environments = []
        for key, (name, fallback) in COMPONENTS.items():
            python = Path(fallback)
            installed = python.is_file()
            environments.append({"id": key, "name": name,
                                 "python": str(python), "ready": installed,
                                 "status_label": "Installed in image" if installed
                                 else "Environment missing"})
        return {"repositories": repositories, "models": models, "environments": environments,
                "managed_source": bool(manifest.get("source")),
                "restart_required": bool(manifest.get("restart_required"))}

    def update_repo(self, target: str) -> None:
        _, url, relative = REPOSITORIES[target]
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not (path / ".git").exists():
            self.run(["git", "clone", url, str(path)])
        else:
            self.run(["git", "remote", "set-url", "origin", url], cwd=path)
            self.run(["git", "fetch", "origin"], cwd=path)
            # The Omni image uses a detached tested commit. Resolve the remote's
            # default branch so Update to latest also works from that checkout.
            self.run(["git", "remote", "set-head", "origin", "--auto"], cwd=path)
            ref = self.run(["git", "symbolic-ref", "refs/remotes/origin/HEAD"], cwd=path).strip()
            self.run(["git", "reset", "--hard", ref], cwd=path)
        if target == "index" and not (path / "fastapi_webui_v2.py").is_file():
            raise FileNotFoundError("Updated IndexTTS repository has no WebUI launcher")
        if target == "confucius":
            self.configure_confucius(convert=False)
        manifest = read_manifest(self.root)
        if target == "index":
            manifest["source"] = True
        manifest["restart_required"] = True
        write_manifest(self.root, manifest)
        self.emit("Repository updated. Redeploy the GPU service to refresh its memory snapshot.")

    def download_model(self, target: str) -> None:
        # Run Hub in a child process to stream its output and keep imports out of
        # the management web server. The Modal secret supplies HF_TOKEN.
        model = MODELS[target]
        manifest = read_manifest(self.root)
        manifest.setdefault("downloads", {})[target] = "running"
        write_manifest(self.root, manifest)
        program = """import json, sys
from huggingface_hub import snapshot_download
for repo, destination, patterns in json.loads(sys.argv[1]):
    print('Downloading ' + repo, flush=True)
    snapshot_download(repo_id=repo, local_dir=destination, allow_patterns=patterns)
"""
        downloads = [(repo, str(self.root / relative), patterns) for repo, relative, patterns in model["downloads"]]
        try:
            self.run([sys.executable, "-u", "-c", program, json.dumps(downloads)])
            if target == "confucius":
                self.configure_confucius(convert=True)
            missing = missing_model_files(self.root, model)
            if missing:
                raise RuntimeError("Downloaded bundle is incomplete: " + ", ".join(missing))
        except Exception:
            manifest = read_manifest(self.root)
            manifest.setdefault("downloads", {})[target] = "failed"
            write_manifest(self.root, manifest)
            raise
        manifest = read_manifest(self.root)
        manifest.setdefault("downloads", {})[target] = "completed"
        manifest["restart_required"] = True
        write_manifest(self.root, manifest)
        self.emit("Model download verified.")

    def configure_confucius(self, *, convert: bool) -> None:
        repo = self.root / REPOSITORIES["confucius"][2]
        if self.patch_confucius:
            status = self.patch_confucius(repo)
            self.emit(status["message"])
            if not status["success"]:
                raise RuntimeError(status["message"])
        source = repo / "config/inference_config.yaml"
        config = repo / "config/inference_config.modal.yaml"
        text = source.read_text(encoding="utf-8").replace(
            "  w2v_bert_path: facebook/w2v-bert-2.0", "  w2v_bert_path: ./pretrained/w2v-bert-2.0"
        ).replace("  vocoder_path: nvidia/bigvgan_v2_22khz_80band_256x",
                  "  vocoder_path: ./pretrained/bigvgan_v2_22khz_80band_256x")
        config.write_text(text, encoding="utf-8")
        (repo / "outputs/api").mkdir(parents=True, exist_ok=True)
        converted = repo / "checkpoints/t2s-vllm"
        if convert and not all((converted / name).is_file() and (converted / name).stat().st_size > 0
                               for name in ("config.json", "model.safetensors")):
            python = COMPONENTS["confucius"][1]
            environment = {**os.environ, "PYTHONPATH": str(repo)}
            self.run([python, "tools/convert_t2s_vllm.py", "--config", str(config),
                      "--output", str(repo / "checkpoints/t2s-vllm"),
                      "--checkpoint", str(repo / "checkpoints/t2s_model.safetensors")], cwd=repo, env=environment)

    def prepare_models(self) -> None:
        """Initialize missing model files using the existing image environments."""
        for target, model in MODELS.items():
            if not missing_model_files(self.root, model):
                self.emit(f"{model['name']}: existing files verified; skipping download.")
                continue
            try:
                self.download_model(target)
            except Exception as exc:
                if not model.get("gated"):
                    raise
                self.emit(f"Warning: optional {model['name']} was not downloaded: {exc}")
        self.emit("Model preparation complete.")

    def execute(self, action: str, target: str) -> None:
        validate_operation(action, target)
        self.bootstrap()
        if action == "prepare-models":
            self.prepare_models()
        else:
            {"update-repo": self.update_repo, "download-model": self.download_model}[action](target)
