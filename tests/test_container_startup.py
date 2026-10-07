"""Exercise container startup without downloading weights or loading CUDA."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GIT_BASH = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
BASH = str(GIT_BASH) if os.name == "nt" and GIT_BASH.is_file() else shutil.which("bash")
WEB_ASSETS = (
    "config.yaml", "gpt.pth", "bpe.model", "s2mel.pth", "wav2vec2bert_stats.pt", "feat1.pt", "feat2.pt",
    "gpt/config.json", "gpt/model.safetensors",
    "qwen0.6bemo4-merge/config.json", "qwen0.6bemo4-merge/model.safetensors",
    "w2v-bert-2.0/config.json", "w2v-bert-2.0/model.safetensors", "w2v-bert-2.0/preprocessor_config.json",
    "semantic_codec/model.safetensors", "campplus/campplus_cn_common.bin",
    "bigvgan/config.json", "bigvgan/bigvgan_generator.pt",
)
LEGACY_ASSETS = ("config.yaml", "gpt.pth", "bpe.model", "bigvgan_generator.pth")


@unittest.skipUnless(BASH, "Bash is required to exercise the container entrypoint")
class ContainerStartupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.models = self.root / "models"
        self.commands = self.root / "bin"
        self.commands.mkdir()
        self.launch_log = self.root / "launch.txt"
        self.download_log = self.root / "download.txt"
        self.conversion_log = self.root / "conversion.txt"
        self.stub("python3", """
if [[ "$1" == "-" ]]; then
    printf '%s' "$2" > "$DOWNLOAD_LOG"
    while IFS= read -r file; do
        mkdir -p "$MODEL_DIR/$(dirname "$file")"
        : > "$MODEL_DIR/$file"
    done <<< "$DOWNLOAD_FILES"
    cat > /dev/null
else
    printf 'VLLM_USE_V1=%s\\n' "${VLLM_USE_V1:-}" > "$LAUNCH_LOG"
    printf '%s\\n' "$@" >> "$LAUNCH_LOG"
fi
""")
        self.stub("modelscope", "printf 'unexpected ModelScope download\\n' >&2\nexit 90\n")
        self.stub("python", """
[[ "$1" == "convert_hf_format.py" && "$2" == "--model_dir" ]] || exit 90
printf '%s' "$3" > "$CONVERSION_LOG"
mkdir -p "$3/gpt"
touch "$3/gpt/config.json" "$3/gpt/pytorch_model.bin"
""")
        self.stub("wget", ': > "${@: -1}"\n')

    def stub(self, name, body):
        target = self.commands / name
        target.write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\n" + body, encoding="utf-8")
        target.chmod(0o755)

    def assets(self, filenames):
        for filename in filenames:
            target = self.models / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()

    def run_entrypoint(self, **overrides):
        environment = os.environ.copy()
        for name in ("APP_SERVER", "MODEL", "VLLM_USE_MODELSCOPE", "CONVERT_MODEL", "VLLM_USE_V1"):
            environment.pop(name, None)
        command_path = self.commands.as_posix()
        if os.name == "nt":
            command_path = "/" + command_path[0].lower() + command_path[2:]
        environment.update(
            PATH=command_path + os.pathsep + environment.get("PATH", ""),
            MODEL_DIR=self.models.as_posix(), DOWNLOAD_MODEL="0", PORT="8123",
            LAUNCH_LOG=self.launch_log.as_posix(), DOWNLOAD_LOG=self.download_log.as_posix(),
            CONVERSION_LOG=self.conversion_log.as_posix(), DOWNLOAD_FILES="\n".join(WEB_ASSETS),
        )
        environment.update(overrides)
        return subprocess.run(
            [BASH, str(ROOT / "entrypoint.sh"), "--host", "127.0.0.1"],
            env=environment, capture_output=True, text=True, timeout=20, check=False,
        )

    def test_web_uses_existing_preconverted_bundle_without_markers(self):
        self.assets(WEB_ASSETS)
        result = self.run_entrypoint(PORT="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("garyswansrs/index_tts_2_vllm", result.stdout)
        self.assertIn("fastapi_webui_v2.py\n", self.launch_log.read_text())
        self.assertIn("--port\n8000\n--host\n127.0.0.1\n", self.launch_log.read_text())
        self.assertFalse(self.conversion_log.exists())

    def test_web_downloads_the_preconverted_bundle_from_huggingface(self):
        result = self.run_entrypoint(DOWNLOAD_MODEL="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.download_log.read_text(), "garyswansrs/index_tts_2_vllm")
        self.assertTrue(self.launch_log.exists())
        self.assertIn("--port\n8123\n", self.launch_log.read_text())
        self.assertFalse(self.conversion_log.exists())

    def test_web_rejects_legacy_assets_with_download_disabled(self):
        self.assets(LEGACY_ASSETS)
        result = self.run_entrypoint()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DOWNLOAD_MODEL=0", result.stderr)
        self.assertFalse(self.launch_log.exists())

    def test_legacy_converts_into_the_gpt_directory_before_startup(self):
        self.assets(LEGACY_ASSETS)
        result = self.run_entrypoint(APP_SERVER="legacy-api", PORT="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IndexTeam/IndexTTS-1.5", result.stdout)
        self.assertTrue(self.conversion_log.exists())
        self.assertIn("VLLM_USE_V1=0\napi_server.py\n", self.launch_log.read_text())
        self.assertIn("--port\n8001\n", self.launch_log.read_text())

    def test_legacy_reuses_existing_conversion_without_marker(self):
        self.assets((*LEGACY_ASSETS, "gpt/config.json", "gpt/tokenizer.json", "gpt/pytorch_model.bin"))
        result = self.run_entrypoint(APP_SERVER="legacy-api", CONVERT_MODEL="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.conversion_log.exists())

    def test_invalid_server_and_web_conversion_fail_before_launch(self):
        for overrides in ({"APP_SERVER": "unknown"}, {"CONVERT_MODEL": "1"}):
            with self.subTest(overrides=overrides):
                result = self.run_entrypoint(**overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.launch_log.exists())
                self.assertFalse(self.download_log.exists())


if __name__ == "__main__":
    unittest.main()
