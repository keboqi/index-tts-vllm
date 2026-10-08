import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from indextts_web.infrastructure.modal_runtime import PERSISTENT_DIRECTORIES, prepare_runtime_code
from indextts_web.infrastructure.model_manager import create_manager_app
from indextts_web.infrastructure.model_setup import (
    MODELS,
    ModelSetup,
    missing_model_files,
    read_manifest,
    repository_source,
    validate_operation,
    write_manifest,
)
from tests.support import ROOT, load_definition


class ManagerAPITests(unittest.TestCase):
    def setUp(self):
        self.state = Mock(return_value={"repositories": [], "models": [], "environments": []})
        self.execute = Mock(side_effect=lambda action, target, emit: emit("Downloading"))
        self.app = create_manager_app(get_state=self.state, execute_operation=self.execute)
        self.client = TestClient(self.app)

    def test_page_opens_without_provisioning_or_token(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Models & Setup", response.text)
        self.assertIn("Initialize</button>", response.text)
        self.assertNotIn('id="login"', response.text)
        self.assertNotIn("Authorization", response.text)
        self.assertNotIn("Set up / repair", response.text)
        self.state.assert_not_called()
        self.execute.assert_not_called()

    def test_initialization_and_downloads_are_accessible_without_credentials(self):
        self.assertEqual(self.client.get("/api/state").status_code, 200)
        for action, target in (("prepare-models", "all"), ("download-model", "index")):
            response = self.client.post("/api/jobs", json={"action": action, "target": target})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()["status"], "queued")
            self.assertEqual(self.execute.call_args.args[:2], (action, target))
            result = self.client.get("/api/jobs/" + response.json()["id"]).json()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["logs"], ["Downloading"])

    def test_rejects_unknown_actions_and_path_targets_before_preparation(self):
        for payload in ({"action": "clear-cache", "target": "index"},
                        {"action": "setup-environment", "target": "main"},
                        {"action": "update-repo", "target": "https://example.com/repo"}):
            self.assertEqual(self.client.post("/api/jobs", json=payload).status_code, 422)
        self.execute.assert_not_called()

    def test_failures_are_visible(self):
        self.execute.side_effect = RuntimeError("download interrupted")
        response = self.client.post("/api/jobs", json={"action": "download-model", "target": "index"})
        result = self.client.get("/api/jobs/" + response.json()["id"]).json()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "download interrupted")
        self.assertEqual(self.client.get("/api/jobs/missing").status_code, 404)

    def test_status_polling_does_not_reload_volumes_during_preparation(self):
        entered, release = threading.Event(), threading.Event()
        self.client.get("/api/state")
        self.state.reset_mock()

        def execute(action, target, emit):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test did not release preparation")

        self.execute.side_effect = execute
        with ThreadPoolExecutor(max_workers=1) as executor:
            request = executor.submit(self.client.post, "/api/jobs",
                                      json={"action": "prepare-models", "target": "all"})
            try:
                self.assertTrue(entered.wait(3))
                response = self.client.get("/api/state")
                self.assertEqual(response.status_code, 200)
                self.state.assert_not_called()
            finally:
                release.set()
            self.assertEqual(request.result(timeout=3).status_code, 202)
        self.state.assert_called_once()

class ProvisioningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.app, self.source = self.root / "app", self.root / "source"
        self.app.mkdir()
        self.source.mkdir()
        self.setup = ModelSetup(self.app, self.source, emit=Mock())

    def test_invalid_operations_do_not_bootstrap(self):
        with patch.object(self.setup, "bootstrap") as bootstrap:
            with self.assertRaises(ValueError):
                self.setup.execute("clear-cache", "index")
            bootstrap.assert_not_called()
        with self.assertRaises(ValueError):
            validate_operation("download-model", "../index")

    def test_empty_or_partial_checkpoint_is_not_ready(self):
        directory = self.app / "checkpoints/hy-mt"
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors").touch()
        with patch("indextts_web.infrastructure.model_setup.COMPONENTS", {}):
            state = self.setup.status()
        self.assertFalse(next(model for model in state["models"] if model["id"] == "hy-mt")["ready"])
        (directory / "model.safetensors").write_bytes(b"weights")
        with patch("indextts_web.infrastructure.model_setup.COMPONENTS", {}):
            state = self.setup.status()
        self.assertTrue(next(model for model in state["models"] if model["id"] == "hy-mt")["ready"])

    def test_download_verifies_bundle_and_passes_only_selected_model(self):
        self.setup.run = Mock()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.setup.download_model("hy-mt")
        command = self.setup.run.call_args.args[0]
        downloads = json.loads(command[-1])
        self.assertEqual(len(downloads), 1)
        self.assertEqual(downloads[0][0], "tencent/Hy-MT2-1.8B")
        self.assertEqual(downloads[0][1], str(self.app / "checkpoints/hy-mt"))
        self.assertNotIn("token", command[-2])
        self.assertEqual(read_manifest(self.app)["downloads"]["hy-mt"], "failed")

    def test_sharded_model_requires_every_indexed_weight_file(self):
        directory = self.app / "checkpoints/hy-mt"
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}")
        (directory / "model-00001.safetensors").write_bytes(b"weights")
        (directory / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"one": "model-00001.safetensors", "two": "model-00002.safetensors"}}))
        missing = missing_model_files(self.app, {"required": ["checkpoints/hy-mt/*.safetensors"]})
        self.assertEqual(missing, ["checkpoints/hy-mt/model-00002.safetensors"])

    def test_gated_download_failure_is_reported_and_does_not_download_other_models(self):
        self.setup.run = Mock(side_effect=RuntimeError("403: gated repository"))
        with self.assertRaisesRegex(RuntimeError, "403"):
            self.setup.download_model("stable-audio-medium")
        self.setup.run.assert_called_once()

    def test_detached_repo_updates_remote_default_without_cleaning_model_data(self):
        repo = self.app / "index-tts-2.5-vllm-omni-experiment"
        (repo / ".git").mkdir(parents=True)
        model = repo / "models/keep.bin"
        model.parent.mkdir()
        model.write_bytes(b"existing")
        self.setup.run = Mock(return_value="refs/remotes/origin/main")
        self.setup.update_repo("index25")
        commands = [call.args[0] for call in self.setup.run.call_args_list]
        self.assertIn(["git", "remote", "set-head", "origin", "--auto"], commands)
        self.assertIn(["git", "reset", "--hard", "refs/remotes/origin/main"], commands)
        self.assertFalse(any("clean" in command for command in commands))
        self.assertEqual(model.read_bytes(), b"existing")

    def write_files(self, paths):
        for relative in paths:
            path = self.app / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")

    def test_default_pytorch_bundle_is_downloaded_even_with_stale_failed_marker(self):
        self.write_files(["checkpoints/config.yaml", "checkpoints/gpt/config.json",
                          "checkpoints/gpt/pytorch_model.bin", "checkpoints/s2mel.pth"])
        write_manifest(self.app, {"downloads": {"index": "running"}})
        state = self.setup.status()
        model = next(item for item in state["models"] if item["id"] == "index")
        self.assertTrue(model["downloaded"])
        self.assertEqual(model["missing"], [])
        self.assertEqual(model["load_policy"], "default")
        self.setup.run = Mock()
        self.setup.download_model("index")
        self.assertEqual(read_manifest(self.app)["downloads"]["index"], "completed")

    def test_pytorch_shard_validation_accepts_a_complete_alternative_format(self):
        directory = self.app / "checkpoints/gpt"
        directory.mkdir(parents=True)
        (directory / "pytorch_model-00001.bin").write_bytes(b"weights")
        (directory / "pytorch_model.bin.index.json").write_text(json.dumps({
            "weight_map": {"one": "pytorch_model-00001.bin", "two": "pytorch_model-00002.bin"}}))
        model = {"required": [MODELS["index"]["required"][2]]}
        self.assertEqual(missing_model_files(self.app, model), ["checkpoints/gpt/pytorch_model-00002.bin"])
        (directory / "model.safetensors").write_bytes(b"converted")
        self.assertEqual(missing_model_files(self.app, model), [])

    def test_bundled_webui_is_available_without_git_or_managed_checkout(self):
        (self.source / "fastapi_webui_v2.py").write_text("source")
        state = self.setup.status()
        webui = next(item for item in state["repositories"] if item["id"] == "index")
        self.assertTrue(webui["ready"])
        self.assertEqual(webui["status_label"], "Bundled source")
        self.assertEqual(repository_source(self.app, self.source), self.source)
        updated = self.app / "repositories/index"
        updated.mkdir(parents=True)
        (updated / "fastapi_webui_v2.py").write_text("updated")
        write_manifest(self.app, {"source": True})
        self.assertEqual(repository_source(self.app, self.source), updated)

    def test_downloaded_optional_models_do_not_claim_gpu_readiness(self):
        self.write_files(["checkpoints/hy-mt/config.json", "checkpoints/hy-mt/model.safetensors"])
        model = next(item for item in self.setup.status()["models"] if item["id"] == "hy-mt")
        self.assertTrue(model["downloaded"])
        self.assertEqual(model["load_policy"], "on-demand")
        self.assertEqual(model["status_label"], "Downloaded · loads on demand")

    def test_prepare_uses_existing_image_environments_and_keeps_gated_downloads_optional(self):
        self.setup.update_repo = Mock()
        def download(target):
            if MODELS[target].get("gated"):
                raise RuntimeError("403")
        self.setup.download_model = Mock(side_effect=download)
        self.setup.bootstrap = Mock()
        self.setup.execute("prepare-models", "all")
        self.setup.bootstrap.assert_called_once()
        self.setup.update_repo.assert_not_called()
        self.assertEqual([call.args[0] for call in self.setup.download_model.call_args_list], list(MODELS))
        with self.assertRaises(ValueError):
            validate_operation("setup-environment", "main")

    def test_initialize_skips_existing_complete_bundles(self):
        self.setup.bootstrap = Mock()
        self.setup.download_model = Mock()
        with patch("indextts_web.infrastructure.model_setup.missing_model_files", return_value=[]):
            self.setup.execute("prepare-models", "all")
        self.setup.download_model.assert_not_called()

    def test_runtime_uses_managed_source_without_copying_git_or_model_placeholders(self):
        (self.source / "app.py").write_text("latest")
        (self.source / ".git").mkdir()
        (self.source / "checkpoints").mkdir()
        (self.source / "checkpoints/config.yaml").write_text("placeholder")
        for name in PERSISTENT_DIRECTORIES:
            (self.app / name).mkdir(parents=True)
        with patch.object(Path, "symlink_to") as symlink:
            runtime = prepare_runtime_code(self.source, self.app, self.root / "runtime", managed_source=True)
        self.assertEqual((runtime / "app.py").read_text(), "latest")
        self.assertFalse((runtime / ".git").exists())
        self.assertEqual(symlink.call_count, len(PERSISTENT_DIRECTORIES))

    def test_single_web_function_commits_partial_downloads_after_failure(self):
        app_volume, cache_volume = Mock(), Mock()
        namespace = {"RUNTIME_SOURCE_DIR": str(ROOT), "Path": Path,
                     "PERSISTENT_APP_DIR": str(self.app), "app_storage": app_volume,
                     "cache_storage": cache_volume, "_ensure_confucius_vllm_patch_compatibility": Mock()}
        prepare = load_definition(ROOT / "deploy_vllm_indextts_v2.py", "prepare_model", namespace)
        with patch("indextts_web.infrastructure.model_manager.create_manager_app") as factory, \
                patch("indextts_web.infrastructure.model_setup.ModelSetup") as setup:
            prepare()
            execute = factory.call_args.kwargs["execute_operation"]
            setup.return_value.execute.side_effect = RuntimeError("interrupted")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                execute("download-model", "index", Mock())
        for volume in (app_volume, cache_volume):
            volume.reload.assert_called_once()
            volume.commit.assert_called_once()
