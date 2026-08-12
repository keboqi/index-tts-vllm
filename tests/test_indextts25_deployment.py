import unittest
from pathlib import Path


class IndexTTS25DeploymentTests(unittest.TestCase):
    def test_modal_image_and_runtime_use_isolated_omni_environment(self):
        root = Path(__file__).resolve().parents[1]
        deployment = (root / "deploy_vllm_indextts_v2.py").read_text(encoding="utf-8")
        for expected in (
            "INDEXTTS25_REPO_URL",
            "INDEXTTS25_VENV_DIR",
            "INDEXTTS25_PERSISTENT_MODEL_DIR",
            "IndexTeam/IndexTTS-2.5",
            "facebook/w2v-bert-2.0",
            "funasr/campplus",
            "nvidia/bigvgan_v2_22khz_80band_256x",
            "--indextts25_start_command",
            "--indextts25_max_parallel_segments",
            "vllm_omni\" / \"deploy\" / \"indextts2_5.yaml",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, deployment)
        self.assertIn("uv venv --python 3.11", deployment)
        self.assertIn("vllm==0.27.0", deployment)

    def test_frontend_exposes_index25_in_synthesis_and_translation(self):
        root = Path(__file__).resolve().parents[1]
        html = (root / "index_new.html").read_text(encoding="utf-8")
        self.assertEqual(html.count('option value="index25"'), 2)
        translation = (root / "static" / "js" / "translation-state.js").read_text(encoding="utf-8")
        self.assertIn("index25: INDEXTTS25_DESTINATION_LANGUAGES", translation)


if __name__ == "__main__":
    unittest.main()
