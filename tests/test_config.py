import unittest

from indextts_web.config import AppSettings, env_flag, env_float, env_int, load_settings


class ConfigTests(unittest.TestCase):
    def test_environment_coercion_and_bounds(self):
        env = {"YES": "yes", "BAD": "wat", "BIG": "999", "RATIO": "1.5"}
        self.assertTrue(env_flag(env, "YES", False))
        self.assertTrue(env_flag(env, "MISSING", True))
        self.assertEqual(env_int(env, "BAD", 4), 4)
        self.assertEqual(env_int(env, "BIG", 4, maximum=100), 100)
        self.assertEqual(env_float(env, "RATIO", 0.1, maximum=1.0), 1.0)

    def test_cli_settings_are_typed(self):
        settings = load_settings(
            [
                "--port",
                "9000",
                "--tts_backend",
                "higgs",
                "--higgs_top_k",
                "12",
                "--no-higgs_manage_backend",
            ],
            environ={},
        )
        self.assertIsInstance(settings, AppSettings)
        self.assertEqual(settings.port, 9000)
        self.assertEqual(settings.tts_backend, "higgs")
        self.assertEqual(settings.higgs_top_k, 12)
        self.assertFalse(settings.higgs_manage_backend)

    def test_unknown_args_can_be_ignored_by_embedding_hosts(self):
        settings = load_settings(["--port", "8123", "--host-owned-flag"], allow_unknown=True)
        self.assertEqual(settings.port, 8123)


if __name__ == "__main__":
    unittest.main()

