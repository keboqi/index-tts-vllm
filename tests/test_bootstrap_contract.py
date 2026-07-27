import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class BootstrapContractTests(unittest.TestCase):
    def test_public_launcher_stays_thin(self):
        source = (ROOT / "fastapi_webui_v2.py").read_text(encoding="utf-8")
        nonempty = [line for line in source.splitlines() if line.strip()]
        self.assertLessEqual(len(nonempty), 10)
        self.assertIn("from indextts_web.main import app, main", source)

    def test_legacy_module_has_no_top_level_directory_creation(self):
        source = (ROOT / "legacy_fastapi_webui_v2.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        violations = []
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                rendered = ast.unparse(node.value.func)
                if rendered.endswith((".mkdir", ".makedirs")):
                    violations.append((node.lineno, rendered))
        self.assertEqual(violations, [])

    def test_runtime_directories_are_lifespan_owned(self):
        source = (ROOT / "legacy_fastapi_webui_v2.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        lifespan = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
        )
        calls = {
            ast.unparse(node.func)
            for node in ast.walk(lifespan)
            if isinstance(node, ast.Call)
        }
        self.assertIn("_ensure_runtime_directories", calls)


if __name__ == "__main__":
    unittest.main()

