"""
test_package_imports.py

Verify that the codebase has clean import structure:
- No sys.path hacks in entrypoints or test files
- Key inner imports are at module level (not inside functions)
"""

import ast
import os
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN_DIR = os.path.join(PROJECT_ROOT, "plugins", "tidal")


def _has_sys_path_insert(filepath):
    """Parse a Python file's AST and check for sys.path.insert calls."""
    with open(filepath, "r") as f:
        tree = ast.parse(f.read(), filename=filepath)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Match sys.path.insert(...)
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "insert"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "path"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "sys"
            ):
                return True
    return False


class TestNoSysPathHacks(unittest.TestCase):
    """Ensure no entrypoint or test file hacks sys.path."""

    ENTRYPOINTS = [
        os.path.join(PLUGIN_DIR, "Main.py"),
        os.path.join(PLUGIN_DIR, "train_rl.py"),
        os.path.join(PLUGIN_DIR, "Generator.py"),
        os.path.join(PLUGIN_DIR, "inference_server.py"),
    ]

    def test_no_sys_path_hack_in_entrypoints(self):
        for filepath in self.ENTRYPOINTS:
            with self.subTest(file=os.path.basename(filepath)):
                self.assertFalse(
                    _has_sys_path_insert(filepath),
                    f"{filepath} still contains sys.path.insert",
                )

    def test_no_sys_path_hack_in_test_files(self):
        test_dir = os.path.join(PLUGIN_DIR, "tests")
        for filename in os.listdir(test_dir):
            if filename.startswith("test_") and filename.endswith(".py"):
                filepath = os.path.join(test_dir, filename)
                with self.subTest(file=filename):
                    self.assertFalse(
                        _has_sys_path_insert(filepath),
                        f"{filepath} still contains sys.path.insert",
                    )


class TestInnerImports(unittest.TestCase):
    """Verify that unjustified inner imports have been moved to module level."""

    def _get_module_level_imports(self, filepath):
        """Return set of imported names at module level."""
        with open(filepath, "r") as f:
            tree = ast.parse(f.read(), filename=filepath)

        names = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.add(node.module)
                for alias in node.names:
                    names.add(alias.name)
        return names

    def test_no_inner_import_of_TransformerLM_in_Trainer(self):
        filepath = os.path.join(PLUGIN_DIR, "Trainer.py")
        names = self._get_module_level_imports(filepath)
        self.assertIn(
            "TransformerLM",
            names,
            "TransformerLM should be imported at module level in Trainer.py",
        )

    def test_counter_import_at_module_level(self):
        filepath = os.path.join(PLUGIN_DIR, "TransformerLM.py")
        names = self._get_module_level_imports(filepath)
        self.assertIn(
            "Counter",
            names,
            "Counter should be imported at module level in TransformerLM.py",
        )


if __name__ == "__main__":
    unittest.main()
