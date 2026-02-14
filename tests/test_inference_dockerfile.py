"""
Tests for the inference Dockerfile and related Docker configuration.

Validates that:
1. inference.Dockerfile uses ARG PLUGIN (plugin-agnostic, not hardcoded)
2. .dockerignore excludes heavy/unnecessary dirs from build context
3. docker-compose.prod.yml passes the PLUGIN build arg to the inference service
"""

import os
import re
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestInferenceDockerfile(unittest.TestCase):
    """Validate inference.Dockerfile is plugin-agnostic."""

    def setUp(self):
        dockerfile_path = os.path.join(PROJECT_ROOT, "inference.Dockerfile")
        with open(dockerfile_path, "r") as f:
            self.content = f.read()
        self.lines = self.content.strip().splitlines()

    def test_has_arg_plugin_with_default(self):
        """Dockerfile should declare ARG PLUGIN with a default value."""
        pattern = re.compile(r"^ARG\s+PLUGIN\s*=\s*\S+", re.MULTILINE)
        self.assertRegex(self.content, pattern,
                         "Expected ARG PLUGIN=<default> in inference.Dockerfile")

    def test_copies_plugin_directory(self):
        """Dockerfile should COPY plugins/__init__.py and plugins/${PLUGIN}/ preserving package structure."""
        # Should copy the plugins package __init__.py
        init_pattern = re.compile(r"^COPY\s+plugins/__init__\.py\s+plugins/__init__\.py", re.MULTILINE)
        self.assertRegex(self.content, init_pattern,
                         "Expected COPY plugins/__init__.py plugins/__init__.py in inference.Dockerfile")
        # Should copy the plugin directory into plugins/${PLUGIN}/ (not flat into .)
        plugin_pattern = re.compile(r"^COPY\s+plugins/\$\{?PLUGIN\}?/\s+plugins/\$\{?PLUGIN\}?/", re.MULTILINE)
        self.assertRegex(self.content, plugin_pattern,
                         "Expected COPY plugins/${PLUGIN}/ plugins/${PLUGIN}/ in inference.Dockerfile")

    def test_no_hardcoded_model_file_copies(self):
        """Dockerfile should NOT have individual COPY lines for model .py files."""
        hardcoded_files = [
            "TransformerLM.py", "DataPipeline.py", "Generator.py",
            "GatingPolicyAgent.py", "GatingModulator.py", "GatingEnvironment.py",
            "inference_server.py",
        ]
        for filename in hardcoded_files:
            pattern = re.compile(rf"^COPY\s+{re.escape(filename)}\s+\.", re.MULTILINE)
            self.assertNotRegex(self.content, pattern,
                                f"Found hardcoded COPY for {filename} — "
                                "should use COPY plugins/${{PLUGIN}}/ . instead")

    def test_no_hardcoded_configs_copy(self):
        """Dockerfile should NOT have a separate COPY configs/ line."""
        pattern = re.compile(r"^COPY\s+configs/\s+configs/", re.MULTILINE)
        self.assertNotRegex(self.content, pattern,
                            "Found hardcoded COPY configs/ — configs/ is inside the "
                            "plugin directory and should be copied with the plugin")

    def test_tokenizer_predownload_preserved(self):
        """The GPT-2 tokenizer pre-download step should still be present."""
        self.assertIn("AutoTokenizer.from_pretrained", self.content)

    def test_gunicorn_cmd_preserved(self):
        """The CMD should still run gunicorn with plugins.tidal.inference_server:app."""
        self.assertIn("gunicorn", self.content)
        self.assertIn("plugins.tidal.inference_server:app", self.content)


class TestDockerignore(unittest.TestCase):
    """Validate .dockerignore at project root excludes heavy directories."""

    def setUp(self):
        dockerignore_path = os.path.join(PROJECT_ROOT, ".dockerignore")
        self.assertTrue(os.path.exists(dockerignore_path),
                        "Expected .dockerignore at project root")
        with open(dockerignore_path, "r") as f:
            self.entries = [
                line.strip() for line in f.readlines()
                if line.strip() and not line.startswith("#")
            ]

    def test_excludes_virtualenv(self):
        self.assertIn("tidal-env/", self.entries,
                       ".dockerignore should exclude tidal-env/")

    def test_excludes_git(self):
        self.assertIn(".git/", self.entries,
                       ".dockerignore should exclude .git/")

    def test_excludes_legacy_research(self):
        self.assertIn("legacy_research/", self.entries,
                       ".dockerignore should exclude legacy_research/")

    def test_excludes_pycache(self):
        matches = [e for e in self.entries if "__pycache__" in e]
        self.assertTrue(matches, ".dockerignore should exclude __pycache__/")

    def test_excludes_plugin_tests(self):
        matches = [e for e in self.entries if "tests" in e and "plugin" in e.lower()
                   or e == "plugins/*/tests/"]
        self.assertTrue(matches, ".dockerignore should exclude plugins/*/tests/")


class TestComposeInferencePlugin(unittest.TestCase):
    """Validate docker-compose.prod.yml passes the PLUGIN build arg."""

    def setUp(self):
        compose_path = os.path.join(PROJECT_ROOT, "dashboard", "docker-compose.prod.yml")
        with open(compose_path, "r") as f:
            self.content = f.read()

    def test_inference_has_plugin_arg(self):
        """The inference service should pass PLUGIN as a build arg."""
        # Look for args: section under inference build with PLUGIN key
        self.assertIn("PLUGIN:", self.content,
                      "Expected PLUGIN: in inference service build args")

    def test_inference_plugin_default_is_tidal(self):
        """The default PLUGIN value should be 'tidal'."""
        pattern = re.compile(r"PLUGIN:\s*tidal")
        self.assertRegex(self.content, pattern,
                         "Expected PLUGIN: tidal in compose args")


if __name__ == "__main__":
    unittest.main()
