#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-only
##
# @file test_cppcheck.py
# @brief Verify actual diagnostics and tool failures reach the CPPCheck CI gate.
# @package test_cppcheck
"""Run with python3 .github/workflows/static.check.scripts/test_cppcheck.py."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class CppcheckGate(unittest.TestCase):
    """@brief Invoke the production shell wrapper on isolated source files."""

    def setUp(self):
        """@brief Keep generated source outside the repository's changed list."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.script = Path(__file__).with_name("cppcheck.sh").resolve()

    def check(self, source, suffix="c", env=None):
        """@brief Run the same error-level wrapper used by the workflow."""
        path = self.root / ("input." + suffix)
        path.write_text(source, encoding="ascii")
        changed = self.root / "changed"
        changed.write_text(str(path) + "\n", encoding="ascii")
        return subprocess.run(
            ["bash", str(self.script), str(changed), "0"],
            cwd=self.root, env=env, capture_output=True, text=True, check=False,
        )

    def test_valid_source_p(self):
        """@brief Clean input must not fail merely because a tool was updated."""
        result = self.check("int value(void) { return 0; }\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("cppcheck result is ok", result.stdout)

    def test_out_of_bounds_c_n(self):
        """@brief Real C array-access diagnostics must fail the workflow."""
        result = self.check("int value(void) { int data[1] = {0}; return data[2]; }\n")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("[arrayIndexOutOfBounds]", result.stdout)

    def test_null_dereference_cpp_n(self):
        """@brief Real C++ error diagnostics must also reach the failure gate."""
        result = self.check("int value() { int *p = nullptr; return *p; }\n", "cpp")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("[nullPointer]", result.stdout)

    def test_glib_format_macros_p(self):
        """@brief Model format fragments without hiding real diagnostics."""
        result = self.check(
            'void log_value(const char *, ...);\n'
            'void value(void) { log_value("%" G_GSIZE_FORMAT " %" '
            'G_GINT64_FORMAT " %" GST_PTR_FORMAT, 1, 2, 0); }\n'
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_caps_and_version_macros_p(self):
        """@brief Select supported APIs and concatenate modeled caps strings."""
        result = self.check(
            '#if GST_CHECK_VERSION(1, 20, 0) && GLIB_CHECK_VERSION(2, 48, 0)\n'
            'const char *value(void) { return GST_TENSOR_CAP_DEFAULT ";" '
            'GST_TENSORS_CAP_DEFAULT ";" GST_TENSORS_FLEX_CAP_DEFAULT ";" '
            'GST_VIDEO_CAPS_MAKE("RGB") ";" GST_AUDIO_CAPS_MAKE("S16LE"); }\n'
            '#endif\n'
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_glib_resource_guard_p(self):
        """@brief G_UNLIKELY preserves an actual allocation-failure guard."""
        result = self.check(
            '#include <dirent.h>\n'
            'void value(const char *path) { DIR *d = opendir(path); '
            'if (G_UNLIKELY(d == 0)) return; readdir(d); closedir(d); }\n'
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_bounds_error_inside_version_guard_n(self):
        """@brief Configuring a version predicate must not skip its code."""
        result = self.check(
            '#if GST_CHECK_VERSION(1, 20, 0)\n'
            'int value(void) { int data[1] = {0}; return data[2]; }\n'
            '#endif\n'
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("[arrayIndexOutOfBounds]", result.stdout)

    def test_null_error_inside_glib_macro_n(self):
        """@brief A modeled branch hint must retain real null dereferences."""
        result = self.check(
            'int value(void) { int *p = 0; '
            'if (G_UNLIKELY(p == 0)) return *p; return 1; }\n'
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("[nullPointer]", result.stdout)

    def test_unconfigured_macro_n(self):
        """@brief Unknown configuration errors remain failures, not suppression."""
        result = self.check('const char *value = "a" UNMODELED_FORMAT "b";\n')
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("[unknownMacro]", result.stdout)

    def test_tool_failure_without_diagnostic_n(self):
        """@brief A crashed checker with empty stderr must never be a pass."""
        tools = self.root / "bin"
        tools.mkdir()
        checker = tools / "cppcheck"
        checker.write_text(
            '#!/bin/sh\nif [ "$1" = "--version" ]; then\n'
            '  echo "Cppcheck 2.19.0"\nelse\n  exit 42\nfi\n', encoding="ascii",
        )
        checker.chmod(0o755)
        env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"])
        result = self.check("int value(void) { return 0; }\n", env=env)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("cppcheck shows errors in:", result.stdout)


if __name__ == "__main__":
    unittest.main()
