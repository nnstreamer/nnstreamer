#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-only
##
# @file test_ssat_fixtures.py
# @brief Regression cases for SSAT fixture-path validation.
# @package test_ssat_fixtures
# @author Minsoo Kim <61856979+33modeling@users.noreply.github.com>
# @bug No known bugs.
"""Run locally with python3 -m unittest discover -s .github/workflows/static.check.scripts."""

from pathlib import Path, PureWindowsPath
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ssat_fixtures import check


class FixtureChecks(unittest.TestCase):
    """Use real temporary files and assert both acceptance and rejection."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.models = self.root / "tests/test_models/models"
        self.models.mkdir(parents=True)
        self.suite = self.root / "tests/optional_backend"
        self.suite.mkdir()
        (self.models / "model.onnx").touch()

    def inspect(self, source):
        """Write a suite with no backend and inspect it without sourcing it."""
        (self.suite / "runTest.sh").write_text(source, encoding="utf-8")
        return check(self.root)

    def test_literal_alias_and_comment_p(self):
        checked, _, errors = self.inspect(
            'BASE="../test_models/models"\n'
            'MODEL="${BASE}/model.onnx" # input fixture\n'
            '# MODEL="../test_models/models/absent.onnx"\n')
        self.assertEqual((checked, errors), (2, []))

    def test_explicit_absent_fixture_p(self):
        checked, _, errors = self.inspect(
            '# ssat-fixture: missing - negative model-open case\n'
            'MODEL="../test_models/models/absent.onnx"\n')
        self.assertEqual((checked, errors), (1, []))

    def test_prefixed_fixture_alias_p(self):
        for prefix in ("./", "././", "././/", "../optional_backend/", "../optional_backend//"):
            with self.subTest(prefix=prefix):
                checked, _, errors = self.inspect(
                    'BASE="../test_models/models"\n'
                    'MODEL="' + prefix + '${BASE}/model.onnx"\n')
                self.assertEqual((checked, errors), (2, []))

    def test_missing_prefixed_fixture_alias_n(self):
        for prefix in ("./", "././", "././/", "../optional_backend/", "../optional_backend//"):
            with self.subTest(prefix=prefix):
                _, _, errors = self.inspect(
                    'BASE="../test_models/models"\n'
                    'MODEL="' + prefix + '${BASE}/absent.onnx"\n')
                self.assertEqual(len(errors), 1)
                self.assertIn("fixture does not exist", errors[0])

    def test_command_environment_and_pipeline_are_not_paths_p(self):
        checked, _, errors = self.inspect(
            'MODEL="../test_models/models/model.onnx"\n'
            'PIPELINE="tensor_filter model=${MODEL} ! fakesink"\n'
            'PIPELINE2="${PIPELINE}"\n'
            'GST_DEBUG=2 GST_DEBUG_NO_COLOR=1 gstTest "${PIPELINE2}"\n')
        self.assertEqual((checked, errors), (1, []))

    def test_long_non_fixture_assignment_n(self):
        """Bound a pathological non-match without hanging the test runner."""
        (self.suite / "runTest.sh").write_text(
            'VALUE="' + '!/' * 5000 + 'x"\n', encoding="utf-8")
        checker = Path(__file__).with_name("ssat_fixtures.py")
        result = subprocess.run(
            [sys.executable, str(checker), str(self.root)], check=False,
            capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Checked 0 fixture references in 1 SSAT suites.", result.stdout)

    def test_alias_can_use_previous_binding_p(self):
        checked, _, errors = self.inspect(
            'MODEL="../test_models/models"\n'
            'MODEL="${MODEL}/model.onnx"\n')
        self.assertEqual((checked, errors), (2, []))

    def test_all_architecture_variants_p(self):
        for arch in ["arm", "x86"]:
            (self.models / (arch + ".so_")).touch()
        checked, _, errors = self.inspect(
            '# ssat-fixture: variants ARCH=arm,x86 - prebuilt models\n'
            'MODEL="../test_models/models/${ARCH}.so_"\n')
        self.assertEqual((checked, errors), (2, []))

    def test_quoted_filename_with_spaces_and_hash_p(self):
        (self.models / "a model#1.onnx").touch()
        checked, _, errors = self.inspect(
            'MODEL="../test_models/models/a model#1.onnx" # fixture\n')
        self.assertEqual((checked, errors), (1, []))

    def test_unquoted_hash_filename_p(self):
        (self.models / "model.onnx").unlink()
        (self.models / "model.onnx#1").touch()
        checked, _, errors = self.inspect(
            'MODEL=../test_models/models/model.onnx#1 # trailing comment\n')
        self.assertEqual((checked, errors), (1, []))

    def test_missing_unquoted_hash_filename_n(self):
        _, _, errors = self.inspect(
            'MODEL=../test_models/models/model.onnx#missing # comment\n')
        self.assertEqual(len(errors), 1)
        self.assertIn("model.onnx#missing", errors[0])

    def test_hash_in_unquoted_alias_binding_p(self):
        (self.models / "#model.onnx").touch()
        checked, _, errors = self.inspect(
            'NAME=#model.onnx # name includes a literal hash\n'
            'MODEL=../test_models/models/${NAME}\n')
        self.assertEqual((checked, errors), (1, []))

    def test_missing_hash_in_unquoted_alias_binding_n(self):
        _, _, errors = self.inspect(
            'NAME=#missing.onnx # literal hash\n'
            'MODEL=../test_models/models/${NAME}\n')
        self.assertEqual(len(errors), 1)
        self.assertIn("#missing.onnx", errors[0])

    def test_hash_after_quoted_segment_p(self):
        (self.models / "model.onnx").unlink()
        (self.models / "model.onnx#1").touch()
        checked, _, errors = self.inspect(
            'MODEL="../test_models/models/model.onnx"#1 # comment\n')
        self.assertEqual((checked, errors), (1, []))

    def test_no_shell_commands_are_executed_p(self):
        sentinel = self.root / "executed"
        _, _, errors = self.inspect(
            'IGNORED="$(touch {})"\n'
            'MODEL="../test_models/models/model.onnx"\n'.format(sentinel))
        self.assertFalse(sentinel.exists())
        self.assertEqual(errors, [])

    def test_nested_suite_fixture_p(self):
        self.inspect("# Suite without model fixtures.\n")
        self.suite = self.root / "tests/nnstreamer_edge/edge"
        self.suite.mkdir(parents=True)
        checked, suites, errors = self.inspect(
            'MODEL="../../test_models/models/model.onnx"\n')
        self.assertEqual((checked, suites, errors), (1, 2, []))

    def test_nested_suite_missing_fixture_n(self):
        self.inspect("# Suite without model fixtures.\n")
        self.suite = self.root / "tests/nnstreamer_edge/query"
        self.suite.mkdir(parents=True)
        _, _, errors = self.inspect(
            'MODEL="../../test_models/models/absent.onnx"\n')
        self.assertEqual(len(errors), 1)
        self.assertIn("tests/nnstreamer_edge/query/runTest.sh:1:", errors[0])
        self.assertIn("fixture does not exist", errors[0])

    def test_windows_diagnostic_path_n(self):
        self.inspect('# Suite without model fixtures.\n')
        self.suite = self.root / "tests/nnstreamer_edge/query"
        self.suite.mkdir(parents=True)
        (self.suite / "runTest.sh").write_text(
            'MODEL="../../test_models/models/absent.onnx"\n', encoding="utf-8")
        original_relative_to = Path.relative_to

        def relative_to(path, *args, **kwargs):
            """Keep fixture resolution native but exercise a Windows suite path."""
            relative = original_relative_to(path, *args, **kwargs)
            return PureWindowsPath(relative) if args == (self.root,) else relative

        with patch.object(Path, "relative_to", relative_to):
            _, _, errors = check(self.root)
        self.assertEqual(len(errors), 1)
        self.assertIn("tests/nnstreamer_edge/query/runTest.sh:1:", errors[0])
        self.assertIn("fixture does not exist", errors[0])

    def test_missing_model_before_backend_probe_n(self):
        _, _, errors = self.inspect(
            'if [ ! -f backend.so ]; then exit; fi\n'
            'MODEL="../test_models/models/yolov5nu_224.onnx"\n')
        self.assertIn("fixture does not exist", errors[0])

    def test_missing_one_architecture_variant_n(self):
        (self.models / "arm.so_").touch()
        _, _, errors = self.inspect(
            '# ssat-fixture: variants ARCH=arm,x86 - prebuilt models\n'
            'MODEL="../test_models/models/${ARCH}.so_"\n')
        self.assertIn("x86.so_", errors[0])

    def test_absent_fixture_must_remain_absent_n(self):
        _, _, errors = self.inspect(
            '# ssat-fixture: missing - negative model-open case\n'
            'MODEL="../test_models/models/model.onnx"\n')
        self.assertIn("unexpectedly exists", errors[0])

    def test_unresolved_variant_alias_cannot_bypass_check_n(self):
        for arch in ["arm", "x86"]:
            (self.models / (arch + ".so_")).touch()
        _, _, errors = self.inspect(
            '# ssat-fixture: variants ARCH=arm,x86 - prebuilt models\n'
            'MODEL="../test_models/models/${ARCH}.so_"\n'
            'BROKEN="${MODEL}.missing"\n')
        self.assertIn("unknown fixture variable MODEL", errors[0])

    def test_unknown_variable_n(self):
        _, _, errors = self.inspect('MODEL="../test_models/${UNKNOWN}/model.onnx"\n')
        self.assertIn("unknown fixture variable", errors[0])

    def test_no_command_substitution_in_fixtures_n(self):
        sentinel = self.root / "executed"
        _, _, errors = self.inspect(
            'MODEL="../test_models/$(touch {})/model.onnx"\n'.format(sentinel))
        self.assertFalse(sentinel.exists())
        self.assertIn("not shell commands", errors[0])

    def test_directory_escape_n(self):
        _, _, errors = self.inspect('MODEL="../test_models/../../outside.onnx"\n')
        self.assertIn("escapes", errors[0])

    def test_inline_paths_must_be_declared_n(self):
        _, _, errors = self.inspect('gstTest "filesrc location=../test_models/models/model.onnx"\n')
        self.assertIn("simple variable assignments", errors[0])

    def test_annotation_requires_reason_n(self):
        _, _, errors = self.inspect(
            '# ssat-fixture: missing\nMODEL="../test_models/models/absent.onnx"\n')
        self.assertIn("needs", errors[0])


if __name__ == "__main__":
    unittest.main()
