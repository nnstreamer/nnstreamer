#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-only
# @file test_valgrind_filters.py
# @brief Exercise stale and malformed Valgrind filters without optional backends.
"""Regression checks for stale and malformed Valgrind GTest filters."""

from pathlib import Path
import tempfile
import unittest

import valgrind_filters as checker


class FilterChecks(unittest.TestCase):
    """Use small source trees so no optional framework is needed."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.workflow = self.root / checker.WORKFLOW
        self.workflow.parent.mkdir(parents=True)
        self.workflow.write_text(
            'skip_cases="Slow.first"\n'
            'skip_cases="${skip_cases}:Slow.second"\n'
            'llamacpp_cases="Llama.invalid_n"\n', encoding="utf-8"
        )
        self.source = self.root / "tests/core/unittest_core.cc"
        self.source.parent.mkdir(parents=True)
        self.source.write_text(
            "TEST (Slow, first) {}\nTEST_F (Slow, second) {}\n", encoding="utf-8"
        )
        self.llama = self.root / checker.LLAMACPP
        self.llama.parent.mkdir(parents=True)
        self.llama.write_text("TEST_F (Llama, invalid_n) {}\n", encoding="utf-8")

    def test_literal_and_appended_names_p(self):
        self.assertEqual(checker.check(self.root), 3)

    def test_multiline_definitions_p(self):
        self.source.write_text(
            "TEST\n(Slow,\nfirst) {}\nGTEST_TEST (Slow, second) {}\n",
            encoding="utf-8",
        )
        self.assertEqual(checker.check(self.root), 3)

    def test_comments_between_macro_arguments_p(self):
        self.source.write_text(
            "TEST (Slow, /* explanation */ first) {}\nTEST_F (Slow, second) {}\n",
            encoding="utf-8",
        )
        self.assertEqual(checker.check(self.root), 3)

    def test_renamed_exclusion_n(self):
        self.source.write_text("TEST (Slow, renamed) {}\nTEST (Slow, second) {}\n")
        with self.assertRaisesRegex(ValueError, "Slow.first"):
            checker.check(self.root)

    def test_renamed_inclusion_n(self):
        self.llama.write_text("TEST_F (Llama, renamed_n) {}\n")
        with self.assertRaisesRegex(ValueError, "Llama.invalid_n"):
            checker.check(self.root)

    def test_comment_or_string_is_not_a_test_n(self):
        for text in ["// TEST_F (Llama, invalid_n) {}", '"TEST_F (Llama, invalid_n)"']:
            with self.subTest(source=text):
                self.llama.write_text(text)
                with self.assertRaisesRegex(ValueError, "Llama.invalid_n"):
                    checker.check(self.root)

    def test_llama_case_in_another_binary_n(self):
        self.llama.write_text("// removed\n")
        with self.source.open("a") as source:
            source.write("TEST_F (Llama, invalid_n) {}\n")
        with self.assertRaisesRegex(ValueError, "llamacpp_cases"):
            checker.check(self.root)

    def test_parameterized_definition_does_not_match_bare_filter_n(self):
        self.llama.write_text("TEST_P (Llama, invalid_n) {}\n")
        with self.assertRaisesRegex(ValueError, "Llama.invalid_n"):
            checker.check(self.root)

    def test_missing_list_n(self):
        self.workflow.write_text('skip_cases="Slow.first"\n')
        with self.assertRaisesRegex(ValueError, "llamacpp_cases"):
            checker.check(self.root)

    def test_malformed_lists_n(self):
        for rhs in ['""', '"Slow.*"', '"${other}"', '$(printf unsafe)',
                    '"${skip_cases}:Slow.first"', '"Slow.first:Slow.first"']:
            with self.subTest(rhs=rhs):
                self.workflow.write_text(
                    f'skip_cases={rhs}\nllamacpp_cases="Llama.invalid_n"\n'
                )
                with self.assertRaises(ValueError):
                    checker.check(self.root)


if __name__ == "__main__":
    unittest.main()
