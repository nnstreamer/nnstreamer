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

    def test_raw_string_does_not_hide_following_test_p(self):
        for prefix in ("", "u8", "u", "U", "L"):
            with self.subTest(prefix=prefix):
                self.llama.write_text(
                    'static const auto text = ' + prefix + 'R"example(a " quote\n#if 0\n)example";\n'
                    'TEST_F (Llama, invalid_n) {}\n')
                self.assertEqual(checker.check(self.root), 3)

    def test_raw_string_cannot_supply_a_test_n(self):
        for delimiter in ("", "example", "sixteen_chars_ok"):
            with self.subTest(delimiter=delimiter):
                self.llama.write_text(
                    'static const auto text = R"' + delimiter +
                    '("TEST_F (Llama, invalid_n) {}\n#if 0\n)' + delimiter + '";\n')
                with self.assertRaisesRegex(ValueError, "Llama.invalid_n"):
                    checker.check(self.root)

    def test_unconditionally_inactive_test_n(self):
        for directive in ("0", "(0)"):
            with self.subTest(directive=directive):
                self.llama.write_text(
                    '#if ' + directive + '\n#if OPTIONAL\n'
                    'TEST_F (Llama, invalid_n) {}\n#endif\n#endif\n')
                with self.assertRaisesRegex(ValueError, "Llama.invalid_n"):
                    checker.check(self.root)

    def test_active_alternative_and_unknown_configuration_p(self):
        for source in (
                '#if 0\n#else\nTEST_F (Llama, invalid_n) {}\n#endif\n',
                '#if 0\n#elif 1\nTEST_F (Llama, invalid_n) {}\n#endif\n',
                '#ifdef OPTIONAL\nTEST_F (Llama, invalid_n) {}\n#endif\n',
                '#ifndef OPTIONAL\nTEST_F (Llama, invalid_n) {}\n#endif\n',
                '#if 0\n#elif OPTIONAL\nTEST_F (Llama, invalid_n) {}\n#endif\n'):
            with self.subTest(source=source):
                self.llama.write_text(source)
                self.assertEqual(checker.check(self.root), 3)

    def test_inactive_alternative_cannot_supply_a_test_n(self):
        for source in (
                '#if 1\n#else\nTEST_F (Llama, invalid_n) {}\n#endif\n',
                '#if 1\n#elif OPTIONAL\nTEST_F (Llama, invalid_n) {}\n#endif\n',
                '#if OPTIONAL\n#elif 0\nTEST_F (Llama, invalid_n) {}\n#endif\n'):
            with self.subTest(source=source):
                self.llama.write_text(source)
                with self.assertRaisesRegex(ValueError, "Llama.invalid_n"):
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
