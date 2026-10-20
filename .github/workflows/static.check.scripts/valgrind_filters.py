#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-only
# @file valgrind_filters.py
# @brief Validate the exact GTest names selected by the Valgrind workflow.
"""Check the exact GTest names selected by the Valgrind workflow."""

import argparse
from pathlib import Path
import re
import sys


WORKFLOW = ".github/workflows/ubuntu_clean_meson_build.yml"
LLAMACPP = "tests/nnstreamer_filter_llamacpp/unittest_filter_llamacpp.cc"
FILTERS = ("skip_cases", "llamacpp_cases")


def read_filters(workflow):
    """Read the literal lists without evaluating shell from a pull request."""
    result = {name: [] for name in FILTERS}
    for line in workflow.splitlines():
        assignment = re.match(r"\s*(skip_cases|llamacpp_cases)\s*=(.*)$", line)
        if not assignment:
            continue
        name, rhs = assignment.groups()
        quoted = re.fullmatch(r'\s*"([^"\n]*)"\s*(?:#.*)?', rhs)
        if not quoted:
            raise ValueError(f"{name}: expected a double-quoted literal list")
        value = quoted.group(1)
        prefix = "${" + name + "}:"
        if value.startswith(prefix):
            if not result[name]:
                raise ValueError(f"{name}: append precedes initialization")
            value = value[len(prefix):]
        elif result[name]:
            raise ValueError(f"{name}: list is assigned more than once")
        for case in value.split(":"):
            if not re.fullmatch(r"[A-Za-z_]\w*\.[A-Za-z_]\w*", case):
                raise ValueError(f"{name}: expected an exact Suite.Case, got {case!r}")
            if case in result[name]:
                raise ValueError(f"{name}: duplicate test {case}")
            result[name].append(case)
    for name, cases in result.items():
        if not cases:
            raise ValueError(f"{name}: no test names found in {WORKFLOW}")
    return result


def active_test_source(source):
    """Remove literal #if 0/1 branches; leave unknown configurations possible."""
    stack, lines, active = [], [], True
    for line in source.splitlines():
        directive = re.match(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$", line)
        if not directive:
            if active:
                lines.append(line)
            continue
        name, expression = directive.groups()
        expression = expression.strip()
        while expression.startswith("(") and expression.endswith(")"):
            expression = expression[1:-1].strip()
        condition = {"0": False, "1": True}.get(expression)
        if name in ("if", "ifdef", "ifndef"):
            if name != "if":
                condition = None
            stack.append([active, condition is not True])
            active = active and condition is not False
        elif stack:
            parent, remaining = stack[-1]
            if name == "endif":
                stack.pop()
                active = parent
            else:
                active = parent and remaining and (name == "else" or condition is not False)
                stack[-1][1] = remaining and name != "else" and condition is not True
    return "\n".join(lines)


def test_names(source):
    """Find non-parameterized tests, ignoring literals and inactive branches."""
    tokens = re.compile(
        r'//[^\n]*|/\*.*?\*/|'
        r'(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n\f\v]{0,16})\(.*?\)(?P=delimiter)"|'
        r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        re.DOTALL,
    )
    source = tokens.sub(lambda match: " " + "\n" * match[0].count("\n"), source)
    source = active_test_source(source)
    # TEST_P names include their instantiation and parameter suffix at runtime;
    # a bare Suite.Case filter would not select those tests.
    return {
        f"{suite}.{case}"
        for suite, case in re.findall(
            r"\b(?:TEST|TEST_F|GTEST_TEST)\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)",
            source,
        )
    }


def check(root):
    """Fail if a listed case was removed, renamed, or belongs to another binary."""
    filters = read_filters((root / WORKFLOW).read_text(encoding="utf-8"))
    sources = list((root / "tests").rglob("*.cc"))
    sources += list((root / "tests").rglob("*.cpp"))
    known = set()
    for path in sources:
        known.update(test_names(path.read_text(encoding="utf-8")))
    llama_names = test_names((root / LLAMACPP).read_text(encoding="utf-8"))
    for name, cases in filters.items():
        available = llama_names if name == "llamacpp_cases" else known
        missing = sorted(set(cases) - available)
        if missing:
            raise ValueError(f"{name}: no matching test definition for {', '.join(missing)}")
    return sum(len(cases) for cases in filters.values())


def main():
    """Run without building optional inference backends or downloading models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        count = check(args.root)
    except (OSError, ValueError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    print(f"All {count} exact Valgrind filter names have matching test definitions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
