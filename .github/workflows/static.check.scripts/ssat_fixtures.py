#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-only
##
# @file ssat_fixtures.py
# @brief Check repository fixture assignments without running the SSAT scripts.
# @package ssat_fixtures
# @author Minsoo Kim <61856979+33modeling@users.noreply.github.com>
# @bug No known bugs.
"""Check constant fixture paths even when an optional subplugin is not built.

Only simple shell assignments and aliases are interpreted, never shell commands.
An intentionally absent path needs a preceding `ssat-fixture: missing - reason`.
Finite architecture choices use `ssat-fixture: variants NAME=a,b - reason`.
Inline fixture literals must first be assigned to a variable, as existing suites
already do. This is a path-existence check, not a shell or model validator.
"""

import argparse
from pathlib import Path
import re
import shlex
import sys


ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_]\w*)=(.*)$")
VARIABLE = re.compile(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))")
VARIANTS = re.compile(r"variants ([A-Za-z_]\w*)=([A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*)$")


def annotation(text):
    """Require a reason and a bounded annotation rather than a blanket skip."""
    directive, separator, reason = text.partition(" - ")
    if not separator or not reason.strip():
        raise ValueError("ssat-fixture annotation needs ' - reason'")
    if directive == "missing":
        return "missing", None, [None]
    match = VARIANTS.fullmatch(directive)
    if not match:
        raise ValueError("unknown ssat-fixture annotation")
    choices = match[2].split(",")
    if len(choices) != len(set(choices)):
        raise ValueError("duplicate fixture variant")
    return "variants", match[1], choices


def fixture_values(raw, bindings, note):
    """Resolve literal assignments and known aliases, without shell evaluation."""
    words = shlex.split(raw, comments=True)
    if len(words) != 1 or "$(" in words[0] or "`" in words[0]:
        raise ValueError("fixture requires a constant assignment, not shell commands")
    value = words[0]
    if raw.lstrip().startswith("'"):
        if "$" in value:
            raise ValueError("single-quoted fixture variables are literal, not aliases")
    kind, variable, choices = note or ("present", None, [None])
    if variable and variable not in [m[1] or m[2] for m in VARIABLE.finditer(value)]:
        raise ValueError("fixture variant does not occur in the assignment")
    results = []
    for choice in choices:
        def replace(match):
            name = match[1] or match[2]
            if name == variable:
                return choice
            if name not in bindings:
                raise ValueError("unknown fixture variable " + name)
            return bindings[name]
        resolved = VARIABLE.sub(replace, value)
        if any(char in resolved for char in "$`*?[]\\"):
            raise ValueError("unsupported fixture expansion: " + value)
        results.append((kind, resolved))
    return results


def check_script(script, fixture_root):
    """Return checked references and diagnostics for one runTest.sh."""
    bindings, errors, checked = {}, [], 0
    fixture_bindings = set()
    pending = None
    for number, line in enumerate(script.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if stripped.startswith("# ssat-fixture:"):
                if pending:
                    errors.append((number, "unused ssat-fixture annotation"))
                try:
                    pending = annotation(stripped.split(":", 1)[1].strip())
                except ValueError as exc:
                    errors.append((number, str(exc)))
            continue
        note, pending = pending, None
        match = ASSIGNMENT.match(line)
        if not match:
            if note or "test_models/" in line:
                errors.append((number, "declare fixture paths in simple variable assignments"))
            continue
        name, raw = match.groups()
        try:
            words = shlex.split(raw, comments=True)
        except ValueError as exc:
            if note or "test_models" in raw:
                errors.append((number, str(exc)))
            bindings.pop(name, None)
            fixture_bindings.discard(name)
            continue
        if not words:
            bindings[name] = ""
            fixture_bindings.discard(name)
            if note:
                errors.append((number, "ssat-fixture annotation is not on a fixture"))
            continue
        # Environment assignments before a command are not persistent bindings.
        if len(words) != 1:
            if note or "test_models" in raw:
                errors.append((number, "fixture requires a constant assignment"))
            continue
        referenced = [m[1] or m[2] for m in VARIABLE.finditer(raw)]
        # The class spans slashes already, so one optional prefix is sufficient.
        is_fixture = "test_models" in words[0] or (
            re.match(r"^(?:[^\s$`|&;=<>]+/)?\$", words[0]) and any(
                var in fixture_bindings for var in referenced))
        try:
            values = fixture_values(raw, bindings, note)
            if not is_fixture:
                if note:
                    raise ValueError("ssat-fixture annotation is not on a fixture")
                bindings[name] = values[0][1]
                fixture_bindings.discard(name)
                continue
            for kind, value in values:
                target = (script.parent / value).resolve()
                try:
                    target.relative_to(fixture_root)
                except ValueError as exc:
                    raise ValueError("fixture path escapes tests/test_models: " + value) from exc
                if kind == "missing":
                    if target.exists():
                        raise ValueError("negative fixture unexpectedly exists: " + value)
                elif not target.exists():
                    raise ValueError("fixture does not exist: " + value)
                checked += 1
            if len(values) == 1:
                bindings[name] = values[0][1]
                fixture_bindings.add(name)
            else:
                bindings.pop(name, None)
                # A later alias must not silently lose its fixture provenance.
                fixture_bindings.add(name)
        except ValueError as exc:
            bindings.pop(name, None)
            fixture_bindings.discard(name)
            if is_fixture or note:
                errors.append((number, str(exc)))
    if pending:
        errors.append((number, "unused ssat-fixture annotation"))
    return checked, errors


def check(root):
    """Inspect every suite independently of its optional backend probes."""
    fixture_root = (root / "tests/test_models").resolve()
    scripts = sorted((root / "tests").rglob("runTest.sh"))
    if not scripts or not fixture_root.is_dir():
        raise ValueError("run from a source tree with tests/test_models and SSAT suites")
    checked, errors = 0, []
    for script in scripts:
        count, diagnostics = check_script(script, fixture_root)
        checked += count
        errors.extend("{}:{}: {}".format(script.relative_to(root).as_posix(), line, error)
                      for line, error in diagnostics)
    return checked, len(scripts), errors


def main():
    """Fail CI on missing or unresolvable repository fixture paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        checked, suites, errors = check(args.root.resolve())
    except (OSError, ValueError) as exc:
        print("::error::" + str(exc), file=sys.stderr)
        return 1
    for error in errors:
        print("::error::" + error, file=sys.stderr)
    if errors:
        return 1
    print("Checked {} fixture references in {} SSAT suites.".format(checked, suites))
    return 0


if __name__ == "__main__":
    sys.exit(main())
