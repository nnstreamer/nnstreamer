#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file     test_pylint.sh
# @brief    Self-test for pylint.sh.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Runs pylint.sh the same way static.check.yml does, from the repository root
# with a changed-file list. Two properties are pinned: a pylint error must
# fail the checker, and convention, refactor, warning and import-error
# messages must not. The first would have caught the checker silently passing
# on every input (#4919); the second keeps it from rejecting nearly every
# Python file in the tree, or every file that imports a module CI does not
# install.

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
CHECKER="${SCRIPT_DIR}/pylint.sh"
workdir=$(mktemp -d)
output="${workdir}/checker.log"
failed=0

trap 'rm -rf "$workdir"' EXIT

# expect_checker <expected 0|1> <description> <file>...
# Runs the checker on the given files from the repo root and compares the
# exit status. The checker output is kept in $output.
expect_checker() {
  local expected=$1 desc=$2
  shift 2
  local list actual

  list=$(mktemp -p "$workdir")
  printf '%s\n' "$@" > "$list"

  (cd "$REPO_ROOT" && bash "$CHECKER" "$list") > "$output" 2>&1 < /dev/null
  actual=$?
  [[ $actual -ne 0 ]] && actual=1

  if [[ "$actual" == "$expected" ]]; then
    echo "PASS: ${desc} (exit ${actual})"
  else
    echo "FAIL: ${desc} expected exit ${expected}, got ${actual}"
    sed 's/^/  | /' "$output"
    failed=1
  fi
}

# expect_log <pattern> <description>
# Checks that the last checker output matches the extended regex, so that a
# failing case cannot pass on an unrelated error such as a missing pylint.
expect_log() {
  if grep -qE "$1" "$output"; then
    echo "PASS: ${2}"
  else
    echo "FAIL: ${2}: no match for '${1}' in the checker log"
    failed=1
  fi
}

cat > "${workdir}/bare_raise.py" << 'EOF'
"""A script that re-raises with no active exception."""


def write(ok):
    """Mirror the bare raise that passThrough_CV.py had."""
    try:
        if not ok:
            raise
    except RuntimeError:
        raise
EOF

cat > "${workdir}/syntax_error.py" << 'EOF'
def broken(:
    pass
EOF

cat > "${workdir}/style_only.py" << 'EOF'
import os


class Old(object):
    pass


def camelCaseName(type):
    return type
EOF

cat > "${workdir}/missing_import.py" << 'EOF'
"""A script importing a module that is not installed."""
import nnstreamer_pylint_selftest_missing_module as missing

print(missing.VALUE)
EOF

cp "${workdir}/syntax_error.py" "${workdir}/not_python.txt"

expect_checker 1 "bare raise outside an except clause" "${workdir}/bare_raise.py"
expect_log "bare_raise\.py:[0-9]+:[0-9]+: E0704:" "bare raise is reported as E0704"
expect_checker 1 "syntax error" "${workdir}/syntax_error.py"
expect_log "syntax_error\.py:[0-9]+:[0-9]+: E0001:" "syntax error is reported as E0001"
expect_checker 1 "error in the second file of the list" \
  "${workdir}/style_only.py" "${workdir}/bare_raise.py"
expect_log "Checking .*/style_only\.py" "the first file of the list is checked"
expect_log "pylint error from .*/bare_raise\.py" "the error is attributed to the second file"
expect_checker 0 "convention, refactor and warning messages only" "${workdir}/style_only.py"
expect_checker 0 "import of a module that is not installed" "${workdir}/missing_import.py"
expect_checker 0 "non-Python file" "${workdir}/not_python.txt"

# An empty changed-file list is a no-op, not a failure.
empty_list=$(mktemp -p "$workdir")
: > "$empty_list"
if (cd "$REPO_ROOT" && bash "$CHECKER" "$empty_list") > /dev/null 2>&1 < /dev/null; then
  echo "PASS: empty file list"
else
  echo "FAIL: empty file list expected exit 0"
  failed=1
fi

if [[ "$failed" != "0" ]]; then
  echo "::error test_pylint.sh failed."
  exit 1
fi
echo "test_pylint.sh: all checks passed."
