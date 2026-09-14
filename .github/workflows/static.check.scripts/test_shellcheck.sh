#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file     test_shellcheck.sh
# @brief    Self-test for shellcheck.sh.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Runs shellcheck.sh the same way static.check.yml does, from the repository
# root with a changed-file list. Two properties are pinned: a finding of
# error severity must fail the checker, and warning, info and style
# findings must not. The first would have caught the checker passing every
# file because the linter never received it (#4965); the second keeps it
# from rejecting most shell scripts in the tree.

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
CHECKER="${SCRIPT_DIR}/shellcheck.sh"
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
# case cannot pass on an unrelated error such as a missing shellcheck.
expect_log() {
  if grep -qE "$1" "$output"; then
    echo "PASS: ${2}"
  else
    echo "FAIL: ${2}: no match for '${1}' in the checker log"
    failed=1
  fi
}

cat > "${workdir}/exit_negative.sh" << 'EOF'
#!/bin/bash
if [ "$#" -ne 1 ]; then
  exit -1
fi
EOF

cat > "${workdir}/warnings_only.sh" << 'EOF'
#!/bin/bash
unused=1
now=`date`
echo $1 "$now"
EOF

cp "${workdir}/exit_negative.sh" "${workdir}/not_shell.txt"
tail -n +2 "${workdir}/exit_negative.sh" > "${workdir}/no_shebang.sh"

# Without a shebang, file(1) reports plain text instead of a shell script.
if file "${workdir}/no_shebang.sh" | grep -q "shell script"; then
  echo "FAIL: no_shebang.sh is still classified as a shell script"
  failed=1
fi

# The warning-only fixture must keep a warning, or its case proves nothing.
if shellcheck -s bash -S warning "${workdir}/warnings_only.sh" > /dev/null 2>&1; then
  echo "FAIL: warnings_only.sh has no warning-level finding"
  failed=1
fi

expect_checker 1 "error-level finding" "${workdir}/exit_negative.sh"
expect_log "SC2242 \(error\)" "the finding is shown in the log"
expect_checker 1 "error in the second file of the list" \
  "${workdir}/warnings_only.sh" "${workdir}/exit_negative.sh"
expect_log "passed\. file name: .*/warnings_only\.sh" "the first file of the list is checked"
expect_log "failed\. file name: .*/exit_negative\.sh" "the error is attributed to the second file"
expect_checker 1 "error in a script without a shebang" "${workdir}/no_shebang.sh"
expect_log "failed\. file name: .*/no_shebang\.sh" "the script without a shebang is checked"
expect_checker 0 "warning, info and style findings only" "${workdir}/warnings_only.sh"
expect_log "passed\. file name: .*/warnings_only\.sh" "the warning-only file is checked"
expect_checker 0 "non-shell file" "${workdir}/not_shell.txt"

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
  echo "::error test_shellcheck.sh failed."
  exit 1
fi
echo "test_shellcheck.sh: all checks passed."
