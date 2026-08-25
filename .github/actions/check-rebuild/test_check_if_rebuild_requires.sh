#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file: test_check_if_rebuild_requires.sh
# @brief    Self-test for check_if_rebuild_requires.sh and the action.yml step pattern.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Runs the same "out=$(...) || exit; grep -c ... || true" pattern as
# .github/actions/check-rebuild/action.yml under "sh -e", which is how GitHub
# executes composite "shell: sh" steps. This keeps the REBUILD=NO path and the
# fail-closed error path exercised by CI on every PR.

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CHECKER="${SCRIPT_DIR}/check_if_rebuild_requires.sh"
failed=0

# run_step <file-list-path> <mode>: emulate the action.yml step under sh -e.
run_step() {
  sh -ec '
    out=$(bash "$1" "$2" "$3") || {
      echo "::error::check_if_rebuild_requires.sh failed" >&2
      exit 1
    }
    printf "%s\n" "$out" | grep -c "REBUILD=YES" || true
  ' sh "$CHECKER" "$1" "$2"
}

# expect_rebuild <expected 0|1> <mode> <changed-file>...
expect_rebuild() {
  expected=$1
  mode=$2
  shift 2
  list=$(mktemp)
  printf '%s\n' "$@" > "$list"
  if actual=$(run_step "$list" "$mode") && [ "$actual" = "$expected" ]; then
    echo "PASS: mode=${mode} files=[$*] rebuild=${actual}"
  else
    echo "FAIL: mode=${mode} files=[$*] expected=${expected} got=${actual:-<step failed>}"
    failed=1
  fi
  rm -f "$list"
}

# A source change triggers a rebuild in every mode.
for mode in build gbs debian android; do
  expect_rebuild 1 "$mode" "gst/nnstreamer/nnstreamer_plugin_api_impl.c"
done

# Docs-only changes must skip every mode (this is the REBUILD=NO / "|| true"
# path that lets docs-only PRs pass all 8 consuming workflows).
for mode in build gbs debian android; do
  expect_rebuild 0 "$mode" "README.md" "docs/logo.png"
done

# Mode-specific arms.
expect_rebuild 1 gbs "packaging/nnstreamer.spec"
expect_rebuild 0 build "packaging/nnstreamer.spec"
expect_rebuild 1 debian "debian/rules"
expect_rebuild 0 android "debian/rules"
expect_rebuild 1 android "jni/Android.mk"
expect_rebuild 0 gbs "jni/Android.mk"

# A checker failure must fail the step instead of silently yielding rebuild=0.
if run_step "/nonexistent/file-list" build > /dev/null 2>&1; then
  echo "FAIL: a missing file list must fail the step"
  failed=1
else
  echo "PASS: a missing file list fails the step"
fi

exit ${failed}
