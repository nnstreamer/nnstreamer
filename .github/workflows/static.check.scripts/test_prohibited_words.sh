#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file     test_prohibited_words.sh
# @brief    Self-test for prohibited-words.sh and its word list.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Runs prohibited-words.sh the same way static.check.yml does, from the
# repository root with a changed-file list. Two properties are pinned: a
# prohibited word must fail the checker, and ordinary prose must not. The
# first would have caught the checker silently passing on every input; the
# second keeps an over-broad entry from turning the word list into a gate
# that rejects nearly every PR.
#
# The checker is invoked with stdin closed, matching how GitHub runs it, so
# that a future reader-of-stdin bug fails here instead of hanging.

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
CHECKER="${SCRIPT_DIR}/prohibited-words.sh"
WORD_LIST="${SCRIPT_DIR}/prohibited-words.txt"
workdir=$(mktemp -d)
failed=0

trap 'rm -rf "$workdir"' EXIT

# expect_checker <expected 0|1> <description> <content>
# Writes content to a markdown file, runs the checker on it from the repo
# root, and compares the exit status.
expect_checker() {
  local expected=$1 desc=$2 content=$3
  local target list actual

  target="${workdir}/$(echo "$desc" | tr -c 'a-zA-Z0-9' '_').md"
  printf '%s\n' "$content" > "$target"
  list=$(mktemp -p "$workdir")
  printf '%s\n' "$target" > "$list"

  (cd "$REPO_ROOT" && bash "$CHECKER" "$list") > /dev/null 2>&1 < /dev/null
  actual=$?
  [[ $actual -ne 0 ]] && actual=1

  if [[ "$actual" == "$expected" ]]; then
    echo "PASS: ${desc} (exit ${actual})"
  else
    echo "FAIL: ${desc} expected exit ${expected}, got ${actual}"
    failed=1
  fi
}

# Every listed word must actually fail the checker. The || [[ -n ]] guard keeps
# a last line without a trailing newline from being skipped silently.
while read -r word || [[ -n "$word" ]]; do
  word=${word%$'\r'}
  [[ -z "$word" ]] && continue
  expect_checker 1 "prohibited word ${word}" "A line that mentions ${word} inline."
done < "$WORD_LIST"

# Ordinary prose must pass. These sentences use the vocabulary that a
# repository-wide word list is most likely to over-match on.
expect_checker 0 "plain prose" "This file describes how to build and test the plugin."
expect_checker 0 "code-ish prose" "Open the file, read the header, then close the file descriptor."

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
  echo "::error test_prohibited_words.sh failed."
  exit 1
fi
echo "test_prohibited_words.sh: all checks passed."
