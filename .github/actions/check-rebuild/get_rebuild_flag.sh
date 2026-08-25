#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file: get_rebuild_flag.sh
# @brief    Print 1 if the given changed-file list requires a rebuild, else 0.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Argument 1 ($1): the file containing the list of changed files.
# Argument 2 ($2): build mode to be checked (see check_if_rebuild_requires.sh).
#
# Exits nonzero if the checker fails, so that the calling CI step (run under
# "sh -e") turns red instead of silently skipping the job.
# "grep -c" is used instead of "grep | wc -l": the BSD wc on macOS pads its
# output with leading spaces, which made the `env.rebuild == '1'` step
# conditions always false there (macOS jobs silently skipped every build step).

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

out=$(bash "${SCRIPT_DIR}/check_if_rebuild_requires.sh" "$1" "$2") || {
  echo "::error::check_if_rebuild_requires.sh failed" >&2
  exit 1
}
printf '%s\n' "$out" | grep -c "REBUILD=YES" || true
