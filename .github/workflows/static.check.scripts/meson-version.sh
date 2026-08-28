#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file     meson-version.sh
# @brief    Check that the minimum meson version is declared consistently.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# The minimum meson is written down in eight places: meson.build, the three
# debian/control* files, packaging/nnstreamer.spec, AGENTS.md and two
# Documentation pages. Nothing tied them together, so raising the floor in
# 647b30e5 left debian/control.ubuntu.ppa and debian/control.debian behind at
# 0.49 and the value that reaches the published .dsc disagreed with the value
# the source build resolves against. meson.build is the authority here,
# because it is the file meson itself enforces.
#
# Argument ($1): repository root to check. Defaults to the repository this
#                script lives in.
#

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=${1:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
failed=0

# A declared requirement: the word meson, then >=, then a version. Plain
# mentions of a version some distro happens to ship carry no >= and are left
# alone.
DECL_RE='[Mm]eson[^0-9]{0,24}>=[[:space:]]*[0-9]+(\.[0-9]+)+'

##
# @brief Print a version as major.minor.patch so 0.56 and 0.56.0 compare equal.
# @param $1 version string
normalize() {
  echo "$1" | awk -F. '{ printf "%d.%d.%d\n", $1, ($2 == "" ? 0 : $2), ($3 == "" ? 0 : $3) }'
}

if [ ! -f "${ROOT}/meson.build" ]; then
  echo "::error::${ROOT}/meson.build not found."
  exit 1
fi

expected_raw=$(grep -oE "meson_version:[[:space:]]*'>=[0-9]+(\.[0-9]+)+'" "${ROOT}/meson.build" \
  | head -1 | grep -oE '[0-9]+(\.[0-9]+)+')
if [ -z "$expected_raw" ]; then
  echo "::error::meson.build does not declare meson_version."
  exit 1
fi
expected=$(normalize "$expected_raw")
echo "meson.build declares meson_version >= ${expected_raw}"

targets=$(cd "$ROOT" && ls -1 AGENTS.md packaging/nnstreamer.spec debian/control* 2>/dev/null; \
          cd "$ROOT" && find Documentation -maxdepth 1 -type f -name '*.md' 2>/dev/null | sort)

for rel in $targets; do
  file="${ROOT}/${rel}"
  [ -f "$file" ] || continue
  while IFS= read -r hit; do
    [ -z "$hit" ] && continue
    lineno=${hit%%:*}
    found_raw=$(echo "${hit#*:}" | grep -oE '[0-9]+(\.[0-9]+)+' | tail -1)
    found=$(normalize "$found_raw")
    if [ "$found" = "$expected" ]; then
      echo "  ok   ${rel}:${lineno} (${found_raw})"
    else
      echo "::error::${rel}:${lineno} requires meson ${found_raw}, but meson.build requires ${expected_raw}. Keep every declaration in step."
      failed=1
    fi
  done < <(grep -nE "$DECL_RE" "$file" | grep -oE "^[0-9]+:.*")
done

if [ "$failed" -ne 0 ]; then
  echo "::error::The meson version declarations disagree."
  exit 1
fi

echo "The meson version declarations agree."
