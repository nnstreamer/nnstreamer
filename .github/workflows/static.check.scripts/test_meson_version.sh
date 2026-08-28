#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file     test_meson_version.sh
# @brief    Self-test for meson-version.sh.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Builds throwaway trees and pins what the checker must say about each: a
# declaration left behind must fail, matching declarations must pass, and a
# version a distro merely ships, written without >=, must not be mistaken for
# a requirement. Without the first the checker could accept any drift; without
# the last it would fail on prose that is simply describing a release. The
# repository's own tree is checked at the end so the shipped files stay
# covered when no fixture changes.
#
# The checker is invoked with stdin closed, matching how GitHub runs it, so
# that a future reader-of-stdin bug fails here instead of hanging.
#

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
CHECKER="${SCRIPT_DIR}/meson-version.sh"
workdir=$(mktemp -d)
failed=0

trap 'rm -rf "$workdir"' EXIT

##
# @brief Build a tree from "path<TAB>content" lines and run the checker on it.
# @param $1 expected exit status, 0 or 1
# @param $2 description used for the tree name and the report line
# @param $3 newline separated "relative path<TAB>file content" entries
expect_checker() {
  local expected=$1 desc=$2 spec=$3
  local root actual rel body

  root="${workdir}/$(echo "$desc" | tr -c 'a-zA-Z0-9' '_')"
  mkdir -p "$root"
  while IFS=$'\t' read -r rel body; do
    [ -z "$rel" ] && continue
    mkdir -p "$(dirname "${root}/${rel}")"
    printf '%s\n' "$body" > "${root}/${rel}"
  done <<< "$spec"

  bash "$CHECKER" "$root" > /dev/null 2>&1 < /dev/null
  actual=$?
  if [ $actual -ne 0 ]; then
    actual=1
  fi

  if [ "$actual" = "$expected" ]; then
    echo "PASS: ${desc} (exit ${actual})"
  else
    echo "FAIL: ${desc} expected exit ${expected}, got ${actual}"
    failed=1
  fi
}

MESON_BUILD=$'meson.build\tproject(\'nnstreamer\', \'c\', meson_version: \'>=0.56.0\')'

expect_checker 0 "declarations that all agree pass" \
"${MESON_BUILD}
$(printf 'debian/control\tBuild-Depends: ninja-build, meson (>=0.56.0), debhelper (>=9)')
$(printf 'AGENTS.md\tBuild system: **Meson >= 0.56.0 + Ninja**.')"

expect_checker 1 "a control file left behind is rejected" \
"${MESON_BUILD}
$(printf 'debian/control\tBuild-Depends: ninja-build, meson (>=0.49), debhelper (>=9)')"

expect_checker 1 "a document left behind is rejected" \
"${MESON_BUILD}
$(printf 'Documentation/getting-started-meson-build.md\t* meson >= 0.62.0')"

expect_checker 1 "a spec file left behind is rejected" \
"${MESON_BUILD}
$(printf 'packaging/nnstreamer.spec\tBuildRequires:\tmeson >= 0.62.0')"

expect_checker 0 "an omitted patch component still matches" \
"${MESON_BUILD}
$(printf 'AGENTS.md\tBuild system: **Meson >= 0.56 + Ninja**.')"

expect_checker 0 "a version a distro ships is not read as a requirement" \
"${MESON_BUILD}
$(printf 'Documentation/getting-started-meson-build.md\tUbuntu 20.04 ships meson 0.53.2, which is too old.')"

echo "Checking the tree shipped in this repository"
if bash "$CHECKER" "$REPO_ROOT" < /dev/null; then
  echo "PASS: the repository declares meson consistently"
else
  echo "FAIL: the repository meson declarations disagree"
  failed=1
fi

if [ "$failed" -ne 0 ]; then
  echo "::error::test_meson_version.sh has failed."
  exit 1
fi

echo "test_meson_version.sh has passed."
