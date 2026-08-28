#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file     test_debian_control.sh
# @brief    Self-test for debian-control.sh.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Pins the properties that make the checker worth running. The compiler
# clause as it stood before #4848, naming only version-suffixed gcc
# packages, must fail; the same clause carrying an unversioned fallback must
# pass. Without the first the checker could accept everything unnoticed;
# without the second it could reject every control file. A gcc-13/gcc-12
# pair is checked too, so the rule is not read as a list of the specific
# compilers that happened to age out. The repository's own control files run
# last, which keeps the shipped tree covered even when no fixture changes.
#
# The checker is invoked with stdin closed, matching how GitHub runs it, so
# that a future reader-of-stdin bug fails here instead of hanging.
#

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
CHECKER="${SCRIPT_DIR}/debian-control.sh"
workdir=$(mktemp -d)
failed=0

trap 'rm -rf "$workdir"' EXIT

##
# @brief Run the checker on a generated control file and compare the exit status.
# @param $1 expected exit status, 0 or 1
# @param $2 description used for the fixture name and the report line
# @param $3 body of the control file
expect_checker() {
  local expected=$1 desc=$2 body=$3
  local target actual

  target="${workdir}/$(echo "$desc" | tr -c 'a-zA-Z0-9' '_')"
  printf '%s\n' "$body" > "$target"

  bash "$CHECKER" "$target" > /dev/null 2>&1 < /dev/null
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

expect_checker 1 "versioned-only gcc clause is rejected" \
'Source: nnstreamer
Build-Depends: gcc-9 | gcc-8 | gcc-7 | gcc-6 | gcc-5 (>=5.4),
 ninja-build, meson (>=0.62.0)
Standards-Version: 3.9.6'

expect_checker 0 "unversioned gcc fallback is accepted" \
'Source: nnstreamer
Build-Depends: gcc-9 | gcc-8 | gcc-7 | gcc-6 | gcc-5 (>=5.4) | gcc,
 ninja-build, meson (>=0.62.0)
Standards-Version: 3.9.6'

expect_checker 1 "a newer versioned-only pair is rejected as well" \
'Source: nnstreamer
Build-Depends: gcc-13 | gcc-12,
 ninja-build
Standards-Version: 3.9.6'

expect_checker 0 "arch and profile qualifiers do not confuse the parser" \
'Source: nnstreamer
Build-Depends: gcc-13 | gcc,
 libprotobuf-dev [amd64 arm64 armhf],
 pytorch <!nocheck> | libtorch-dev | gcc
Standards-Version: 3.9.6'

expect_checker 1 "a multiarch qualifier does not hide a versioned-only group" \
'Source: nnstreamer
Build-Depends: gcc-13:native | gcc-12:native,
 ninja-build
Standards-Version: 3.9.6'

expect_checker 0 "a multiarch qualifier on the fallback is accepted" \
'Source: nnstreamer
Build-Depends: gcc-13:native | gcc:native,
 ninja-build
Standards-Version: 3.9.6'

expect_checker 0 "a comment inside the field does not terminate it" \
'Source: nnstreamer
Build-Depends: gcc-9 | gcc,
# Please list more neural network frameworks as Debian includes them.
 ninja-build
Standards-Version: 3.9.6'

expect_checker 1 "an invalid package name is rejected" \
'Source: nnstreamer
Build-Depends: GCC-9 | GCC,
 ninja-build
Standards-Version: 3.9.6'

expect_checker 1 "a group in Build-Depends-Indep is not hidden by the field before it" \
'Source: nnstreamer
Build-Depends: ninja-build
Build-Depends-Indep: gcc-9 | gcc-8
Standards-Version: 3.9.6'

expect_checker 1 "a group in Build-Depends is not hidden by the field after it" \
'Source: nnstreamer
Build-Depends: gcc-9 | gcc-8
Build-Depends-Arch: ninja-build
Standards-Version: 3.9.6'

expect_checker 0 "several Build-Depends fields without a versioned-only group pass" \
'Source: nnstreamer
Build-Depends: gcc-9 | gcc
Build-Depends-Arch: ninja-build
Build-Depends-Indep: doxygen
Standards-Version: 3.9.6'

expect_checker 0 "fields after Build-Depends are not scanned" \
'Source: nnstreamer
Build-Depends: gcc-9 | gcc
Standards-Version: 3.9.6

Package: nnstreamer
Depends: gcc-9 | gcc-8
Description: not a build dependency'

echo "Checking the control files shipped in this repository"
if (cd "$REPO_ROOT" && bash "$CHECKER") < /dev/null; then
  echo "PASS: the repository control files satisfy the checker"
else
  echo "FAIL: the repository control files do not satisfy the checker"
  failed=1
fi

if [ "$failed" -ne 0 ]; then
  echo "::error::test_debian_control.sh has failed."
  exit 1
fi

echo "test_debian_control.sh has passed."
