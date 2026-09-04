#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# SPDX-License-Identifier: LGPL-2.1-only
#
# @file     test_run_unittests_binaries.sh
# @brief    Self-test for run_unittests_binaries.sh.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Pins the three properties the Valgrind job depends on and that a plain
# reading of the script cannot show: every binary in the tree is run even when
# an earlier one fails, --recursive finds the binaries built one directory down
# while a plain run still does not, and a file that merely looks like a test
# binary is not executed. Each was wrong before: the loop stopped at the first
# failure, so whichever binaries the file system happened to list after it went
# unexamined, and there was no way to reach the binaries under
# tests/cpp_methods and tests/nnstreamer_datarepo at all.
#
# Both kinds of fixture name themselves in their output, through the
# --gtest_output argument the runner builds from the binary's own name: a
# passing one is a copy of echo, a failing one a copy of env, which rejects
# that argument and exits non-zero. That is what makes "did this one run"
# observable without a test framework, for the failing entries too.
#
# Two of the fixtures fail, which is what makes the fail-fast check independent
# of the order the file system lists them in: whichever order that is, a loop
# that stopped at the first failure would leave at least one later entry
# unrun, because a second failing entry is always there to follow it.
#

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RUNNER="${SCRIPT_DIR}/run_unittests_binaries.sh"
ECHO_BIN=$(type -P echo || true)
ENV_BIN=$(type -P env || true)
workdir=$(mktemp -d)
failed=0

trap 'rm -rf "$workdir"' EXIT

##
# @brief Report one expectation and remember a failure.
# @param $1 0 when the expectation held, anything else when it did not
# @param $2 description of the expectation
report() {
  if [ "$1" -eq 0 ]; then
    echo "PASS: $2"
  else
    echo "FAIL: $2"
    failed=1
  fi
}

##
# @brief Assert that the captured output contains the given text.
# @param $1 captured output
# @param $2 text that must appear
# @param $3 description of the expectation
expect_in() {
  case "$1" in
    *"$2"*) report 0 "$3" ;;
    *) report 1 "$3" ;;
  esac
}

##
# @brief Assert that the captured output does not contain the given text.
# @param $1 captured output
# @param $2 text that must not appear
# @param $3 description of the expectation
expect_not_in() {
  case "$1" in
    *"$2"*) report 1 "$3" ;;
    *) report 0 "$3" ;;
  esac
}

##
# @brief Build a throwaway build/tests tree of fixture binaries.
# @param $1 directory to build the tree in
make_tree() {
  local root=$1
  mkdir -p "${root}/build/tests/sub"
  cp "${ECHO_BIN}" "${root}/build/tests/unittest_alpha"
  cp "${ENV_BIN}" "${root}/build/tests/unittest_beta"
  cp "${ECHO_BIN}" "${root}/build/tests/unittest_gamma"
  cp "${ENV_BIN}" "${root}/build/tests/unittest_epsilon"
  cp "${ECHO_BIN}" "${root}/build/tests/sub/unittest_delta"
  printf '#!/bin/sh\necho ran-the-text-file\n' > "${root}/build/tests/unittest_zeta.cc"
  chmod +x "${root}/build/tests/unittest_zeta.cc"
}

if [ ! -f "${RUNNER}" ]; then
  echo "FAIL: ${RUNNER} is missing."
  exit 1
fi

if [ -z "${ECHO_BIN}" ] || [ -z "${ENV_BIN}" ]; then
  echo "FAIL: this test needs the echo and env binaries on PATH."
  exit 1
fi

tree="${workdir}/whole"
make_tree "${tree}"

# Negative controls. Without these, the check that the runner does not run the
# text file and the checks that read a fixture's own output would all hold for
# the wrong reason if the fixtures did not behave as assumed.
control=$("${tree}/build/tests/unittest_zeta.cc" 2>&1 || true)
expect_in "${control}" "ran-the-text-file" \
    "the text fixture does run when it is invoked directly"
control=$("${tree}/build/tests/unittest_beta" --gtest_output=probe 2>&1 || true)
expect_in "${control}" "probe" "the failing fixture echoes the argument it rejected"
"${tree}/build/tests/unittest_beta" --gtest_output=probe > /dev/null 2>&1
[ $? -ne 0 ] && report 0 "the failing fixture exits non-zero" \
    || report 1 "the failing fixture exits non-zero"

out=$(cd "${tree}" && bash "${RUNNER}" --recursive ./tests/ 2>&1)
status=$?

[ ${status} -ne 0 ] && report 0 "a failing binary makes the run report a failure" \
    || report 1 "a failing binary makes the run report a failure"
expect_in "${out}" "unittest_alpha.xml" "a passing binary is run"
expect_in "${out}" "unittest_gamma.xml" "every passing binary is run, not just the first"
expect_in "${out}" "unittest_beta.xml" "a failing binary is run"
expect_in "${out}" "unittest_epsilon.xml" "a failure does not stop the binaries after it"
expect_in "${out}" "unittest_delta.xml" "--recursive runs a binary in a sub-directory"
expect_in "${out}" "unittest_beta" "the failing binaries are named in the summary"
expect_in "${out}" "unittest_epsilon" "the summary names every failing binary"
expect_not_in "${out}" "ran-the-text-file" \
    "an executable file that is not an ELF binary is not run"

# Without --recursive the search must stay where it was, so that the callers
# that walk the sub-directories themselves do not run them a second time.
out=$(cd "${tree}" && bash "${RUNNER}" ./tests/ 2>&1)
expect_in "${out}" "unittest_alpha.xml" "the top level is searched without --recursive"
expect_not_in "${out}" "unittest_delta.xml" \
    "a sub-directory is left alone without --recursive"

tree="${workdir}/skipped"
make_tree "${tree}"
out=$(cd "${tree}" && bash "${RUNNER}" --skip unittest_gamma ./tests/ 2>&1)
expect_not_in "${out}" "unittest_gamma.xml" "--skip keeps the named binary from running"
expect_in "${out}" "unittest_alpha.xml" "--skip leaves the other binaries running"

tree="${workdir}/single"
make_tree "${tree}"
out=$(cd "${tree}" && bash "${RUNNER}" ./tests/unittest_alpha 2>&1)
status=$?
[ ${status} -eq 0 ] && report 0 "a single passing binary is accepted by path" \
    || report 1 "a single passing binary is accepted by path"
expect_in "${out}" "unittest_alpha.xml" "a single binary given by path is run"
expect_not_in "${out}" "unittest_gamma.xml" "a single binary given by path runs alone"

if [ ${failed} -ne 0 ]; then
  echo "run_unittests_binaries.sh self-test failed."
  exit 1
fi

echo "run_unittests_binaries.sh self-test passed."
exit 0
