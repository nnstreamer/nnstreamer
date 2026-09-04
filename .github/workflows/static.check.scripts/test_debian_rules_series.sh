#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file     test_debian_rules_series.sh
# @brief    Pin the series-dependent decisions debian/rules makes.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# debian/rules picks the control file and the openvino option from the Ubuntu
# release it is running on, because the PPA stack differs between series (see
# the header of debian/control). Getting that wrong publishes a package built
# against the wrong dependencies, and the only other check for it is a full
# pdebuild, which needs a populated PPA and a quarter of an hour.
#
# This asserts the decisions themselves, from the recipes make would run, so
# it costs a second and works on any Ubuntu host:
#
#   - which control file override_dh_clean copies, if any,
#   - which -Denable-openvino value override_dh_auto_configure passes,
#   - that DEB_BUILD_OPTIONS=nocheck skips override_dh_auto_test, which dh
#     does not do by itself for an override target at this compat level.
#
# Each skip is paired with the assertion that the same command runs without
# nocheck, so that deleting a test command cannot turn the skip green.
#
# make gives a command-line variable precedence over a := assignment, so the
# release under test is supplied as UBUNTU_VERSION_ID=<value> and no stub of
# /etc/os-release is needed. Every case runs "make -n", which prints recipes
# without running them, so debian/control is never actually overwritten.
#
# Everything below sits in a function because the doxygen build check parses
# a shell script as C, where a top-level "if" guarding an echo of a GitHub
# "::error::" annotation reads as a declaration and warns.
#

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
RULES="${REPO_ROOT}/debian/rules"
EXPANDED=""
failed=0

##
# @brief Stop before the checks when this host cannot answer them. A host that
#        is not Ubuntu, or has no make, cannot reach the branch under test, so
#        that is a skip rather than a failure; a missing debian/rules is not.
require_ubuntu_make() {
  if [ ! -f "${RULES}" ]; then
    echo "::error::${RULES} not found."
    exit 1
  fi

  if ! command -v make > /dev/null 2>&1 || ! command -v dpkg-vendor > /dev/null 2>&1; then
    echo "Skipped: make or dpkg-vendor is not available here."
    exit 0
  fi

  if ! dpkg-vendor --derives-from Ubuntu; then
    echo "Skipped: this host is not Ubuntu, so the Ubuntu branch is unreachable."
    exit 0
  fi
}

##
# @brief Expand into EXPANDED the recipe make would run for a target.
#        Fails the suite when make cannot read debian/rules at all, so that a
#        broken file cannot leave the assertions below matching nothing.
# @param $1 value for UBUNTU_VERSION_ID, may be empty
# @param $2 target to expand
expand() {
  if ! EXPANDED=$(make -f "${RULES}" -C "${REPO_ROOT}" -n "$2" "UBUNTU_VERSION_ID=$1" 2>&1); then
    echo "::error::  FAIL make could not expand $2 at UBUNTU_VERSION_ID=${1:-<empty>}:"
    echo "${EXPANDED}"
    failed=1
    return 1
  fi
  return 0
}

##
# @brief Report one assertion.
# @param $1 outcome, 0 for pass
# @param $2 description
report() {
  if [ "$1" -eq 0 ]; then
    echo "  ok   $2"
  else
    echo "::error::  FAIL $2"
    failed=1
  fi
}

##
# @brief Report whether a recipe contains a pattern, and whether it should.
# @param $1 recipe text
# @param $2 pattern to look for
# @param $3 "yes" if the pattern is expected, "no" if it must be absent
# @param $4 description
expect_match() {
  local found=1

  if echo "$1" | grep -q -- "$2"; then
    found=0
  fi

  if [ "$3" = yes ]; then
    report "${found}" "$4"
  else
    report "$([ "${found}" -ne 0 ] && echo 0 || echo 1)" "$4"
  fi
}

##
# @brief Assert the control file and openvino option chosen for one release.
# @param $1 value for UBUNTU_VERSION_ID, may be empty
# @param $2 "legacy" if control.ubuntu.ppa should be copied, else "committed"
# @param $3 expected -Denable-openvino value
expect_series() {
  local version=$1 stack=$2 openvino=$3
  local shown=${version:-<empty>}
  local clean configure copied

  expand "${version}" override_dh_clean || return
  clean=${EXPANDED}
  expand "${version}" override_dh_auto_configure || return
  configure=${EXPANDED}

  if [ "${stack}" = legacy ]; then copied=yes; else copied=no; fi
  expect_match "${clean}" 'cp debian/control.ubuntu.ppa' "${copied}" \
    "${shown}: takes the ${stack} control"
  expect_match "${configure}" "-Denable-openvino=${openvino}" yes \
    "${shown}: configures -Denable-openvino=${openvino}"
}

##
# @brief Assert that nocheck reaches the test override, and only nocheck does.
check_nocheck() {
  local with without

  DEB_BUILD_OPTIONS="" expand "26.04" override_dh_auto_test || return
  without=${EXPANDED}
  DEB_BUILD_OPTIONS="nocheck parallel=4" expand "26.04" override_dh_auto_test || return
  with=${EXPANDED}

  # Each pair pins one command: present by default, gone under nocheck. The
  # positive half is what keeps a deleted command from passing as a skip.
  expect_match "${without}" 'run_unittests_binaries.sh' yes \
    "without nocheck: the unit tests are run"
  expect_match "${with}" 'run_unittests_binaries.sh' no \
    "with nocheck: the unit tests are skipped"
  expect_match "${without}" 'ssat' yes \
    "without nocheck: SSAT is run"
  expect_match "${with}" 'ssat' no \
    "with nocheck: SSAT is skipped"
}

##
# @brief Run every check and exit with the verdict.
main() {
  require_ubuntu_make

  echo "Checking the series-dependent decisions in debian/rules"

  # The PPA carries openvino, onert and tvm up to 24.04 only, so every release
  # below 26.04 has to take the legacy control and every release from 26.04 on
  # has to take the committed one. An unset VERSION_ID compares as the earliest
  # version, which lands on the legacy stack: the safe side, since that is what
  # every published series used before 26.04.
  expect_series "22.04" legacy    true
  expect_series "24.04" legacy    true
  expect_series "25.10" legacy    true
  expect_series "26.04" committed false
  expect_series "26.10" committed false
  expect_series ""      legacy    true

  echo "Checking DEB_BUILD_OPTIONS=nocheck"
  check_nocheck

  if [ "${failed}" -ne 0 ]; then
    echo "::error::test_debian_rules_series.sh has failed."
    exit 1
  fi

  echo "test_debian_rules_series.sh has passed."
  exit 0
}

main "$@"
