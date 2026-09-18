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
#     does not do by itself for an override target at this compat level,
#   - that override_dh_link creates the python module link from the multiarch
#     path of this host and still runs dh_link for the other packages, and
#     that no .links file under debian/ relies on a glob, which dh_link does
#     not expand: it shipped a link to a literal "*" for years.
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
# @brief Assert that the build never drops nnstreamer-openvino.install.
#        While it stays, dh_install aborts a legacy build whose openvino
#        filter is missing, which debian-package-check.sh relies on instead
#        of checking for the package itself. The nnfw line is the positive
#        half: it shows the recipe that removes .install files was read.
check_openvino_install_kept() {
  expand "24.04" override_dh_auto_build || return

  expect_match "${EXPANDED}" 'rm debian/nnstreamer-nnfw.install' yes \
    "override_dh_auto_build: the conditional .install removals are visible"
  expect_match "${EXPANDED}" 'nnstreamer-openvino.install' no \
    "override_dh_auto_build: nnstreamer-openvino.install is never removed"
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
# @brief Assert that the python module link is built from the multiarch path
#        and that the override still links the other packages.
check_python3_link() {
  local multiarch link

  multiarch=$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null)
  if [ -z "${multiarch}" ]; then
    echo "::error::  FAIL dpkg-architecture gave no DEB_HOST_MULTIARCH"
    failed=1
    return
  fi

  expand "24.04" override_dh_link || return
  link=${EXPANDED}

  expect_match "${link}"     "dh_link -pnnstreamer-python3 usr/lib/${multiarch}/nnstreamer_python3.so usr/lib/python3/dist-packages/nnstreamer_python.so"     yes "override_dh_link: links nnstreamer_python.so to the ${multiarch} helper"
  expect_match "${link}" 'dh_link --remaining-packages' yes     "override_dh_link: still links the remaining packages"
  expect_match "${link}" '[*?[]' no     "override_dh_link: uses no glob"
}

##
# @brief Assert that no .links file under debian/ contains a glob character.
#        The glob below closes its quote after the slash: the doxygen check
#        parses shell as C, where a slash followed by a star opens a comment.
check_links_files() {
  local file found=0

  for file in "${REPO_ROOT}/debian/"*.links; do
    [ -e "${file}" ] || continue
    found=1
    if grep -q '[*?[]' "${file}"; then
      report 1 "$(basename "${file}"): contains a glob, which dh_link does not expand"
    else
      report 0 "$(basename "${file}"): has no glob"
    fi
  done

  if [ "${found}" -eq 0 ]; then
    report 0 "no .links file under debian/ to check"
  fi
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

  check_openvino_install_kept

  echo "Checking DEB_BUILD_OPTIONS=nocheck"
  check_nocheck

  echo "Checking the python module link"
  check_python3_link
  check_links_files

  if [ "${failed}" -ne 0 ]; then
    echo "::error::test_debian_rules_series.sh has failed."
    exit 1
  fi

  echo "test_debian_rules_series.sh has passed."
  exit 0
}

main "$@"
