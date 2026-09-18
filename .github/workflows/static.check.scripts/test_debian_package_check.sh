#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file     test_debian_package_check.sh
# @brief    Self-test for debian-package-check.sh.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Builds small .deb files with dpkg-deb and runs the checker on them, so that
# each rule is shown to reject the defect it exists for and to accept the
# corrected form. Without the rejecting half a checker that matches nothing
# passes unnoticed; without the accepting half one that rejects everything
# does. The two shipped defects are reproduced literally: the python module
# link with "*" as a directory of its target, and nnstreamer-openvino without
# openvino-cpu-mkldnn. A hard link and paths with a space are covered as
# well, because a column-based reading of the package listing gets both
# wrong and the checker must not fail a build over them.
#
# The "*" comes from a variable: the doxygen check parses shell as C, and a
# slash followed by a star anywhere in this file would open a comment for it.
#

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CHECKER="${SCRIPT_DIR}/debian-package-check.sh"
workdir=$(mktemp -d)
failed=0

trap 'rm -rf "$workdir"' EXIT

##
# @brief Build a .deb into a case directory.
# @param $1 case directory
# @param $2 package name
# @param $3 architecture
# @param $4 Depends field, may be empty
# @param $5... "file:<path>" to ship an empty file, "link:<path>:<target>" to
#              ship a symlink, "hardlink:<path>:<existing>" to ship a hard link,
#              "field:<Name>: <value>" to add a control field
make_deb() {
  local dir=$1 name=$2 arch=$3 depends=$4
  local root="${dir}/${name}.root" entry kind path target
  shift 4

  mkdir -p "${root}/DEBIAN"
  {
    echo "Package: ${name}"
    echo "Version: 1"
    echo "Architecture: ${arch}"
    echo "Maintainer: test <test@example.com>"
    [ -n "$depends" ] && echo "Depends: ${depends}"
    echo "Description: fixture"
  } > "${root}/DEBIAN/control"

  for entry in "$@"; do
    kind=${entry%%:*}
    path=${entry#*:}
    case "$kind" in
      file)
        mkdir -p "${root}$(dirname "$path")"
        : > "${root}${path}"
        ;;
      link)
        target=${path#*:}
        path=${path%%:*}
        mkdir -p "${root}$(dirname "$path")"
        ln -s "$target" "${root}${path}"
        ;;
      field)
        echo "$path" >> "${root}/DEBIAN/control"
        ;;
      hardlink)
        target=${path#*:}
        path=${path%%:*}
        mkdir -p "${root}$(dirname "$path")"
        ln "${root}${target}" "${root}${path}"
        ;;
    esac
  done

  dpkg-deb -b --root-owner-group "$root" "${dir}/${name}_1_${arch}.deb" > /dev/null 2>&1 \
    || dpkg-deb -b "$root" "${dir}/${name}_1_${arch}.deb" > /dev/null 2>&1
  rm -rf "$root"
}

##
# @brief Run the checker on a case directory and compare the exit status.
# @param $1 expected exit status, 0 or 1
# @param $2 description
# @param $3 case directory
expect_checker() {
  local expected=$1 desc=$2 dir=$3
  local actual

  bash "$CHECKER" "$dir" > /dev/null 2>&1 < /dev/null
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

##
# @brief Create a fresh case directory and print its path.
# @param $1 case name
new_case() {
  local dir="${workdir}/$1"
  mkdir -p "$dir"
  echo "$dir"
}

##
# @brief Run every fixture and exit with the verdict.
main() {
  local dir star='*'

  if ! command -v dpkg-deb > /dev/null 2>&1; then
    echo "Skipped: dpkg-deb is not available here."
    exit 0
  fi

  dir=$(new_case relative_link_into_another_package)
  make_deb "$dir" helper amd64 "" file:/usr/lib/x86_64-linux-gnu/nnstreamer_python3.so
  make_deb "$dir" module amd64 "helper" \
    link:/usr/lib/python3/dist-packages/nnstreamer_python.so:../../x86_64-linux-gnu/nnstreamer_python3.so
  expect_checker 0 "a relative link into another package of the build is accepted" "$dir"

  dir=$(new_case absolute_link_within_package)
  make_deb "$dir" util amd64 "" file:/usr/lib/nnstreamer/bin/nnstreamer-check \
    link:/usr/bin/nnstreamer-check:/usr/lib/nnstreamer/bin/nnstreamer-check
  expect_checker 0 "an absolute link within the same package is accepted" "$dir"

  dir=$(new_case link_through_a_dependency_chain)
  make_deb "$dir" helper amd64 "" file:/usr/lib/x86_64-linux-gnu/nnstreamer_python3.so
  make_deb "$dir" middle amd64 "libc6 (>= 2.34) | other, helper (= 1)"
  make_deb "$dir" module amd64 "middle:any" \
    link:/usr/lib/python3/dist-packages/nnstreamer_python.so:../../x86_64-linux-gnu/nnstreamer_python3.so
  expect_checker 0 "a link into a package reached through another dependency is accepted" "$dir"

  dir=$(new_case link_through_pre_depends)
  make_deb "$dir" helper amd64 "" file:/usr/lib/x86_64-linux-gnu/nnstreamer_python3.so
  make_deb "$dir" module amd64 "" "field:Pre-Depends: helper (>= 1)" \
    link:/usr/lib/python3/dist-packages/nnstreamer_python.so:../../x86_64-linux-gnu/nnstreamer_python3.so
  expect_checker 0 "a link into a package named in Pre-Depends is accepted" "$dir"

  dir=$(new_case link_through_an_alternative)
  make_deb "$dir" helper amd64 "" file:/usr/lib/x86_64-linux-gnu/nnstreamer_python3.so
  make_deb "$dir" module amd64 "helper | other" \
    link:/usr/lib/python3/dist-packages/nnstreamer_python.so:../../x86_64-linux-gnu/nnstreamer_python3.so
  expect_checker 1 "a link into a package that is only an alternative is rejected" "$dir"

  dir=$(new_case openvino_cpu_extension_as_alternative)
  make_deb "$dir" nnstreamer-openvino amd64 "nnstreamer-single, openvino, openvino-cpu-mkldnn | other" \
    file:/usr/lib/nnstreamer/filters/libnnstreamer_filter_openvino.so
  expect_checker 1 "openvino-cpu-mkldnn offered only as an alternative is rejected" "$dir"

  dir=$(new_case link_into_an_unrelated_package)
  make_deb "$dir" helper amd64 "" file:/usr/lib/x86_64-linux-gnu/nnstreamer_python3.so
  make_deb "$dir" module amd64 "libc6" \
    link:/usr/lib/python3/dist-packages/nnstreamer_python.so:../../x86_64-linux-gnu/nnstreamer_python3.so
  expect_checker 1 "a link into a package the linking one does not depend on is rejected" "$dir"

  dir=$(new_case link_to_a_hard_link)
  make_deb "$dir" helper amd64 "" file:/usr/lib/x86_64-linux-gnu/nnstreamer_python3.so \
    hardlink:/usr/lib/x86_64-linux-gnu/nnstreamer_python3_alias.so:/usr/lib/x86_64-linux-gnu/nnstreamer_python3.so
  make_deb "$dir" module amd64 "helper" \
    link:/usr/lib/python3/dist-packages/nnstreamer_python.so:../../x86_64-linux-gnu/nnstreamer_python3_alias.so
  expect_checker 0 "a link to a hard-linked file is accepted" "$dir"

  dir=$(new_case paths_with_spaces)
  make_deb "$dir" helper amd64 "" "file:/usr/share/nnstreamer test/helper lib.so"
  make_deb "$dir" module amd64 "helper" \
    "link:/usr/lib/python3/dist-packages/module link.so:../../../share/nnstreamer test/helper lib.so"
  expect_checker 0 "a link and a target containing spaces are accepted" "$dir"

  dir=$(new_case dangling_link_with_spaces)
  make_deb "$dir" module amd64 "" \
    "link:/usr/lib/python3/dist-packages/module link.so:../../../share/nnstreamer test/helper lib.so"
  expect_checker 1 "a dangling link containing spaces is still rejected" "$dir"

  dir=$(new_case glob_link)
  make_deb "$dir" helper amd64 "" file:/usr/lib/x86_64-linux-gnu/nnstreamer_python3.so
  make_deb "$dir" module amd64 "helper" \
    "link:/usr/lib/python3/dist-packages/nnstreamer_python.so:../../${star}/nnstreamer_python3.so"
  expect_checker 1 "the shipped glob link is rejected" "$dir"

  dir=$(new_case unreadable_package)
  make_deb "$dir" util amd64 "" file:/usr/lib/nnstreamer/bin/nnstreamer-check
  head -c 300 "${dir}/util_1_amd64.deb" > "${dir}/truncated_1_amd64.deb"
  expect_checker 1 "a package that cannot be read is rejected, not skipped" "$dir"

  dir=$(new_case dangling_link)
  make_deb "$dir" module amd64 "" \
    link:/usr/lib/python3/dist-packages/nnstreamer_python.so:../../x86_64-linux-gnu/nnstreamer_python3.so
  expect_checker 1 "a link to a path no package ships is rejected" "$dir"

  dir=$(new_case openvino_with_cpu_extension)
  make_deb "$dir" nnstreamer-openvino amd64 "nnstreamer-single, openvino, openvino-cpu-mkldnn, libc6 (>= 2.34)" \
    file:/usr/lib/nnstreamer/filters/libnnstreamer_filter_openvino.so
  expect_checker 0 "nnstreamer-openvino amd64 with openvino-cpu-mkldnn is accepted" "$dir"

  dir=$(new_case openvino_without_cpu_extension)
  make_deb "$dir" nnstreamer-openvino amd64 "nnstreamer-single, openvino, libc6 (>= 2.34)" \
    file:/usr/lib/nnstreamer/filters/libnnstreamer_filter_openvino.so
  expect_checker 1 "nnstreamer-openvino amd64 without openvino-cpu-mkldnn is rejected" "$dir"

  dir=$(new_case openvino_arm64)
  make_deb "$dir" nnstreamer-openvino arm64 "nnstreamer-single, openvino" \
    file:/usr/lib/nnstreamer/filters/libnnstreamer_filter_openvino.so
  expect_checker 0 "nnstreamer-openvino arm64 needs no openvino-cpu-mkldnn" "$dir"

  dir=$(new_case other_package_without_cpu_extension)
  make_deb "$dir" nnstreamer-tvm amd64 "nnstreamer-single" \
    file:/usr/lib/nnstreamer/filters/libnnstreamer_filter_tvm.so
  expect_checker 0 "the openvino rule does not apply to other packages" "$dir"

  dir=$(new_case empty)
  expect_checker 1 "a directory without .deb files is rejected" "$dir"

  expect_checker 1 "a missing directory is rejected" "${workdir}/does-not-exist"

  if [ "$failed" -ne 0 ]; then
    echo "::error::test_debian_package_check.sh has failed."
    exit 1
  fi

  echo "test_debian_package_check.sh has passed."
  exit 0
}

main "$@"
