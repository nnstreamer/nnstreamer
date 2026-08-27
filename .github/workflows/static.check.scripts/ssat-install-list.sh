#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file     ssat-install-list.sh
# @brief    Keep the install_subdir list in tests/meson.build in sync with the SSAT suites
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# The list that fills the unittest-nnstreamer package is hand-maintained, so a new
# tests/<suite>/runTest.sh can be added without ever being shipped, and an entry can
# outlive the directory it names. Both happened. This checks the two directions.
#

meson_file="tests/meson.build"

# Suites deliberately absent from the package. Keep the reason next to the name.
#   codegen: runs tools/development/nnstreamerCodeGenCustomFilter.py and compiles
#            against gst/nnstreamer through ../../ paths that only a source tree has.
not_installed=("codegen")

if [ ! -f "${meson_file}" ]; then
  echo "::error::${meson_file} not found. Run this from the repository root."
  exit 1
fi

failed=0
installed=$(grep -vE '^[[:space:]]*#' "${meson_file}" | grep -oE "install_subdir\('[^']+'" | cut -d"'" -f2)

# Every SSAT suite should be installed, unless it is listed above.
while IFS= read -r script; do
  suite=$(basename "$(dirname "${script}")")

  skip=0
  for exempt in "${not_installed[@]}"; do
    if [ "${suite}" = "${exempt}" ]; then skip=1; break; fi
  done
  [ ${skip} -eq 1 ] && continue

  if ! echo "${installed}" | grep -qxF "${suite}"; then
    echo "::error::tests/${suite} has a runTest.sh but is missing from ${meson_file}, so it will not reach the unittest-nnstreamer package. Add install_subdir('${suite}', install_dir: unittest_install_dir), or add it to not_installed in $0 with the reason."
    failed=1
  fi
done < <(find tests -mindepth 2 -maxdepth 2 -name runTest.sh)

# Every entry should name a directory that exists.
for suite in ${installed}; do
  if [ ! -d "tests/${suite}" ]; then
    echo "::error::${meson_file} installs '${suite}', but tests/${suite} does not exist."
    failed=1
  fi
done

if [ ${failed} -ne 0 ]; then
  exit 1
fi

echo "The SSAT install list in ${meson_file} is in sync."
