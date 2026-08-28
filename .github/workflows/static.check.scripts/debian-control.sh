#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file     debian-control.sh
# @brief    Verify that debian/control* build-dependencies stay satisfiable.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Ubuntu 25.04 and later, and every current Debian release, dropped gcc-5
# through gcc-9 from their archives. A Build-Depends alternative group that
# names only version-suffixed toolchain packages therefore becomes
# unsatisfiable on a new distro, and the PPA source build parks in
# "Dependency wait" instead of failing loudly, so no package is ever
# published for that series. Require an unversioned alternative, which the
# distro default compiler always provides.
#
# Argument ($@): control files to check. Defaults to the repository's
#                debian/control* when no argument is given.
#

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
failed=0

# Toolchain packages whose version-suffixed form leaves distro archives as
# the compiler ages out.
TOOLCHAIN_RE='^(gcc|g\+\+|cpp|clang|clang\+\+|llvm)-[0-9]+(\.[0-9]+)*$'
PKGNAME_RE='^[a-z0-9][a-z0-9+.-]*$'

##
# @brief Print a string with its runs of whitespace collapsed into single spaces.
# @param $1 string to normalize
squeeze() {
  echo "$1" | awk '{ $1 = $1; print }'
}

##
# @brief Print the Build-Depends* fields of a control file, one clause per line.
# @param $1 path of the control file
extract_clauses() {
  awk '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { collecting = 0; next }
    /^[A-Za-z0-9][A-Za-z0-9-]*:/ {
      collecting = ($0 ~ /^Build-Depends(-Arch|-Indep)?:/)
      if (!collecting) { next }
      sub(/^[^:]*:/, "")
    }
    collecting { printf "%s ", $0 }
    END { printf "\n" }
  ' "$1" | tr ',' '\n'
}

##
# @brief Report the build dependencies of one control file that cannot hold.
# @param $1 path of the control file
check_file() {
  local file=$1
  local clause alt name toolchain_only alt_count

  if [ ! -f "$file" ]; then
    echo "::error::$file does not exist."
    failed=1
    return
  fi

  while IFS= read -r clause; do
    clause=$(echo "$clause" | tr -d '\r' | sed -e 's/([^)]*)//g' -e 's/\[[^]]*\]//g' -e 's/<[^>]*>//g')
    if [ -z "$(echo "$clause" | tr -d '[:space:]')" ]; then
      continue
    fi

    toolchain_only=1
    alt_count=0
    while IFS= read -r alt; do
      name=$(echo "$alt" | awk '{print $1}')
      if [ -z "$name" ]; then
        continue
      fi
      alt_count=$((alt_count + 1))
      if ! echo "$name" | grep -Eq "$PKGNAME_RE"; then
        echo "::error::$file: invalid package name '$name' in build dependency '$(squeeze "$clause")'."
        failed=1
      fi
      if ! echo "$name" | grep -Eq "$TOOLCHAIN_RE"; then
        toolchain_only=0
      fi
    done < <(echo "$clause" | tr '|' '\n')

    if [ "$alt_count" -gt 0 ] && [ "$toolchain_only" -eq 1 ]; then
      echo "::error::$file: '$(squeeze "$clause")' lists only version-suffixed toolchain packages, which distros eventually drop. Add an unversioned alternative such as '| gcc'."
      failed=1
    fi
  done < <(extract_clauses "$file")
}

files=("$@")
if [ "${#files[@]}" -eq 0 ]; then
  mapfile -t files < <(find "${REPO_ROOT}/debian" -maxdepth 1 -type f -name 'control*' 2>/dev/null | sort)
fi

if [ "${#files[@]}" -eq 0 ]; then
  echo "::error::No debian control file to check."
  exit 1
fi

for file in "${files[@]}"; do
  echo "Checking $file"
  check_file "$file"
done

if [ "$failed" -ne 0 ]; then
  echo "::error::The debian control build-dependency check has failed."
  exit 1
fi

echo "The debian control build-dependency check has passed."
