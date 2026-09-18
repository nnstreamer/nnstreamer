#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file     debian-package-check.sh
# @brief    Verify properties of the built .deb files that only show at install time.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# A package can build, pass its tests and publish while being broken for
# whoever installs it. Two such defects lived in the PPA for years:
#
#   - nnstreamer-python3 shipped the symlink nnstreamer_python.so with a
#     literal "*" as the multiarch directory of its target, because dh_link
#     does not expand the glob its .links file used. The python3 sub-plugin
#     could not start.
#   - nnstreamer-openvino needs libcpu_extension.so, which has no SONAME, so
#     dpkg-shlibdeps could not turn it into a dependency and the filter failed
#     to load unless openvino-cpu-mkldnn happened to be installed.
#
# So this checks the packages a build produced:
#
#   - every package can be read at all, so that a damaged one cannot pass by
#     listing nothing,
#   - every symlink in every package resolves, without a glob, to a path
#     shipped by the same package or by one it depends on, directly or
#     through other packages of the build, since only those are certain to
#     be installed with it; a dependency offered as one of several
#     alternatives is not certain and does not count,
#   - nnstreamer-openvino for amd64 depends on openvino-cpu-mkldnn.
#
# A legacy-series build that lost nnstreamer-openvino altogether is not this
# script's concern: dh_install fails on the package's unmatched .install
# entry, and test_debian_rules_series.sh pins the option that enables it.
#
# The file lists come from the data tarball through tar rather than from the
# columns of "dpkg-deb -c", so that a hard link is recorded under its own
# name and a path containing a space stays whole. Globs below are written
# with the quote closed after the slash, because the doxygen check parses
# shell as C and would read a slash followed by a star as a comment opener.
#
# Argument ($1): directory holding the .deb files of one build.
#

set -u

failed=0

##
# @brief Print a path with "." and ".." components resolved lexically.
# @param $1 absolute path to normalize
normalize() {
  echo "$1" | awk -F/ '
    {
      n = 0
      for (i = 1; i <= NF; i++) {
        if ($i == "" || $i == ".") { continue }
        if ($i == "..") { if (n > 0) { n-- }; continue }
        parts[n++] = $i
      }
      out = ""
      for (i = 0; i < n; i++) { out = out "/" parts[i] }
      print (out == "" ? "/" : out)
    }'
}

##
# @brief Print every path a package ships, one absolute path per line.
# @param $1 data tarball of the package
list_shipped() {
  tar -tf "$1" | sed -e 's#^\.##' -e 's#/$##' -e 's#^$#/#'
}

##
# @brief Print every symlink of a package as "<link><TAB><target>".
#        The verbose tar listing has five fixed columns before the name; the
#        rest is split at the first " -> ", so a space on either side survives.
# @param $1 data tarball of the package
list_symlinks() {
  tar -tvf "$1" | awk '
    $1 ~ /^l/ {
      rest = $0
      for (i = 1; i <= 5; i++) { sub(/^[^ ]+ +/, "", rest) }
      n = index(rest, " -> ")
      if (n == 0) { next }
      link = substr(rest, 1, n - 1)
      sub(/^\./, "", link)
      print link "\t" substr(rest, n + 4)
    }'
}

##
# @brief Print the packages a package depends on, one "<package> <dependency>"
#        per line. Versions are dropped, and so is a clause with alternatives,
#        because none of its packages is certain to be installed.
# @param $1 path of the .deb
list_depends() {
  local name

  name=$(dpkg-deb -f "$1" Package)
  { dpkg-deb -f "$1" Pre-Depends; dpkg-deb -f "$1" Depends; } | tr ',' '\n' \
    | awk -v name="$name" 'NF && !/[|]/ { sub(/:.*/, "", $1); print name " " $1 }'
}

##
# @brief Tell whether installing one package brings another one with it.
# @param $1 package that is installed
# @param $2 package that has to come with it
# @param $3 file of "<package> <dependency>" lines for the whole build
brings() {
  awk -v from="$1" -v to="$2" '
    { edge[$1] = edge[$1] " " $2 }
    END {
      seen[from] = 1
      queue[0] = from
      head = 0
      tail = 1
      while (head < tail) {
        n = split(edge[queue[head++]], dep, " ")
        for (i = 1; i <= n; i++) {
          if (!(dep[i] in seen)) { seen[dep[i]] = 1; queue[tail++] = dep[i] }
        }
      }
      exit !(to in seen)
    }' "$3"
}

##
# @brief Check that every symlink of a package resolves to an installed path.
# @param $1 path of the .deb
# @param $2 data tarball of the package
# @param $3 file of "<path><TAB><package>" lines for the whole build
# @param $4 file of "<package> <dependency>" lines for the whole build
check_symlinks() {
  local deb=$1 data=$2 owners=$3 depends=$4
  local name link target resolved owner reached

  name=$(dpkg-deb -f "$deb" Package)
  while IFS=$'\t' read -r link target; do
    if echo "$target" | grep -q '[*?[]'; then
      echo "::error::$name: $link -> $target contains a glob; dh_link does not expand it."
      failed=1
      continue
    fi
    if [ "${target#/}" = "$target" ]; then
      resolved=$(normalize "$(dirname "$link")/$target")
    else
      resolved=$(normalize "$target")
    fi

    reached=""
    while IFS= read -r owner; do
      if brings "$name" "$owner" "$depends"; then
        reached=$owner
        break
      fi
      reached="-"
    done < <(awk -F'\t' -v path="$resolved" '$1 == path { print $2 }' "$owners")

    if [ -z "$reached" ]; then
      echo "::error::$name: $link -> $target resolves to $resolved, which no package of this build ships."
      failed=1
    elif [ "$reached" = "-" ]; then
      echo "::error::$name: $link -> $target resolves to $resolved, shipped only by packages $name does not depend on."
      failed=1
    else
      echo "ok   $name: $link -> $target (in $reached)"
    fi
  done < <(list_symlinks "$data")
}

##
# @brief Check the dependency the openvino filter needs beyond ${shlibs:Depends}.
# @param $1 path of the .deb
# @param $2 file of "<package> <dependency>" lines for the whole build
check_openvino_depends() {
  local deb=$1 depends=$2
  local name arch

  name=$(dpkg-deb -f "$deb" Package)
  [ "$name" = nnstreamer-openvino ] || return
  arch=$(dpkg-deb -f "$deb" Architecture)
  [ "$arch" = amd64 ] || return
  if grep -qxF "$name openvino-cpu-mkldnn" "$depends"; then
    echo "ok   $name: depends on openvino-cpu-mkldnn"
  else
    echo "::error::$name ($arch) does not depend on openvino-cpu-mkldnn, which ships libcpu_extension.so; the filter cannot load without it."
    failed=1
  fi
}

##
# @brief Check every package in a directory and exit with the verdict.
# @param $1 directory holding the .deb files
main() {
  local dir=${1:-}
  local workdir deb data name count=0

  if [ -z "$dir" ] || [ ! -d "$dir" ]; then
    echo "::error::Usage: $0 <directory with .deb files>"
    exit 1
  fi
  if ! command -v dpkg-deb > /dev/null 2>&1 || ! command -v tar > /dev/null 2>&1; then
    echo "::error::dpkg-deb or tar is not available."
    exit 1
  fi

  workdir=$(mktemp -d)
  trap 'rm -rf "$workdir"' EXIT
  touch "${workdir}/owners" "${workdir}/depends" "${workdir}/readable"

  for deb in "${dir}/"*.deb; do
    [ -e "$deb" ] || continue
    count=$((count + 1))
    data="${workdir}/${count}.tar"
    if ! dpkg-deb -f "$deb" Package > /dev/null 2>&1 \
        || ! dpkg-deb --fsys-tarfile "$deb" > "$data" 2> /dev/null \
        || ! tar -tf "$data" > /dev/null 2>&1; then
      echo "::error::$(basename "$deb") cannot be read as a .deb file."
      failed=1
      continue
    fi
    name=$(dpkg-deb -f "$deb" Package)
    list_shipped "$data" | awk -v name="$name" '{ print $0 "\t" name }' >> "${workdir}/owners"
    list_depends "$deb" >> "${workdir}/depends"
    printf '%s\t%s\n' "$deb" "$data" >> "${workdir}/readable"
  done

  if [ "$count" -eq 0 ]; then
    echo "::error::No .deb file found in $dir."
    exit 1
  fi

  while IFS=$'\t' read -r deb data; do
    echo "Checking $(basename "$deb")"
    check_symlinks "$deb" "$data" "${workdir}/owners" "${workdir}/depends"
    check_openvino_depends "$deb" "${workdir}/depends"
  done < "${workdir}/readable"

  if [ "$failed" -ne 0 ]; then
    echo "::error::The debian package check has failed."
    exit 1
  fi

  echo "The debian package check has passed."
  exit 0
}

main "$@"
