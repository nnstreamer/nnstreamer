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
# 0.49, and the value that reaches the published .dsc disagreed with the value
# the source build resolves against. meson.build is the authority here,
# because it is the file meson itself enforces.
#
# Every tracked document and packaging file is scanned rather than a list of
# the sites that happen to declare it today, so a ninth one cannot appear
# unnoticed - which is the whole point of the check. Tracked, via git, so a
# build directory or a vendored copy left in the tree cannot fail the check on
# a file nobody owns; find is the fallback for a directory that is not a
# repository, which is how the self-test drives this.
#
# Argument ($1): repository root to check. Defaults to the repository this
#                script lives in.
#

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=${1:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
failed=0

# A declared requirement: meson, then >=, then a version, with a gap between
# them that may hold decoration but nothing that means the sentence moved on.
# Letters and digits are excluded, so a named neighbour breaks the match -
# "meson.build wants glib >= 2.60" is prose about glib, not a meson
# requirement - and so the only digits in a match belong to the version.
# Comma, period and semicolon are excluded too, so an unnamed version further
# along a sentence, as in "see meson, (>=5.0) elsewhere", is not attributed to
# meson either. What is left through is decoration: "meson (>=X)",
# "**meson** >= X", "`meson` >= X", "meson_version: >=X".
#
# This is a heuristic over prose and is deliberately not grown any further. A
# markdown link, "[meson](https://example.com) >= X", puts letters in the gap
# and is not recognised. If some other phrasing escapes it, write the
# requirement in one of the plain forms above rather than widening this.
DECL_RE='meson(_version)?[^[:alnum:],.;]{0,12}>=[[:space:]]*[0-9]+(\.[0-9]+)+'

##
# @brief Print a version padded to four components so 0.56 and 0.56.0 compare equal.
# @param $1 version string
normalize() {
  echo "$1" | awk -F. '{
    for (i = 1; i <= 4; i++) {
      printf "%d%s", ($i == "" ? 0 : $i), (i < 4 ? "." : "\n")
    }
  }'
}

if [ ! -f "${ROOT}/meson.build" ]; then
  echo "::error::${ROOT}/meson.build not found."
  exit 1
fi

expected_raw=$(grep -oE "meson_version:[[:space:]]*'>=[[:space:]]*[0-9]+(\.[0-9]+)+'" "${ROOT}/meson.build" \
  | head -1 | grep -oE '[0-9]+(\.[0-9]+)+')
if [ -z "$expected_raw" ]; then
  echo "::error::meson.build does not declare meson_version."
  exit 1
fi
expected=$(normalize "$expected_raw")
echo "meson.build declares meson_version >= ${expected_raw}"

##
# @brief Compare every declaration in one file against meson.build.
# @param $1 path of the file relative to the repository root
check_file() {
  local rel=$1
  local file="${ROOT}/${rel}"
  local hit lineno text decl found_raw found

  [ -f "$file" ] || return 0

  while IFS= read -r hit; do
    [ -z "$hit" ] && continue
    lineno=${hit%%:*}
    text=${hit#*:}
    # One line may carry more than one declaration, and the version has to be
    # read out of the matched text rather than the whole line: a neighbouring
    # constraint such as "debhelper (>=9.20)" would otherwise be picked up as
    # the meson requirement.
    while IFS= read -r decl; do
      [ -z "$decl" ] && continue
      found_raw=$(echo "$decl" | grep -oE '[0-9]+(\.[0-9]+)+' | tail -1)
      found=$(normalize "$found_raw")
      if [ "$found" = "$expected" ]; then
        echo "  ok   ${rel}:${lineno} (${found_raw})"
      else
        echo "::error::${rel}:${lineno} requires meson ${found_raw}, but meson.build requires ${expected_raw}. Keep every declaration in step."
        failed=1
      fi
    done < <(echo "$text" | grep -oiE "$DECL_RE")
  done < <(grep -niE "$DECL_RE" "$file")
}

while IFS= read -r rel; do
  [ -z "$rel" ] && continue
  check_file "$rel"
done < <(
  cd "$ROOT" || exit 1
  {
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      git ls-files -- '*.md' '*.spec' 'debian/control*'
    else
      find . -name .git -prune -o -type f \( -name '*.md' -o -name '*.spec' \) -print
      ls -1 debian/control* 2>/dev/null
    fi
  } | sed "s|^\./||" | sort -u
)

if [ "$failed" -ne 0 ]; then
  echo "::error::The meson version declarations disagree."
  exit 1
fi

echo "The meson version declarations agree."
