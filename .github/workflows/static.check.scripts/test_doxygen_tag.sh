#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file     test_doxygen_tag.sh
# @brief    Self-test for doxygen-tag.sh in the advanced (per-function) mode.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Runs doxygen-tag.sh the way static.check.yml does, with a changed-file list
# and the advanced flag. The per-function check had passed vacuously for two
# years because a stray "local" emptied the ctags kind list, so the first
# property pinned here is the negative control: an undocumented function
# definition must fail. The rest pin the scope of the check (definitions
# everywhere, prototypes only in headers), the trailing "/**<" form, and the
# per-file reset of the tag state.
#
# Fixtures are generated into a temporary directory rather than committed, so
# that the intentionally undocumented ones never appear in a changed-file list.

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
CHECKER="${SCRIPT_DIR}/doxygen-tag.sh"
workdir=$(mktemp -d)
failed=0

trap 'rm -rf "$workdir"' EXIT

if ! command -v ctags > /dev/null 2>&1; then
  echo "::error ctags is required by doxygen-tag.sh but is not installed."
  exit 1
fi

# File-level tags every C/C++ fixture needs so that only the per-function
# rules decide the outcome. The include line matters: the checker treats a
# @brief as pending until the next code line, and a real file always has one
# between the header comment and the first function.
header_block() {
  printf '/**\n * @file %s\n * @brief fixture\n * @author fixture\n * @bug none\n */\n#include <stddef.h>\n' "$1"
}

# write_fixture <name> <content>
# Creates workdir/<name> with the standard file header followed by content.
write_fixture() {
  local name=$1 content=$2
  { header_block "$name"; printf '%s\n' "$content"; } > "${workdir}/${name}"
}

# run_checker <expected 0|1> <description> <fixture>...
# Runs the checker on the listed fixtures in advanced mode, from the repo
# root, and compares the exit status. Also fails on any stderr output, which
# is how the broken "local" and the unset report_path used to show up.
run_checker() {
  local expected=$1 desc=$2
  shift 2
  local list errfile actual f

  list=$(mktemp -p "$workdir")
  errfile=$(mktemp -p "$workdir")
  for f in "$@"; do
    printf '%s\n' "${workdir}/${f}" >> "$list"
  done

  (cd "$REPO_ROOT" && bash "$CHECKER" "$list" 1) > /dev/null 2> "$errfile" < /dev/null
  actual=$?
  [[ $actual -ne 0 ]] && actual=1

  if [[ "$actual" != "$expected" ]]; then
    echo "FAIL: ${desc} expected exit ${expected}, got ${actual}"
    failed=1
  elif [[ -s "$errfile" ]]; then
    echo "FAIL: ${desc} wrote to stderr:"
    sed 's/^/    /' "$errfile"
    failed=1
  else
    echo "PASS: ${desc} (exit ${actual})"
  fi
}

write_fixture documented.c '/**
 * @brief add one
 */
int
add_one (int x)
{
  return x + 1;
}'
run_checker 0 "documented definition passes" documented.c

write_fixture undocumented.c 'int
add_one (int x)
{
  return x + 1;
}'
run_checker 1 "undocumented definition fails" undocumented.c

write_fixture forward_decl.c 'static int add_one (int x);

/**
 * @brief add one
 */
static int
add_one (int x)
{
  return x + 1;
}'
run_checker 0 "undocumented forward declaration in a .c passes" forward_decl.c

write_fixture undocumented_proto.h 'int add_one (int x);'
run_checker 1 "undocumented prototype in a .h fails" undocumented_proto.h

write_fixture documented_proto.h '/**
 * @brief add one
 */
int add_one (int x);'
run_checker 0 "documented prototype in a .h passes" documented_proto.h

write_fixture trailing_same_line.hh '/** @brief adder */
class adder
{
  public:
  /** @brief the class */
  adder ();
  int add_one (int x); /**< add one */
};'
run_checker 0 "trailing /**< on the declaration line passes" trailing_same_line.hh

write_fixture trailing_next_line.hh '/** @brief adder */
class adder
{
  public:
  /** @brief the class */
  adder ();
  virtual int add_one (int x) = 0;
  /**< add one */
  virtual int add_two (int x,
      int y) = 0;
  /**< add two, declared over two lines */
};'
run_checker 0 "trailing /**< on the following line passes" trailing_next_line.hh

write_fixture trailing_after_body.hh '/** @brief adder */
class adder
{
  public:
  /** @brief the class */
  adder ();
  virtual int add_one (int x)
  {
    return x + 1;
  }
  /**< add one, documented after the inline body */
};'
run_checker 0 "trailing /**< after an inline body passes" trailing_after_body.hh

write_fixture trailing_missing.hh '/** @brief adder */
class adder
{
  public:
  /** @brief the class */
  adder ();
  int add_one (int x);
  int add_two (int x);
  /**< documents add_two only */
};'
run_checker 1 "declaration without any doc between documented ones fails" trailing_missing.hh

write_fixture trailing_belongs_to_next.hh '/** @brief adder */
class adder
{
  public:
  /** @brief the class */
  adder ();
  int add_one (int x);
  int count; /**< documents count, not add_one */
};'
run_checker 1 "a /**< on the next declaration does not cover this one" trailing_belongs_to_next.hh

write_fixture undocumented_struct.h 'struct point
{
  int x;
};'
run_checker 1 "undocumented struct fails" undocumented_struct.h

# The tag must not survive a code line just because that line has a '*' in
# it: a pointer parameter is not a comment.
write_fixture pointer_line.h '/**
 * @brief add one in place
 */
void add_one (int *x);
void add_two (int *x);'
run_checker 1 "a brief does not carry over a pointer-typed declaration" pointer_line.h

write_fixture ends_in_brief.c '/**
 * @brief add one
 */
int
add_one (int x)
{
  return x + 1;
}
/**
 * @brief a dangling comment that leaves the tag state set
 */'
run_checker 0 "file ending in a brief passes on its own" ends_in_brief.c

# The file-level tags sit at the bottom here so that nothing resets the tag
# state before the first function; only the per-file reset can fail it.
{ printf 'int
add_one (int x)
{
  return x + 1;
}
'; header_block tail_header.c; } > "${workdir}/tail_header.c"
run_checker 1 "undocumented first function fails on its own" tail_header.c
run_checker 1 "tag state does not leak into the next file" ends_in_brief.c tail_header.c

write_fixture no_author.c '/**
 * @brief add one
 */
int
add_one (int x)
{
  return x + 1;
}'
sed -i '/@author/d' "${workdir}/no_author.c"
run_checker 1 "missing file-level @author fails" no_author.c

if [[ "$failed" != "0" ]]; then
  echo "::error test_doxygen_tag.sh failed."
  exit 1
fi
echo "test_doxygen_tag.sh: all checks passed."
