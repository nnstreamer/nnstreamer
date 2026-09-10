#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# SPDX-License-Identifier: LGPL-2.1-only
#
# @file     check_valgrind_log.sh
# @brief    Fail when a memcheck error comes from this repository's own code.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# usage: check_valgrind_log.sh [--repo-root DIR] [--require-output] LOGFILE...
#
# Reads the text memcheck writes and reports each error twice over: once by
# what kind of error it is, and once by whose code the reported frame is in.
# Only a non-leak error whose reported frame is ours makes this exit non-zero.
#
# An error is attributed to the frame memcheck blames for it - the first one
# that is not valgrind's own allocation wrapper - and that frame is ours when
# it names a source file this repository holds or an object under the build
# directory. Nothing below that frame counts, so neither the origin stack that
# --track-origins appends nor the stack of the block an address belongs to can
# make an error ours; both are usually ours even when the error is not.
#
# The cost of that is an error blamed on a library function we called badly,
# where our frame is the one below. Memcheck redirects the string and memory
# functions through its own wrappers, which this skips, so in practice the
# blame lands on our caller anyway; across the two runs this was measured on,
# no error blamed on a library had a frame of ours beneath it.
#
# An error every one of whose frames is a wrapper therefore has nothing to be
# blamed on, and is counted as neither. Deciding otherwise would mean carrying
# the error past the end of its own stack, which is the mistake this rule
# exists to prevent; neither of the two runs holds such an error.
#
# Both halves of "ours" carry an assumption. A file is matched by its base
# name, so a source of ours sharing a name with one a library was built from
# would start gating on that library's bug - no name in the tree does today.
# The object half looks for /build/gst, /build/ext or /build/tests anywhere in
# the path, which is what lets a log recorded on another machine be checked
# here, and which stops recognising objects if the build directory is ever
# renamed; the file-name half still covers everything built with debug info.
#
# A report whose first line matches none of the kinds below is neither passed
# nor failed on, so it is printed as a warning instead of disappearing.
#
# A definite leak is attributed differently. Memcheck reports it where its
# block was allocated, which for most of ours is inside GLib, so it is ours
# when any frame of that stack is, and it is decided by the innermost such
# frame. Two callers are exempt even so: a block allocated while the dynamic
# loader loads a library or runs its initialisers belongs to the loader or to
# that library, and GStreamer leaks the list of log functions it replaces on
# purpose. A stack that passes through either before reaching a frame of ours
# is not ours. Definite leaks of ours are listed and warned about but do not
# fail the check while the tree still holds some; possible and indirect leaks
# are only counted, the former being mostly the thread-local storage of
# threads still running at exit and the latter reachable only through a
# definite one.
#
# --require-output additionally fails when the log holds no memcheck output at
# all. A caller that always runs memcheck wants that, because a log that lost
# it - a runner that died early, a flag that stopped matching - would otherwise
# be indistinguishable here from a clean one.
#

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "${SCRIPT_DIR}/../.." && pwd)
require_output=""
logs=()

while [ $# -gt 0 ]; do
  case "$1" in
    --repo-root)
      if [ $# -lt 2 ]; then
        echo "$0: --repo-root requires an argument" >&2
        exit 2
      fi
      repo_root=$2
      shift 2
      ;;
    --require-output)
      require_output=1
      shift
      ;;
    -h|--help)
      echo "usage: $(basename "$0") [--repo-root DIR] [--require-output] LOGFILE..." >&2
      exit 0
      ;;
    *)
      logs+=("$1")
      shift
      ;;
  esac
done

if [ ${#logs[@]} -eq 0 ]; then
  echo "$0: no log file given" >&2
  exit 2
fi

for log in "${logs[@]}"; do
  if [ ! -f "${log}" ]; then
    echo "$0: ${log}: no such file" >&2
    exit 2
  fi
done

if [ ! -d "${repo_root}" ]; then
  echo "$0: ${repo_root}: no such directory" >&2
  exit 2
fi

if ! sources=$(mktemp); then
  echo "$0: cannot create a temporary file" >&2
  exit 2
fi
trap 'rm -f "${sources}"' EXIT

find "${repo_root}" \
    -path "${repo_root}/.git" -prune -o \
    -path "${repo_root}/build" -prune -o \
    -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o \
               -name '*.h' -o -name '*.hh' -o -name '*.hpp' \) -print |
  sed 's|.*/||' | sort -u > "${sources}"

# Without this list every frame looks like a library's, and the check would
# pass whatever it was given.
if [ ! -s "${sources}" ]; then
  echo "$0: ${repo_root}: no source file found; is that the repository root?" >&2
  exit 2
fi

awk -v sources="${sources}" -v require_output="${require_output}" '
BEGIN {
  while ((getline name < sources) > 0)
    ours_source[name] = 1
  close(sources)
  binary = "(unknown)"
}

# Errors are reported by the frame memcheck blames, so skip the wrappers it
# substitutes for the allocator and for the string functions.
function is_valgrind_own(where) {
  return where ~ /^vg_replace_/
}

function is_ours(where,   file) {
  if (where ~ /^in /) {
    return (where ~ /\/build\/(gst|ext|tests)\//)
  }
  file = where
  sub(/:[0-9]+$/, "", file)
  return (file in ours_source)
}

function is_exempt_from_leaks(frame,   symbol) {
  symbol = frame
  sub(/^(at|by) 0x[0-9A-Fa-f]+: /, "", symbol)
  return symbol ~ /^(_dl_init|_dl_open|gst_debug_add_log_function|gst_debug_remove_log_function)[. ]/
}

function record_leak(what, where, frame,   key) {
  key = binary SUBSEP "definite leak" SUBSEP where
  if (key in seen)
    return
  seen[key] = 1
  leak_ours_count++
  leak_binary[leak_ours_count] = binary
  leak_what[leak_ours_count] = what
  leak_frame[leak_ours_count] = frame
}

function record(kind, where, frame,   key) {
  key = binary SUBSEP kind SUBSEP where
  if (key in seen)
    return
  seen[key] = 1
  if (is_ours(where)) {
    ours_count++
    ours_binary[ours_count] = binary
    ours_kind[ours_count] = kind
    ours_frame[ours_count] = frame
  } else {
    library_count++
  }
}

{
  line = $0
  sub(/^[0-9-]+T[0-9:.]+Z /, "", line)
  if (line !~ /^==[0-9]+== /)
    next
  sub(/^==[0-9]+== /, "", line)
}

line ~ /^Command: / {
  examined++
  binary = line
  sub(/^Command: /, "", binary)
  sub(/ .*$/, "", binary)
  sub(/.*\//, "", binary)
  next
}

line ~ /^[0-9,]+ (\([^)]*\) )?bytes in [0-9,]+ blocks are .*lost/ {
  leak_count++
  pending = (line ~ / are definitely lost/) ? "leak" : 0
  pending_kind = line
  sub(/ in loss record .*$/, "", pending_kind)
  candidate = ""
  next
}

# Memcheck writes these above a stack of their own; they are part of an error
# already counted, or an announcement, not a kind this script should classify.
# Ending the error here matters as much as not classifying it: the stack that
# follows is where the block was allocated, and blaming an error on that is
# how a bug in a library becomes ours.
line ~ /^ *(Address 0x|Uninitialised value was created|Block was alloc|Location 0x|At least one of|Process terminating|Thread [0-9]+|Access not within mapped region|Bad permissions for mapped region|General Protection Fault|Stack overflow in thread)/ {
  pending = 0
  candidate = ""
  next
}

line ~ /^(Invalid (read|write|free|memory pool|alignment)|Mismatched free|Conditional jump or move depends on uninitialised|Use of uninitialised|Syscall param |Source and destination overlap|Jump to the invalid address|Argument .* has a (fishy|bad)|realloc\(\) with size 0)/ {
  pending = "error"
  pending_kind = line
  sub(/ of size [0-9]+$/, "", pending_kind)
  candidate = ""
  next
}

line ~ /^ +(at|by) 0x[0-9A-Fa-f]+: / {
  if (candidate != "") {
    unclassified_count++
    unclassified[unclassified_count] = candidate
    candidate = ""
  }
  if (!pending)
    next
  where = line
  if (where ~ /\([^)]*\)$/) {
    sub(/^.*\(/, "", where)
    sub(/\)$/, "", where)
  } else {
    next
  }
  if (is_valgrind_own(where))
    next
  frame = line
  sub(/^ +/, "", frame)
  if (pending == "leak") {
    if (is_exempt_from_leaks(frame)) {
      pending = 0
    } else if (is_ours(where)) {
      record_leak(pending_kind, where, frame)
      pending = 0
    }
    next
  }
  record(pending_kind, where, frame)
  pending = 0
  next
}

{
  pending = 0
  candidate = (line ~ /^ *$/) ? "" : line
}

END {
  if (require_output != "" && examined == 0) {
    print "valgrind log check: the log holds no memcheck output at all."
    exit 1
  }
  printf "valgrind log check: %d error contexts from this repository, ", ours_count
  printf "%d from libraries, %d leak reports, ", library_count, leak_count
  printf "%d definite leak contexts from this repository, %d binaries examined\n", leak_ours_count, examined
  for (i = 1; i <= unclassified_count; i++)
    printf "::warning::valgrind log check does not classify this report, so it can neither pass nor fail on it: %s\n", unclassified[i]
  if (leak_ours_count > 0) {
    printf "::warning::valgrind log check found %d definite leak contexts allocated in this repository; they are listed in the log and do not fail the check.\n", leak_ours_count
    print ""
    print "Definite leaks allocated in this repository (reported, not failed on):"
    for (i = 1; i <= leak_ours_count; i++) {
      printf "  [%s] %s\n", leak_binary[i], leak_what[i]
      printf "      %s\n", leak_frame[i]
    }
  }
  if (ours_count == 0)
    exit 0
  print ""
  print "Memcheck errors reported in this repository'\''s own code:"
  for (i = 1; i <= ours_count; i++) {
    printf "  [%s] %s\n", ours_binary[i], ours_kind[i]
    printf "      %s\n", ours_frame[i]
  }
  exit 1
}
' "${logs[@]}"
