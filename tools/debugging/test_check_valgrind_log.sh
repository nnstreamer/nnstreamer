#!/usr/bin/env bash

##
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# SPDX-License-Identifier: LGPL-2.1-only
#
# @file     test_check_valgrind_log.sh
# @brief    Self-test for check_valgrind_log.sh.
# @see      https://github.com/nnstreamer/nnstreamer
# @author   MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Each case is a hand-written memcheck log against a throwaway source tree, so
# that what the checker is supposed to key on - the file a frame names, the
# object a frame names, whether the error is a leak - is the only thing that
# varies between them.
#
# The cases that expect a pass also assert the counts in the summary line. A
# checker that parsed nothing at all would exit 0 for every log, and without
# those counts every one of them would agree with it.
#

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CHECKER="${SCRIPT_DIR}/check_valgrind_log.sh"
workdir=$(mktemp -d)
repo="${workdir}/repo"
failed=0

trap 'rm -rf "${workdir}"' EXIT

mkdir -p "${repo}/gst"
touch "${repo}/gst/myelement.c" "${repo}/gst/myelement.h"

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
# @brief Run the checker on a log written from stdin.
# @param $1 name of the case, used for the log file
# @param $@ further options handed to the checker
# Sets `status` and `output`.
run_case() {
  local name=$1
  local log="${workdir}/${name}.log"
  shift
  cat > "${log}"
  output=$(bash "${CHECKER}" --repo-root "${repo}" "$@" "${log}" 2>&1)
  status=$?
}

##
# @brief Assert the checker's exit status.
# @param $1 expected status
# @param $2 description of the expectation
expect_status() {
  if [ "${status}" -eq "$1" ]; then
    report 0 "$2"
  else
    report 1 "$2 (got status ${status}: ${output})"
  fi
}

##
# @brief Assert that the checker's output contains the given text.
# @param $1 text that must appear
# @param $2 description of the expectation
expect_in() {
  case "${output}" in
    *"$1"*) report 0 "$2" ;;
    *) report 1 "$2 (output: ${output})" ;;
  esac
}

##
# @brief Assert that the checker's output does not contain the given text.
# @param $1 text that must not appear
# @param $2 description of the expectation
expect_not_in() {
  case "${output}" in
    *"$1"*) report 1 "$2 (output: ${output})" ;;
    *) report 0 "$2" ;;
  esac
}

if [ ! -f "${CHECKER}" ]; then
  echo "FAIL: ${CHECKER} is missing."
  exit 1
fi

run_case ours <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invalid read of size 4
==1==    at 0x1111: my_element_chain (myelement.c:42)
==1==    by 0x2222: g_closure_invoke (in /usr/lib/libgobject-2.0.so.0)
==1==
EOF
expect_status 1 "an error reported in a source of this repository fails the check"
expect_in "myelement.c:42" "the failing error is printed with its frame"
expect_in "unittest_demo" "the failing error is printed with its binary"

run_case library <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invalid read of size 8
==1==    at 0x3333: strncmp (strcmp.S:172)
==1==    by 0x4444: my_element_chain (myelement.c:42)
==1==
EOF
expect_status 0 "an error blamed on a library passes even when we called it"
expect_in "1 from libraries" "the library error is still counted"

run_case wrapper_then_ours <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invalid write of size 1
==1==    at 0x5555: malloc (vg_replace_malloc.c:381)
==1==    by 0x6666: my_element_init (myelement.c:11)
==1==
EOF
expect_status 1 "valgrind's own wrapper is skipped when attributing an error"
expect_in "myelement.c:11" "attribution lands on the first frame that is not the wrapper"

run_case wrapper_then_library_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invalid write of size 1
==1==    at 0x5555: malloc (vg_replace_malloc.c:381)
==1==    by 0x6666: g_malloc (in /usr/lib/libglib-2.0.so.0)
==1==
EOF
expect_status 0 "skipping the wrapper does not by itself make an error ours"
expect_in "1 from libraries" "that error is counted as a library one"

run_case object_ours <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invalid read of size 4
==1==    at 0x7777: some_symbol (in /home/runner/work/nnstreamer/build/gst/libnnstreamer.so)
==1==
EOF
expect_status 1 "a frame naming an object under the build directory is ours"

run_case object_library_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invalid read of size 4
==1==    at 0x7777: some_symbol (in /usr/lib/x86_64-linux-gnu/libglib-2.0.so.0)
==1==
EOF
expect_status 0 "a frame naming a system object is not ours"
expect_in "1 from libraries" "that error is counted as a library one"

run_case leak <<'EOF'
==1== Command: ./tests/unittest_demo
==1== 400 bytes in 1 blocks are definitely lost in loss record 3 of 9
==1==    at 0x8888: my_element_init (myelement.c:11)
==1==
EOF
expect_status 0 "a leak allocated by this repository does not fail the check"
expect_in "1 leak reports" "the leak is counted"
expect_in "1 definite leak contexts from this repository" "the leak is counted as ours"
expect_in "::warning::valgrind log check found 1 definite leak" "a leak of ours is warned about"
expect_in "[unittest_demo] 400 bytes in 1 blocks are definitely lost" "the leak is listed with its binary and size"
expect_in "myelement.c:11" "the leak is listed with the frame of ours"
expect_not_in "Memcheck errors reported" "the leak is not reported as a failing error"

# Memcheck reports a leak where the block was allocated, which is usually a
# library; the leak is still ours when a frame of ours made that call.
run_case leak_through_library <<'EOF'
==1== Command: ./tests/unittest_demo
==1== 32 (16 direct, 16 indirect) bytes in 1 blocks are definitely lost in loss record 7 of 9
==1==    at 0x1111: malloc (vg_replace_malloc.c:381)
==1==    by 0x2222: g_malloc (in /usr/lib/libglib-2.0.so.0)
==1==    by 0x3333: g_strdup (in /usr/lib/libglib-2.0.so.0)
==1==    by 0x4444: my_element_init (myelement.c:11)
==1==    by 0x5555: my_element_class_init (myelement.c:90)
==1==
EOF
expect_status 0 "a leak of ours allocated inside a library does not fail the check"
expect_in "1 definite leak contexts from this repository" "a leak allocated in a library for us is ours"
expect_in "myelement.c:11" "the leak is listed with the innermost frame of ours"
expect_not_in "myelement.c:90" "and not with any frame below it"
expect_in "32 (16 direct, 16 indirect) bytes in 1 blocks are definitely lost" "a leak holding indirect blocks is listed with its sizes"
expect_not_in "loss record" "the loss record number is not part of the listing"

run_case leak_object_ours <<'EOF'
==1== Command: ./tests/unittest_demo
==1== 64 bytes in 1 blocks are definitely lost in loss record 7 of 9
==1==    at 0x1111: malloc (vg_replace_malloc.c:381)
==1==    by 0x2222: g_malloc (in /usr/lib/libglib-2.0.so.0)
==1==    by 0x3333: some_symbol (in /home/runner/work/nnstreamer/build/gst/libnnstreamer.so)
==1==    by 0x4444: my_test_body (myelement.c:50)
==1==
EOF
expect_in "1 definite leak contexts from this repository" "a leak through an object under the build directory is ours"
expect_in "libnnstreamer.so" "and is listed with that frame, the innermost of ours"
expect_not_in "myelement.c:50" "not with the source frame below it"

run_case leak_object_library_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== 64 bytes in 1 blocks are definitely lost in loss record 7 of 9
==1==    at 0x1111: malloc (vg_replace_malloc.c:381)
==1==    by 0x2222: some_symbol (in /opt/vendor/build/lib/libvendor.so)
==1==
EOF
expect_in "0 definite leak contexts from this repository" "a leak through an object merely built somewhere is not ours"

run_case leak_library_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== 64 bytes in 1 blocks are definitely lost in loss record 7 of 9
==1==    at 0x1111: malloc (vg_replace_malloc.c:381)
==1==    by 0x2222: g_malloc (in /usr/lib/libglib-2.0.so.0)
==1==    by 0x3333: some_library_call (in /usr/lib/libfoo.so.1)
==1==
EOF
expect_status 0 "a leak with no frame of ours passes"
expect_in "1 leak reports" "that leak is counted"
expect_in "0 definite leak contexts from this repository" "but it is not ours"
expect_not_in "::warning::" "and nothing is warned about"

# A library allocates while the dynamic loader runs its initialisers; that
# block belongs to the library even when a frame of ours asked for the load.
run_case leak_loader_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== 72 bytes in 1 blocks are definitely lost in loss record 7 of 9
==1==    at 0x1111: malloc (vg_replace_malloc.c:381)
==1==    by 0x2222: ???
==1==    by 0x3333: call_init.part.0 (dl-init.c:70)
==1==    by 0x4444: _dl_init (dl-init.c:117)
==1==    by 0x5555: dl_open_worker (dl-open.c:808)
==1==    by 0x6666: _dl_open (dl-open.c:883)
==1==    by 0x7777: g_module_open_full (in /usr/lib/libgmodule-2.0.so.0)
==1==    by 0x8888: my_element_open (myelement.c:30)
==1==
EOF
expect_status 0 "a leak from a library initialiser passes"
expect_in "0 definite leak contexts from this repository" "a leak made while loading a library is not ours"
expect_not_in "myelement.c:30" "the frame of ours that asked for the load is not listed"

run_case leak_loader_bookkeeping_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== 0 bytes in 1 blocks are definitely lost in loss record 1 of 9
==1==    at 0x1111: malloc (vg_replace_malloc.c:381)
==1==    by 0x2222: malloc (rtld-malloc.h:56)
==1==    by 0x2222: _dl_find_object_update (dl-find_object.c:791)
==1==    by 0x3333: dl_open_worker_begin (dl-open.c:735)
==1==    by 0x4444: dl_open_worker (dl-open.c:782)
==1==    by 0x5555: _dl_open (dl-open.c:883)
==1==    by 0x6666: dlopen_doit (dlopen.c:56)
==1==    by 0x7777: my_element_open (myelement.c:30)
==1==
EOF
expect_in "0 definite leak contexts from this repository" "the bookkeeping of the loader itself is not ours"

run_case leak_log_function_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== 32 (16 direct, 16 indirect) bytes in 1 blocks are definitely lost in loss record 7 of 9
==1==    at 0x1111: malloc (vg_replace_malloc.c:381)
==1==    by 0x2222: g_malloc (in /usr/lib/libglib-2.0.so.0)
==1==    by 0x3333: g_slist_prepend (in /usr/lib/libglib-2.0.so.0)
==1==    by 0x4444: gst_debug_add_log_function (in /usr/lib/libgstreamer-1.0.so.0)
==1==    by 0x5555: my_test_body (myelement.c:50)
==1==
==1== 16 bytes in 1 blocks are definitely lost in loss record 5 of 9
==1==    at 0x1111: malloc (vg_replace_malloc.c:381)
==1==    by 0x2222: g_slist_copy (in /usr/lib/libglib-2.0.so.0)
==1==    by 0x3333: gst_debug_remove_log_function (in /usr/lib/libgstreamer-1.0.so.0)
==1==    by 0x5555: my_test_body (myelement.c:65)
==1==
EOF
expect_in "2 leak reports" "the list GStreamer leaks on purpose is counted"
expect_in "0 definite leak contexts from this repository" "but adding or removing a log function does not make it ours"

# The exemption covers what those callers allocate, not what they call.
run_case leak_ours_under_loader <<'EOF'
==1== Command: ./tests/unittest_demo
==1== 24 bytes in 1 blocks are definitely lost in loss record 7 of 9
==1==    at 0x1111: malloc (vg_replace_malloc.c:381)
==1==    by 0x2222: my_element_constructor (myelement.c:5)
==1==    by 0x3333: call_init (dl-init.c:70)
==1==    by 0x4444: _dl_init (dl-init.c:117)
==1==
EOF
expect_in "1 definite leak contexts from this repository" "an initialiser of ours run by the loader still leaks as ours"
expect_in "myelement.c:5" "and that initialiser is listed"

run_case leak_similar_name <<'EOF'
==1== Command: ./tests/unittest_demo
==1== 24 bytes in 1 blocks are definitely lost in loss record 7 of 9
==1==    at 0x1111: malloc (vg_replace_malloc.c:381)
==1==    by 0x2222: _dl_opener (in /usr/lib/libfoo.so.1)
==1==    by 0x3333: gst_debug_add_log_function_full (in /usr/lib/libgstreamer-1.0.so.0)
==1==    by 0x4444: my_element_init (myelement.c:11)
==1==
EOF
expect_in "1 definite leak contexts from this repository" "a function merely named like an exempt one exempts nothing"

run_case possible_leak_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== 400 bytes in 1 blocks are possibly lost in loss record 7 of 9
==1==    at 0x1111: calloc (vg_replace_malloc.c:1328)
==1==    by 0x2222: my_element_start (myelement.c:20)
==1==
==1== 16 bytes in 1 blocks are indirectly lost in loss record 2 of 9
==1==    at 0x1111: malloc (vg_replace_malloc.c:381)
==1==    by 0x2222: my_element_init (myelement.c:11)
==1==
EOF
expect_in "2 leak reports" "possible and indirect leaks are counted"
expect_in "0 definite leak contexts from this repository" "but neither is listed as a definite leak of ours"

# A definite leak with no frame of ours ends at the next report; the stack of
# that report must not be read as its continuation.
run_case leak_ends_at_next_report_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== 64 bytes in 1 blocks are definitely lost in loss record 7 of 9
==1==    at 0x1111: malloc (vg_replace_malloc.c:381)
==1==    by 0x2222: g_malloc (in /usr/lib/libglib-2.0.so.0)
==1==
==1== 400 bytes in 1 blocks are possibly lost in loss record 8 of 9
==1==    at 0x1111: calloc (vg_replace_malloc.c:1328)
==1==    by 0x2222: my_element_start (myelement.c:20)
==1==
==1== LEAK SUMMARY:
==1==    definitely lost: 64 bytes in 1 blocks
EOF
expect_in "0 definite leak contexts from this repository" "a definite leak does not borrow the stack of the next report"

run_case leak_repeated <<'EOF'
==1== Command: ./tests/unittest_demo
==1== 400 bytes in 1 blocks are definitely lost in loss record 3 of 9
==1==    at 0x1111: malloc (vg_replace_malloc.c:381)
==1==    by 0x2222: my_element_init (myelement.c:11)
==1==
==1== 800 bytes in 2 blocks are definitely lost in loss record 4 of 9
==1==    at 0x1111: calloc (vg_replace_malloc.c:1328)
==1==    by 0x2222: my_element_init (myelement.c:11)
==1==
==2== Command: ./tests/unittest_other
==2== 400 bytes in 1 blocks are definitely lost in loss record 3 of 9
==2==    at 0x1111: malloc (vg_replace_malloc.c:381)
==2==    by 0x2222: my_element_init (myelement.c:11)
==2==
EOF
expect_in "2 definite leak contexts from this repository" "a leak site counts once per binary"
expect_in "[unittest_other]" "the second binary is listed"

run_case leak_and_error <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invalid read of size 4
==1==    at 0x1111: my_element_chain (myelement.c:42)
==1==
==1== 400 bytes in 1 blocks are definitely lost in loss record 3 of 9
==1==    at 0x1111: malloc (vg_replace_malloc.c:381)
==1==    by 0x2222: my_element_init (myelement.c:11)
==1==
EOF
expect_status 1 "an error of ours still fails the check when leaks of ours are listed"
expect_in "1 error contexts from this repository" "the error is counted"
expect_in "1 definite leak contexts from this repository" "and so is the leak"
expect_in "myelement.c:42" "the failing error is printed"

run_case timestamped <<'EOF'
2026-09-07T07:21:27.4959062Z ==1== Command: ./tests/unittest_demo
2026-09-07T07:21:27.4959062Z ==1== Invalid read of size 4
2026-09-07T07:21:27.4959062Z ==1==    at 0x1111: my_element_chain (myelement.c:42)
2026-09-07T07:21:27.4959062Z ==1==
EOF
expect_status 1 "a log carrying the timestamps a CI run adds is still parsed"

run_case repeated <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invalid read of size 4
==1==    at 0x1111: my_element_chain (myelement.c:42)
==1==
==1== Invalid read of size 4
==1==    at 0x1111: my_element_chain (myelement.c:42)
==1==
EOF
expect_status 1 "a repeated error still fails the check"
expect_in "1 error contexts from this repository" "the same error in one binary is counted once"

run_case two_binaries <<'EOF'
==1== Command: ./tests/unittest_one
==1== Invalid read of size 4
==1==    at 0x1111: my_element_chain (myelement.c:42)
==1==
==2== Command: ./tests/unittest_two
==2== Invalid read of size 4
==2==    at 0x1111: my_element_chain (myelement.c:42)
==2==
EOF
expect_in "2 error contexts from this repository" "the same error in two binaries is counted twice"

run_case quiet <<'EOF'
[  PASSED  ] 12 tests.
EOF
expect_status 0 "a log with no memcheck output passes"
expect_in "0 error contexts" "and reports nothing"

run_case overlap <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Source and destination overlap in memcpy(0x1000, 0x1004, 12)
==1==    at 0x1111: my_element_chain (myelement.c:42)
==1==
EOF
expect_status 1 "an overlapping copy is an error like any other"

run_case fishy_argument <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Argument 'size' of function malloc has a fishy (possibly negative) value: -1
==1==    at 0x1111: my_element_init (myelement.c:11)
==1==
EOF
expect_status 1 "an allocation of a fishy size is an error like any other"

# --track-origins=yes appends the stack of the allocation the uninitialised
# value came from. That stack is often ours even when the error is not, so an
# error is decided by the frame memcheck blames and not by anything below it.
run_case origin_stack_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Conditional jump or move depends on uninitialised value(s)
==1==    at 0x1111: g_str_equal (in /usr/lib/libglib-2.0.so.0)
==1==  Uninitialised value was created by a heap allocation
==1==    at 0x2222: malloc (vg_replace_malloc.c:381)
==1==    by 0x3333: my_element_init (myelement.c:11)
==1==
EOF
expect_status 0 "the origin stack of an uninitialised value does not decide the error"
expect_in "1 from libraries" "the error is counted where memcheck blamed it"

run_case address_stack_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invalid read of size 4
==1==    at 0x1111: g_hash_table_lookup (in /usr/lib/libglib-2.0.so.0)
==1==  Address 0x5204040 is 0 bytes after a block of size 8 alloc'd
==1==    at 0x2222: malloc (vg_replace_malloc.c:381)
==1==    by 0x3333: my_element_init (myelement.c:11)
==1==
EOF
expect_status 0 "the stack of the block an address belongs to does not decide the error"
expect_in "1 from libraries" "that error is counted as a library one"

# The stack under a prelude line is where the block was allocated. If an error
# ends without a frame to blame - every frame a valgrind wrapper, or one with
# no location at all - that allocation stack must not be read as a
# continuation of it, or the code that allocated the block answers for a bug
# in the code that touched it.
run_case wrapper_only_then_prelude_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invalid read of size 4
==1==    at 0x1111: memcpy (vg_replace_strmem.c:1123)
==1==  Address 0x5204040 is 0 bytes after a block of size 8 alloc'd
==1==    at 0x2222: malloc (vg_replace_malloc.c:381)
==1==    by 0x3333: my_element_init (myelement.c:11)
==1==
EOF
expect_status 0 "an error with no frame to blame does not fall through to the next stack"
expect_not_in "myelement.c:11" "the allocating frame is not blamed for it"

run_case unlocatable_then_prelude_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Use of uninitialised value of size 8
==1==    at 0x1111: ???
==1==  Uninitialised value was created by a heap allocation
==1==    at 0x2222: malloc (vg_replace_malloc.c:381)
==1==    by 0x3333: my_element_init (myelement.c:11)
==1==
EOF
expect_status 0 "a frame with no location does not carry the error into the origin stack"
expect_not_in "myelement.c:11" "the allocating frame is not blamed for it either"

run_case build_elsewhere_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invalid read of size 4
==1==    at 0x7777: some_symbol (in /opt/vendor/build/lib/libvendor.so)
==1==
EOF
expect_status 0 "an object merely built somewhere is not ours"
expect_in "1 from libraries" "that error is counted as a library one"

run_case unknown_kind <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invented error nobody has seen before
==1==    at 0x1111: my_element_chain (myelement.c:42)
==1==
EOF
expect_status 0 "a report of a kind this checker does not know does not fail the check"
expect_in "::warning::" "but it is called out rather than passed over"
expect_in "Invented error" "and the report is quoted"

run_case known_prelude_n <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invalid read of size 4
==1==    at 0x1111: g_str_equal (in /usr/lib/libglib-2.0.so.0)
==1==  Address 0x5204040 is 0 bytes after a block of size 8 alloc'd
==1==    at 0x2222: malloc (vg_replace_malloc.c:381)
==1==    by 0x3333: my_element_init (myelement.c:11)
==1==
==1== Process terminating with default action of signal 6 (SIGABRT)
==1==    at 0x4444: raise (raise.c:51)
==1==
EOF
expect_not_in "::warning::" "the lines memcheck writes above a stack of their own are not reports"

run_case quiet_required --require-output <<'EOF'
[  PASSED  ] 12 tests.
EOF
expect_status 1 "--require-output rejects a log that lost its memcheck output"
expect_in "no memcheck output" "and says why"

run_case examined_required --require-output <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invalid read of size 8
==1==    at 0x3333: strncmp (strcmp.S:172)
==1==
EOF
expect_status 0 "--require-output passes a log that has memcheck output and no error of ours"
expect_in "1 binaries examined" "the binary count is reported"

output=$(bash "${CHECKER}" --repo-root "${repo}" "${workdir}/does-not-exist.log" 2>&1)
status=$?
expect_status 2 "a missing log file is a usage error, not a pass"

output=$(bash "${CHECKER}" --repo-root "${repo}" 2>&1)
status=$?
expect_status 2 "no log file at all is a usage error, not a pass"

# A root with no source in it leaves every frame looking like a library's, so
# the check would pass anything. It has to say so instead.
empty="${workdir}/empty-root"
mkdir -p "${empty}"
cat > "${workdir}/for-empty-root.log" <<'EOF'
==1== Command: ./tests/unittest_demo
==1== Invalid read of size 4
==1==    at 0x1111: my_element_chain (myelement.c:42)
==1==
EOF
output=$(bash "${CHECKER}" --repo-root "${empty}" "${workdir}/for-empty-root.log" 2>&1)
status=$?
expect_status 2 "a repository root holding no source is a usage error, not a pass"
expect_in "no source file found" "and says why"

if [ ${failed} -ne 0 ]; then
  echo "check_valgrind_log.sh self-test failed."
  exit 1
fi

echo "check_valgrind_log.sh self-test passed."
exit 0
