#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file run_unittests_binaries.sh
## @author Parichay Kapoor <pk.kapoor@gmail.com>
## @date Dec 20 2019
## @brief Runs all the unittests binaries in the specified folder or file

input=""
skip_tests=""
this_script="$(basename -- $0)"
VALGRIND=""
RECURSIVE=""
while (( "$#" )); do
  case "$1" in
    -k|--skip)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        tmp=$2
        shift 2
        readarray -td, skip_tests <<<"$tmp,"; unset 'skip_tests[-1]'; declare skip_tests; >&2
      else
        echo "$this_script: $1: option requires an argument" >&2
        exit 1
      fi
      ;;
    -h|--help)
      echo "$this_script: usage: $this_script [options] target" >&2
      echo "    -k | --skip  BINARY_NAME[,*]" >&2
      echo "        Skip the test cases whose names are...(valid only if target is a directory)" >&2
      echo "    -r | --recursive" >&2
      echo "        Look in sub-directories too (valid only if target is a directory)" >&2
      exit 0
      ;;
    -r|--recursive)
      RECURSIVE="1"
      shift 1
      ;;
    --valgrind)
      VALGRIND="valgrind"
      shift 1
      ;;
    -*|--*)
      echo "$1: invalid option" >&2
      exit 1
      ;;
    *)
      input=$1
      shift 1
      ;;
  esac
done

[[ -z "$input" ]] && echo "$this_script: target should be given" && exit 1
export NNSTREAMER_SOURCE_ROOT_PATH=$(pwd)
pushd build
export NNSTREAMER_BUILD_ROOT_PATH=$(pwd)
export NNSTREAMER_CONF=${NNSTREAMER_BUILD_ROOT_PATH}/nnstreamer-test.ini
export NNSTREAMER_FILTERS=${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer/tensor_filter
export NNSTREAMER_DECODERS=${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer/tensor_decoder
export NNSTREAMER_CONVERTERS=${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer/tensor_converter
export NNSTREAMER_TRAINERS=${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer/tensor_trainer
export _PYTHONPATH=${PYTHONPATH}

run_entry() {
  entry=$1
  if [[ $entry == *"python3"* || $entry == *"unittest_converter"* ]]; then
    PY="python3"
    pushd ext/nnstreamer/extra
    TEST_PYTHONPATH=${PY}_module
    rm -rf ${TEST_PYTHONPATH}
    mkdir -p ${TEST_PYTHONPATH}
    pushd ${TEST_PYTHONPATH}
    # Covert to an absolute path from the relative path
    TEST_PYTHONPATH=$(pwd)
    export PYTHONPATH=${TEST_PYTHONPATH}
    if [[ ! -f ${TEST_PYTHONPATH}/nnstreamer_python.so ]]; then
      ln -sf ../nnstreamer_${PY}.so nnstreamer_python.so
    fi
    popd
    popd
  fi

  if [[ "$VALGRIND" == "valgrind" ]]; then
    valgrind -v --suppressions=../tools/debugging/valgrind_suppression --track-origins=yes --tool=memcheck --num-callers=200 --leak-check=full ${entry} --gtest_output="xml:${entry##*/}.xml"
  else
    ${entry} --gtest_output="xml:${entry##*/}.xml"
  fi

  retval=$?
  export PYTHONPATH=${_PYTHONPATH}

  return ${retval}
}

##
# @brief Tell an actual test binary from a file that merely matches the name.
# @param $1 path to the candidate
# Meson copies generated test sources next to the binaries it builds them into,
# and a checkout whose files carry the executable bit leaves those copies
# executable as well, so the name and the mode alone do not identify a binary.
is_test_binary() {
  [ -f "$1" ] && [ -x "$1" ] || return 1
  # The first four bytes of an ELF file are 0x7f 'E' 'L' 'F'.
  [ "$(od -A n -N 4 -t x1 < "$1" | tr -d ' ')" = "7f454c46" ]
}

##
# @brief Run every entry of the given list, and remember the ones that failed.
# @param $@ the test binaries to run
# The list is run to the end even after a failure: stopping at the first one
# would leave every binary after it unexamined, and which binaries those are
# depends on the order the file system hands them over.
run_list() {
  local entry
  for entry in "$@"
  do
    if ! is_test_binary "${entry}"; then
      continue
    fi
    run_entry ${entry}
    entry_ret=$?
    if [ ${entry_ret} -ne 0 ]; then
      failed_entries+=("${entry}")
      ret=${entry_ret}
    fi
  done
}

ret=0
failed_entries=()
if [ -f "${input}" ]; then
  run_entry $input
  ret=$?
elif [ -d "${input}" ]; then
  # With --recursive, sub-directories are searched too: several test binaries
  # are built one level down (tests/cpp_methods, tests/nnstreamer_datarepo,
  # ...) and a search that stops at the top level never sees them. It is not
  # the default because callers that already walk those sub-directories
  # themselves, as nnstreamer.spec does, would otherwise run them twice and
  # without the library paths they set up for each one. The list is sorted so
  # that a run is reproducible rather than following directory order.
  #
  # Recursion relies on the naming convention: an executable called unittest_*
  # is a gtest binary that runs to completion on its own. A helper program that
  # an SSAT script drives - one that builds a pipeline and waits on a main loop
  # - must not be given that name, or it will be started here with no arguments
  # and hang until the caller's timeout. tensor_repo_dynamic_test and
  # tensor_filter_reload_test were named unittest_* once and did exactly that.
  if [ -n "${RECURSIVE}" ]; then
    depth_args=""
  else
    depth_args="-maxdepth 1"
  fi
  filelist=(`find "${input}" -mindepth 1 ${depth_args} -type f -executable $(for stest in "${skip_tests[@]}"; do [[ ! -z ${stest} ]] && echo -n "! -name ${stest} "; done) -name "unittest_*" | sort`)
  run_list "${filelist[@]}"
else
  filename=${input##*/}
  dirname=${input%/*}
  filelist=(`find "${dirname}" -mindepth 1 -maxdepth 1 -type f -executable -name "${filename}" | sort`)
  run_list "${filelist[@]}"
fi

if [ ${#failed_entries[@]} -ne 0 ]; then
  echo "$this_script: these test binaries reported a failure:" >&2
  for entry in "${failed_entries[@]}"; do echo "  ${entry}" >&2; done
fi

popd
exit $ret
