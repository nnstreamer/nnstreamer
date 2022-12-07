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
      exit 0
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

  ${entry} --gtest_output="xml:${entry##*/}.xml"
  retval=$?
  export PYTHONPATH=${_PYTHONPATH}

  return ${retval}
}

ret=0
if [ -f "${input}" ]; then
  run_entry $input
  ret=$?
elif [ -d "${input}" ]; then
  filelist=(`find "${input}" -mindepth 1 -maxdepth 1 -type f -executable $(for stest in "${skip_tests[@]}"; do [[ ! -z ${stest} ]] && echo -n "! -name ${stest} "; done) -name "unittest_*"`)
  for entry in "${filelist[@]}"
  do
    run_entry $entry
    ret=$?
    if [ $ret -ne 0 ]; then
      break
    fi
  done
else
  filename=${input##*/}
  dirname=${input%/*}
  filelist=(`find "${dirname}" -mindepth 1 -maxdepth 1 -type f -executable -name "${filename}"`)
  for entry in "${filelist[@]}"
  do
    run_entry $entry
    ret=$?
    if [ $ret -ne 0 ]; then
      break
    fi
  done
fi

popd
exit $ret
