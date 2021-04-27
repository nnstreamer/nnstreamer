#!/bin/bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file run_unittests.sh
## @author Parichay Kapoor <pk.kapoor@gmail.com>
## @date Dec 20 2019
## @brief Runs all the unittests binaries in the specified folder or file
input=$1

export NNSTREAMER_SOURCE_ROOT_PATH=$(pwd)
pushd build
export NNSTREAMER_BUILD_ROOT_PATH=$(pwd)
export NNSTREAMER_CONF=${NNSTREAMER_BUILD_ROOT_PATH}/nnstreamer-test.ini
export NNSTREAMER_FILTERS=${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer/tensor_filter
export NNSTREAMER_DECODERS=${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer/tensor_decoder
export NNSTREAMER_CONVERTERS=${NNSTREAMER_BUILD_ROOT_PATH}/ext/nnstreamer/tensor_converter
export _PYTHONPATH=${PYTHONPATH}

run_entry() {
  entry=$1
  if [[ $entry == *"python3"* || $entry == *"python2"* ]]; then
    PY=$(echo ${entry} | grep -oP "python[0-9]")
    pushd ext/nnstreamer/tensor_filter
    TEST_PYTHONPATH=${PY}_module
    rm -rf ${TEST_PYTHONPATH}
    mkdir -p ${TEST_PYTHONPATH}
    pushd ${TEST_PYTHONPATH}
    # Covert to an absolute path from the relative path
    TEST_PYTHONPATH=$(pwd)
    export PYTHONPATH=${TEST_PYTHONPATH}
    if [[ ! -f ${TEST_PYTHONPATH}/nnstreamer_python.so ]]; then
      ln -sf ../../extra/nnstreamer_${PY}.so nnstreamer_python.so
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
  filelist=(`find "${input}" -mindepth 1 -maxdepth 1 -type f -executable -name "unittest_*"`)
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
