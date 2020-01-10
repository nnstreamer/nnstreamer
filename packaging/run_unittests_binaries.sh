#!/bin/bash
##
## @file run_unittests.sh
## @author Parichay Kapoor <pk.kapoor@gmail.com>
## @date Dec 20 2019
## @brief Runs all the unittests binaries in the specified folder or file
input=$1

pushd build
export GST_PLUGIN_PATH=$(pwd)/gst/nnstreamer
export NNSTREAMER_CONF=$(pwd)/nnstreamer-test.ini
export NNSTREAMER_FILTERS=$(pwd)/ext/nnstreamer/tensor_filter
export NNSTREAMER_DECODERS=$(pwd)/ext/nnstreamer/tensor_decoder
export PYTHONPATH=$(pwd)/ext/nnstreamer/tensor_filter/:$PYTHONPATH


run_entry() {
  entry=$1

  if [[ $entry == *"python3"* ]]; then
    pushd ext/nnstreamer/tensor_filter
    ln -sf nnstreamer_python3.so nnstreamer_python.so
    popd
  elif [[ $entry == *"python2"* ]]; then
    pushd ext/nnstreamer/tensor_filter
    ln -sf nnstreamer_python2.so nnstreamer_python.so
    popd
  fi

  echo $entry
  ${entry} --gst-plugin-path=. --gtest_output="xml:${entry##*/}.xml"
}

if [ -f "${input}" ]; then
  run_entry $input
elif [ -d "${input}" ]; then
  filelist=(`find "${input}" -mindepth 1 -maxdepth 1 -type f -executable -name "unittest_*"`)
  for entry in "${filelist[@]}"
  do
    run_entry $entry
  done
else
  filename=${input##*/}
  dirname=${input%/*}
  filelist=(`find "${dirname}" -mindepth 1 -maxdepth 1 -type f -executable -name "${filename}"`)
  for entry in "${filelist[@]}"
  do
    run_entry $entry
  done
fi

popd
