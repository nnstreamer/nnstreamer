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

if [ -f "${input}" ]; then
  echo $input
  ${input} --gst-plugin-path=. --gtest_output="xml:${input##*/}.xml"
elif [ -d "${input}" ]; then
  filelist=(`find "${input}" -mindepth 1 -maxdepth 1 -type f -name "unittest_*"`)
  for entry in "${filelist[@]}"
  do
    echo $entry
    ${entry} --gst-plugin-path=. --gtest_output="xml:${entry##*/}.xml"
  done
else
  filename=${input##*/}
  dirname=${input%/*}
  filelist=(`find "${dirname}" -mindepth 1 -maxdepth 1 -type f -name "${filename}"`)
  for entry in "${filelist[@]}"
  do
    ${entry} --gst-plugin-path=. --gtest_output="xml:${entry##*/}.xml"
  done
fi

popd
