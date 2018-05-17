#!/usr/bin/env bash

##
# Copyright (C) 2018 Samsung Electronics
# License: Apache-2.0
#
# @file testcase02.sh
# @brief Test tensor_converter for testcase 02
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
# @dependency cmp

if [ $# -eq 0 ]
then
  PATH_TO_PLUGIN="$PWD/../../build"
else
  PATH_TO_PLUGIN="$1"
fi

failed=0

# Generate one frame only (num-buffers=1)
do_test()
{
  echo
  echo "Case 2 - $1 $2 x $3"
  echo
  echo gst-launch-1.0 -q --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase02_${1}_${2}x${3}.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=${1},width=${2},height=${3},framerate=0/1 ! tensor_converter ! filesink location="testcase02_${1}_${2}x${3}.log" sync=true
  gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase02_${1}_${2}x${3}.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=${1},width=${2},height=${3},framerate=0/1 ! tensor_converter ! filesink location="testcase02_${1}_${2}x${3}.log" sync=true || failed=1

  if [ "$failed" -eq "1" ]
  then
    echo gst-launch failed
  fi
  cmp -n `stat --printf="%s" testcase02_${1}_${2}x${3}.golden` testcase02_${1}_${2}x${3}.log testcase02_${1}_${2}x${3}.golden || failed=1
}

do_test BGRx 640 480
do_test RGB 640 480
do_test BGRx 642 480
do_test RGB 642 480

if [ "$failed" -eq "0" ]
then
  echo Testcase 02: SUCCESS
  exit 0
else
  echo Testcase 02: FAILED
  exit -1
fi
