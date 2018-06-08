#!/usr/bin/env bash

##
# Copyright (C) 2018 Samsung Electronics
# License: Apache-2.0
#
# @file testcase01.sh
# @brief Test tensor_converter for testcase 01
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
# @dependency cmp

if [ $# -eq 0 ]
then
  PATH_TO_PLUGIN="$PWD/../../build/tensor_converter:$PWD/../../build/tensor_filter"
else
  PATH_TO_PLUGIN="$1"
fi

failed=0

# Generate one frame only (num-buffers=1)

gst-launch-1.0 -v --gst-debug=GST_CAPS:4 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter silent=FALSE ! tensor_filter debug=TRUE framework='tensorflow-lite' model='./nothing' input='4:280:40' inputtype='uint8' output='1' outputtype='uint8' ! filesink location="test.bgrx.log" sync=true || failed=1

if [ "$failed" -eq "0" ]
then
  echo Testcase 01: SUCCESS
  exit 0
else
  echo Testcase 01: FAILED
  exit -1
fi
