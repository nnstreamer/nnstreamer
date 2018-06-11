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

PATH_TO_MODEL="../../build/nnstreamer_example/custom_example_passthrough/libnnstreamer_customfilter_passthrough.so"

failed=0

# Generate one frame only (num-buffers=1)

gst-launch-1.0 -v --gst-debug=GST_CAPS:4 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter silent=FALSE ! tee name=t ! queue ! tensor_filter debug=TRUE framework='custom' model="${PATH_TO_MODEL}" input='3:280:40' inputtype='uint8' output='3:280:40' outputtype='uint8' ! filesink location="testcase01.passthrough.log" sync=true t. ! queue ! filesink location="testcase01.direct.log" sync=true || failed=1

# Check the results (test.rgb.log, test.bgrx.log)
cmp testcase01.passthrough.log testcase01.direct.log || failed=1

if [ "$failed" -eq "0" ]
then
  echo Testcase 01: SUCCESS
  exit 0
else
  echo Testcase 01: FAILED
  exit -1
fi
