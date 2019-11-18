#!/usr/bin/env bash
##
## @file runTest.sh
## @author MyungJoo Ham <myungjoo.ham@gmail.com>
## @date Nov 01 2018
## @brief SSAT Test Cases for NNStreamer
##
if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}
"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

# Test constant passthrough decoder (1, 2)
PATH_TO_PLUGIN="../../build"
CASESTART=0
CASEEND=1

# THIS SHOULD EMIT ERROR
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=RGB ! tensor_converter ! tensor_split name=a tensorseg=1:640:480:1,2:640:480:1 a.src_0 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_decoder mode=pose_estimation option1=320:240 option2=640:480 ! fakesink" 0_n 0 1 $PERFORMANCE

# THIS WON'T FAIL, BUT NOT MUCH MEANINGFUL.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num_buffers=4 ! videoconvert ! videoscale ! video/x-raw,width=14,height=14,format=RGB ! tensor_converter ! tensor_split name=a tensorseg=1:14:14:1,2:14:14:1 a.src_0 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_decoder mode=pose_estimation option1=320:240 option2=14:14 ! fakesink" 1 0 0 $PERFORMANCE

report
