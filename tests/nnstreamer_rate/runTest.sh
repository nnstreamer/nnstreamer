#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-2.1-only
#
# @file runTest.sh
# @author Dongju Chae <dongju.chae@samsung.com>
# @date Oct 05 2020
# @brief SSAT Test Cases for NNStreamer
#

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

PATH_TO_PLUGIN="../../build"

# Dump original frames, passthrough
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=300 ! video/x-raw,width=640,height=480,framerate=30/1 ! tensor_converter ! multifilesink location=original_%1d.log" 1 0 0 $PERFORMANCE

# Adjust frame rates (30 --> 15), downscaled with frame drops
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=300 ! video/x-raw,width=640,height=480,framerate=30/1 ! tensor_converter ! tensor_rate framerate=15/1 throttle=false ! multifilesink location=result_%1d.log" 2 0 0 $PERFORMANCE

callCompareTest original_0.log result_0.log 2-1 "Compare 2-1" 1 0
callCompareTest original_150.log result_75.log 2-2 "Compare 2-2" 1 0
callCompareTest original_200.log result_100.log 2-3 "Compare 2-3" 1 0

# Adjust frame rates (30 --> 3), downscaled with frame drops
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=300 ! video/x-raw,width=640,height=480,framerate=30/1 ! tensor_converter ! tensor_rate framerate=3/1 throttle=false ! multifilesink location=result_%1d.log" 3 0 0 $PERFORMANCE

callCompareTest original_0.log result_0.log 3-1 "Compare 3-1" 1 0
callCompareTest original_150.log result_15.log 3-2 "Compare 3-2" 1 0
callCompareTest original_200.log result_20.log 3-3 "Compare 3-3" 1 0

# Adjust frame rates (30 --> 45), upscaled with frame duplication
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=300 ! video/x-raw,width=640,height=480,framerate=30/1 ! tensor_converter ! tensor_rate framerate=45/1 throttle=false ! multifilesink location=result_%1d.log" 4 0 0 $PERFORMANCE

callCompareTest original_0.log result_0.log 4-1 "Compare 4-1" 1 0
callCompareTest original_150.log result_225.log 4-2 "Compare 4-2" 1 0
callCompareTest original_200.log result_300.log 4-3 "Compare 4-3" 1 0

# Adjust frame rates (30 --> 60), upscaled with frame duplication
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=300 ! video/x-raw,width=640,height=480,framerate=30/1 ! tensor_converter ! tensor_rate framerate=60/1 throttle=false ! multifilesink location=result_%1d.log" 5 0 0 $PERFORMANCE

callCompareTest original_0.log result_0.log 5-1 "Compare 5-1" 1 0
callCompareTest original_150.log result_300.log 5-2 "Compare 5-2" 1 0
callCompareTest original_200.log result_400.log 5-3 "Compare 5-3" 1 0

rm original_*.log
rm result_*.log

report
