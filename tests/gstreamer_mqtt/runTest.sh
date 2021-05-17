#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author HyoungJoo Ahn <hello.ahn@samsung.com>
## @date May 17 2021
## @brief SSAT Test Cases for gstreamer element mqttsink & mqttsrc
##
if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

PATH_TO_PLUGIN="../../build"

# Compare the published result of 'mqttsink' to 'mqttsrc' results.
TEST_TOPIC=TEST1
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} mqttsrc sub-topic=$TEST_TOPIC sub-timeout=1000000 ! multifilesink location="src_%1d.out" &
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc name=vsrc num-buffers=5 pattern=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=160,height=120,framerate=10/1 ! tee name=t t. ! queue ! mqttsink pub-topic=$TEST_TOPIC t. ! queue ! multifilesink location="sink_%1d.out"

sleep 1.5
callCompareTest sink_0.out src_0.out 1-1 "Compare 1-1" 1 0
callCompareTest sink_1.out src_1.out 1-2 "Compare 1-2" 1 0
callCompareTest sink_2.out src_2.out 1-3 "Compare 1-3" 1 0
callCompareTest sink_3.out src_3.out 1-4 "Compare 1-4" 1 0
callCompareTest sink_4.out src_4.out 1-5 "Compare 1-5" 1 0
rm *.out

report
