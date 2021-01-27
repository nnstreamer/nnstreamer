#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
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

PATH_TO_PLUGIN="../../build"

if [ "$SKIPGEN" == "YES" ]; then
    echo "Test Case Generation Skipped"
    sopath=$2
else
    echo "Test Case Generation Started"
    python generateGoldenTestResult.py
    python ../nnstreamer_converter/generateGoldenTestResult.py 8
    sopath=$1
fi
convertBMP2PNG

function do_test() {
    param="${4}_gen_golden_data"
    gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase01_${1}_${2}x${3}.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=${1},width=${2},height=${3},framerate=0/1 ! filesink location=\"testcase01_${1}_${2}x${3}.golden.raw\" sync=true" $param 0 0 $PERFORMANCE

    gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase01_${1}_${2}x${3}.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=${1},width=${2},height=${3},framerate=0/1 ! tensor_converter ! tensor_decoder mode=direct_video ! filesink location=\"testcase01_${1}_${2}x${3}.log\" sync=true" ${4} 0 0 $PERFORMANCE

    callCompareTest testcase01_${1}_${2}x${3}.golden.raw testcase01_${1}_${2}x${3}.log ${4} "Golden Test ${4}" 1 0
}

do_test RGB 640 480 1
do_test RGB 642 480 1-1
do_test BGRx 640 480 1-2
do_test BGRx 642 480 1-3

# Test with a stream of 10 small PNG frames
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_decoder mode=direct_video ! filesink location=\"testcase2.dec.log\" sync=true t. ! queue ! filesink location=\"testcase2.con.log\" sync=true" 2 0 0 $PERFORMANCE
callCompareTest testcase2.con.log testcase2.dec.log 2 "Compare for case 2" 0 0

# Expand the unit test coverage
## Set Property woth option*
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_decoder mode=direct_video option1=\"nothing\" option2=\"else\" option3=\"matters\" option4=\"whattheheck=is\" option5=\"goingon=idontknow\" option6=\"whydowehave\" option7=\"somany\" option8=\"options\" option9=\"whydontyouguess\" ! filesink location=\"testcase3.dec.log\" sync=true t. ! queue ! filesink location=\"testcase3.con.log\" sync=true" 3 0 0 $PERFORMANCE
callCompareTest testcase3.con.log testcase3.dec.log 3 "Compare for case 3" 0 0

## Trigger "decoder not configured"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc ! tensor_converter ! tensor_decoder mode=direct_video ! video/mpeg ! fakesink" 4F_n 0 1 $PERFORMANCE

# Dynamic tensor config update test
## After converting the flatbuf to the tensor, it is tested whether the config is updated in runtime.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! video/x-raw, width=640, height=480, framerate=5/1,format=RGB ! \
    tee name=t ! queue ! multifilesink location=\"test_05_raw_%1d.log\"
    t. ! queue ! tensor_converter ! tensor_decoder mode=flatbuf ! other/flatbuf-tensor ! tensor_converter ! \
        tensor_decoder mode=direct_video ! multifilesink location=\"test_05_decoded_%1d.log\" sync=true" 5 "Flatbuf dynamic tensor config update" 0 0 $PERFORMANCE
callCompareTest test_05_raw_0.log test_05_decoded_0.log 5-1 "Compare for case 5-1" 1 0
callCompareTest test_05_raw_1.log test_05_decoded_1.log 5-2 "Compare for case 5-2" 1 0
callCompareTest test_05_raw_2.log test_05_decoded_2.log 5-3 "Compare for case 5-3" 1 0

## After converting the protobuf to the tensor, it is tested whether the config is updated in runtime.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! videoconvert ! video/x-raw, width=640, height=480, framerate=5/1,format=RGB ! \
    tee name=t ! queue ! multifilesink location=\"test_06_raw_%1d.log\"
    t. ! queue ! tensor_converter ! tensor_decoder mode=protobuf ! other/protobuf-tensor ! tensor_converter ! \
        tensor_decoder mode=direct_video ! multifilesink location=\"test_06_decoded_%1d.log\" sync=true" 6 "Protobuf dynamic tensor config update" 0 0 $PERFORMANCE
callCompareTest test_06_raw_0.log test_06_decoded_0.log 6-1 "Compare for case 6-1" 1 0
callCompareTest test_06_raw_1.log test_06_decoded_1.log 6-2 "Compare for case 6-2" 1 0
callCompareTest test_06_raw_2.log test_06_decoded_2.log 6-3 "Compare for case 6-3" 1 0

rm *.log

report
