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
    python ../nnstreamer_converter/generateGoldenTestResult.py 10
    sopath=$1
fi
convertBMP2PNG

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! tensor_demux name=demux filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 demux.src_0 !queue! filesink location=demux00.log" 1 0 0 $PERFORMANCE

callCompareTest testcase.golden demux00.log 1 "Golden Test 1" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! tensor_demux name=demux filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1 demux.src_0 ! queue ! filesink location=demux02_0.log demux.src_1 ! queue ! filesink location=demux02_1.log" 2 0 0 $PERFORMANCE

callCompareTest testcase.golden demux02_0.log 2_0 "Golden Test 2-0" 1 0
callCompareTest testcase.golden demux02_1.log 2_1 "Golden Test 2-1" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! tensor_demux name=demux filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_2 demux.src_0 ! queue ! filesink location=demux03_0.log demux.src_1 ! queue ! filesink location=demux03_1.log demux.src_2 ! queue ! filesink location=demux03_2.log" 3 0 0 $PERFORMANCE

callCompareTest testcase.golden demux03_0.log 3_0 "Golden Test 3-0" 1 0
callCompareTest testcase.golden demux03_1.log 3_1 "Golden Test 3-1" 1 0
callCompareTest testcase.golden demux03_2.log 3_2 "Golden Test 3-2" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! tensor_demux name=demux multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 demux.src_0 ! queue ! filesink location=demux04.log" 4 0 0 $PERFORMANCE

callCompareTest testcase_stream.golden demux04.log 4 "Golden Test 4" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! tensor_demux name=demux multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1 demux.src_0 ! queue ! filesink location=demux05_0.log demux.src_1 ! queue ! filesink location=demux05_1.log" 5 0 0 $PERFORMANCE

callCompareTest testcase_stream.golden demux05_0.log 5_0 "Golden Test 5-0" 1 0
callCompareTest testcase_stream.golden demux05_1.log 5_1 "Golden Test 5-1" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! tensor_demux name=demux multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_2 demux.src_0 ! queue ! filesink location=demux06_0.log demux.src_1 ! queue ! filesink location=demux06_1.log demux.src_2 ! queue ! filesink location=demux06_2.log" 6 0 0 $PERFORMANCE

callCompareTest testcase_stream.golden demux06_0.log 6_0 "Golden Test 6-0" 1 0
callCompareTest testcase_stream.golden demux06_1.log 6_1 "Golden Test 6-1" 1 0
callCompareTest testcase_stream.golden demux06_2.log 6_2 "Golden Test 6-2" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! tensor_demux name=demux multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_2 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_3 demux.src_0 ! queue ! filesink location=demux07_0.log demux.src_1 ! queue ! filesink location=demux07_1.log demux.src_2 ! queue ! filesink location=demux07_2.log demux.src_3 ! queue ! filesink location=demux07_3.log" 7 0 0 $PERFORMANCE

callCompareTest testcase_stream.golden demux07_0.log 7_0 "Golden Test " 1 0
callCompareTest testcase_stream.golden demux07_1.log 7_1 "Golden Test " 1 0
callCompareTest testcase_stream.golden demux07_2.log 7_2 "Golden Test " 1 0
callCompareTest testcase_stream.golden demux07_3.log 7_3 "Golden Test " 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! tensor_demux name=demux tensorpick=1 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_2 demux. ! queue ! filesink location=demux08_0.log" 8 0 0 $PERFORMANCE

callCompareTest testcase.golden demux08_0.log 8_0 "Golden Test 8-0" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! tensor_demux name=demux tensorpick=2 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_2 demux. ! queue ! filesink location=demux09_0.log" 9 0 0 $PERFORMANCE

callCompareTest testcase.golden demux09_0.log 9_0 "Golden Test 9-0" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! tensor_demux name=demux tensorpick=1 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_2 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_3 demux. ! queue ! filesink location=demux10_0.log" 10 0 0 $PERFORMANCE
callCompareTest testcase_stream.golden demux10_0.log 10_0 "Golden Test 10-0" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! tensor_demux name=demux tensorpick=2 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_2 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_3 demux. ! queue ! filesink location=demux11_0.log" 11 0 0 $PERFORMANCE
callCompareTest testcase_stream.golden demux11_0.log 11_0 "Golden Test 11-0" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! tensor_demux name=demux tensorpick=1,2 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_2 demux. ! queue ! filesink location=demux12_0.log demux. ! queue ! filesink location=demux12_1.log" 12 0 0 $PERFORMANCE

callCompareTest testcase.golden demux12_0.log 12_0 "Golden Test 12-0" 1 0
callCompareTest testcase.golden demux12_1.log 12_1 "Golden Test 12-1" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! tensor_demux name=demux tensorpick=0,2 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_2 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_3 demux. ! queue ! filesink location=demux13_0.log demux. ! queue ! filesink location=demux13_1.log" 13 0 0 $PERFORMANCE
callCompareTest testcase_stream.golden demux13_0.log 13_0 "Golden Test 13-0" 1 0
callCompareTest testcase_stream.golden demux13_1.log 13_1 "Golden Test 13-1" 1 0

# generate golden test result
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 pattern=13 ! videoconvert ! videoscale ! video/x-raw,width=200,height=200,format=RGB ! tensor_converter! filesink location=TestResult_14_0.golden
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 pattern=14 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter! filesink location=TestResult_14_1.golden
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 pattern=15 ! videoconvert ! videoscale ! video/x-raw,width=400,height=400,format=RGB ! tensor_converter! filesink location=TestResult_14_2.golden
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 pattern=16 ! videoconvert ! videoscale ! video/x-raw,width=500,height=500,format=RGB ! tensor_converter! filesink location=TestResult_14_3.golden

# other/tensors output test. input: tensor 0,1,2 output: tensor 0, tensor 1,2 and tensor 2,0
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! tensor_demux name=demux tensorpick=0,1:2,2:0 \
videotestsrc num-buffers=1 pattern = 13 ! videoconvert ! videoscale ! video/x-raw,width=200,height=200,format=RGB ! tensor_converter ! mux.sink_0 \
videotestsrc num-buffers=1 pattern = 14 ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB ! tensor_converter ! mux.sink_1 \
videotestsrc num-buffers=1 pattern = 15 ! videoconvert ! videoscale ! video/x-raw,width=400,height=400,format=RGB ! tensor_converter ! mux.sink_2 \
videotestsrc num-buffers=1 pattern = 16 ! videoconvert ! videoscale ! video/x-raw,width=500,height=500,format=RGB ! tensor_converter ! mux.sink_3 \
demux.src_0 ! queue ! filesink location=demux14_0.log \
demux.src_1 ! queue ! tensor_demux name=demux_1 demux_1.src_0 ! queue ! filesink location=demux14_1_1.log demux_1.src_1 ! queue ! filesink location=demux14_1_2.log \
demux.src_2 ! queue ! tensor_demux name=demux_2 demux_2.src_0 ! queue ! filesink location=demux14_2_2.log demux_2.src_1 ! queue ! filesink location=demux14_2_0.log" 14 0 0 $PERFORMANCE
callCompareTest TestResult_14_0.golden demux14_0.log 14_0 "Golden Test 14-0" 1 0
callCompareTest TestResult_14_1.golden demux14_1_1.log 14_1_1 "Golden Test 14-1" 1 0
callCompareTest TestResult_14_2.golden demux14_1_2.log 14_1_2 "Golden Test 14-2" 1 0
callCompareTest TestResult_14_2.golden demux14_2_2.log 14_2_2 "Golden Test 14-3" 1 0
callCompareTest TestResult_14_0.golden demux14_2_0.log 14_2_0 "Golden Test 14-4" 1 0

report
