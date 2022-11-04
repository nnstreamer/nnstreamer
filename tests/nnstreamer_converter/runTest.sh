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
    printf "${Blue}Independent Mode${NC}"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

if [ "$SKIPGEN" == "YES" ]; then
    echo "Test Case Generation Skipped"
    sopath=$2
else
    echo "Test Case Generation Started"
    python3 generateGoldenTestResult.py
    sopath=$1
fi
convertBMP2PNG

PATH_TO_PLUGIN="../../build"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! tee name=t ! queue ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter silent=TRUE ! filesink location=\"test.bgrx.log\" sync=true t. ! queue ! filesink location=\"test.rgb.log\" sync=true" 1R 0 0 $PERFORMANCE

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=GRAY8,width=280,height=40,framerate=0/1 ! queue ! tensor_converter silent=TRUE ! filesink location=\"test.gray8.log\" sync=true" 1G 0 0 $PERFORMANCE

callCompareTest testcase01.bgrx.golden test.bgrx.log 1-1 "BGRX Golden Test" 1 0
callCompareTest testcase01.rgb.golden test.rgb.log 1-2 "RGB Golden Test" 1 0
callCompareTest testcase01.gray8.golden test.gray8.log 1-3 "Gray8 Golden Test" 1 0

##
## @brief Execute gstreamer pipeline and do golden-test with the output of the pipeline.
## @param $1 Colorspace
## @param $2 Width
## @param $3 Height
## @param $4 Test Case Number
function do_test() {
    gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase02_${1}_${2}x${3}.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=${1},width=${2},height=${3},framerate=0/1 ! tensor_converter silent=TRUE ! filesink location=\"testcase02_${1}_${2}x${3}.log\" sync=true" ${4} 0 0 $PERFORMANCE

    callCompareTest testcase02_${1}_${2}x${3}.golden testcase02_${1}_${2}x${3}.log ${4} "PNG Golden Testing ${4}" 1 0
}

do_test BGRx 640 480 2-1
do_test RGB 640 480 2-2
do_test GRAY8 640 480 2-3
do_test BGRx 642 480 2-4
do_test RGB 642 480 2-5
do_test GRAY8 642 480 2-6

# @TODO Change this when YUV becomes supported by tensor_converter
# Fail Test: YUV is given
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=YUV,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=YUV ! tensor_converter silent=TRUE ! filesink location=\"test.yuv.fail.log\" sync=true" 3F_n 0 1 $PERFORMANCE

# Fail Test: Unknown property is given
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter silent=TRUE whatthehell=isthis ! filesink location=\"test.yuv.fail.log\" sync=true" 4F_n 0 1 $PERFORMANCE

# audio format S16LE, 8k sample rate, samples per buffer 8000
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} audiotestsrc num-buffers=1 samplesperbuffer=8000 ! audioconvert ! audio/x-raw,format=S16LE,rate=8000 ! tee name=t ! queue ! audioconvert ! tensor_converter frames-per-tensor=8000 ! filesink location=\"test.audio8k.s16le.log\" sync=true t. ! queue ! filesink location=\"test.audio8k.s16le.origin.log\" sync=true" 5-1 0 0 $PERFORMANCE
callCompareTest test.audio8k.s16le.origin.log test.audio8k.s16le.log 5-2 "Audio8k-s16le Golden Test" 0 0

# audio format U8, 16k sample rate, samples per buffer 8000
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} audiotestsrc num-buffers=1 samplesperbuffer=8000 ! audioconvert ! audio/x-raw,format=U8,rate=16000 ! tee name=t ! queue ! audioconvert ! tensor_converter frames-per-tensor=8000 ! filesink location=\"test.audio16k.u8.log\" sync=true t. ! queue ! filesink location=\"test.audio16k.u8.origin.log\" sync=true" 5-3 0 0 $PERFORMANCE
callCompareTest test.audio16k.u8.origin.log test.audio16k.u8.log 5-4 "Audio16k-u8 Golden Test" 0 0

# audio format U16LE, 16k sample rate, 2 channels, samples per buffer 8000
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} audiotestsrc num-buffers=1 samplesperbuffer=8000 ! audioconvert ! audio/x-raw,format=U16LE,rate=16000,channels=2 ! tee name=t ! queue ! audioconvert ! tensor_converter frames-per-tensor=8000 ! filesink location=\"test.audio16k2c.u16le.log\" sync=true t. ! queue ! filesink location=\"test.audio16k2c.u16le.origin.log\" sync=true" 5-5 0 0 $PERFORMANCE
callCompareTest test.audio16k2c.u16le.origin.log test.audio16k2c.u16le.log 5-6 "Audio16k2c-u16le Golden Test" 0 0

# audio format S32LE, 8k sample rate, samples per buffer 8000
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} audiotestsrc num-buffers=1 samplesperbuffer=8000 ! audioconvert ! audio/x-raw,format=S32LE,rate=8000 ! tensor_converter frames-per-tensor=8000 ! filesink location=\"test.audio8k.s32le.log\" sync=true" 5-7 0 0 $PERFORMANCE

# Stream test case (genCase08 in generateGoldenTestResult.py)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! tensor_converter ! filesink location=\"testcase08.log\"" 6-1 0 0 $PERFORMANCE
callCompareTest testcase08.golden testcase08.log 6-2 "PNG Stream Test" 0 0

# Dimension declaration test cases
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! videoscale ! videoconvert ! video/x-raw,format=RGB,framerate=30/1,height=300,width=300 ! tensor_converter ! other/tensor,dimension=3:300:300,types=uint8,framerate=30/1,format=static ! fakesink" 7-1 0 0 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! videoscale ! videoconvert ! video/x-raw,format=RGB,framerate=30/1,height=300,width=300 ! tensor_converter ! other/tensors,num_tensors=1,dimension=3:300:300,types=uint8,framerate=30/1,format=static ! fakesink" 7-2 0 0 $PERFORMANCE

rm *.log *.bmp *.png *.golden

report
