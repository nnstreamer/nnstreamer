#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Gichan Jang <gichan2.jang@samsung.com>
## @date June 08 2020
## @brief SSAT Test Cases for protobuf subplugin of tensor converter and decoder
## @details After decoding the tensor into protobuf, convert it to tensor(s) again to check if it matches the original
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
    python ../nnstreamer_converter/generateGoldenTestResult.py 9
    python3 ../nnstreamer_merge/generateTest.py
    sopath=$1
fi
convertBMP2PNG

PATH_TO_PLUGIN="../../build"

##
## @brief Execute gstreamer pipeline and compare the output of the pipeline
## @param $1 Colorspace
## @param $2 Width
## @param $3 Height
## @param $4 Test Case Number
function do_test() {
    gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 pattern=13 ! video/x-raw,format=${1},width=${2},height=${3},framerate=5/1 ! \
    tee name=t ! queue ! multifilesink location=\"raw_${1}_${2}x${3}_%1d.log\"
    t. ! queue ! tensor_converter ! tensor_decoder mode=protobuf ! other/protobuf-tensor ! tensor_converter ! multifilesink location=\"protob_${1}_${2}x${3}_%1d.log\" sync=true" ${4} 0 0 $PERFORMANCE

    callCompareTest raw_${1}_${2}x${3}_0.log protob_${1}_${2}x${3}_0.log "${4}-1" "protobuf conversion test ${4}-1" 1 0
    callCompareTest raw_${1}_${2}x${3}_1.log protob_${1}_${2}x${3}_1.log "${4}-2" "protobuf conversion test ${4}-2" 1 0
    callCompareTest raw_${1}_${2}x${3}_2.log protob_${1}_${2}x${3}_2.log "${4}-3" "protobuf conversion test ${4}-3" 1 0
}
# The width and height of video should be multiple of 4
do_test BGRx 320 240 1-1
do_test RGB 320 240 1-2
do_test GRAY8 320 240 1-3

# audio format S16LE, 8k sample rate, samples per buffer 8000
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} audiotestsrc num-buffers=1 samplesperbuffer=8000 ! audioconvert ! audio/x-raw,format=S16LE,rate=8000 ! \
    tee name=t ! queue ! audioconvert ! tensor_converter frames-per-tensor=8000 ! tensor_decoder mode=protobuf ! \
        other/protobuf-tensor ! tensor_converter ! filesink location=\"test.audio8k.s16le.log\" sync=true \
    t. ! queue ! filesink location=\"test.audio8k.s16le.origin.log\" sync=true" 2-1 0 0 $PERFORMANCE
callCompareTest test.audio8k.s16le.origin.log test.audio8k.s16le.log 2-2 "Audio8k-s16le Golden Test" 0 0

# audio format U8, 16k sample rate, samples per buffer 8000
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} audiotestsrc num-buffers=1 samplesperbuffer=8000 ! audioconvert ! audio/x-raw,format=U8,rate=16000 ! \
    tee name=t ! queue ! audioconvert ! tensor_converter frames-per-tensor=8000 ! tensor_decoder mode=protobuf ! \
        other/protobuf-tensor ! tensor_converter ! filesink location=\"test.audio16k.u8.log\" sync=true \
    t. ! queue ! filesink location=\"test.audio16k.u8.origin.log\" sync=true" 2-3 0 0 $PERFORMANCE
callCompareTest test.audio16k.u8.origin.log test.audio16k.u8.log 2-4 "Audio16k-u8 Golden Test" 0 0

# audio format U16LE, 16k sample rate, 2 channels, samples per buffer 8000
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} audiotestsrc num-buffers=1 samplesperbuffer=8000 ! audioconvert ! audio/x-raw,format=U16LE,rate=16000,channels=2 ! \
    tee name=t ! queue ! audioconvert ! tensor_converter frames-per-tensor=8000 ! tensor_decoder mode=protobuf ! \
        other/protobuf-tensor ! tensor_converter ! filesink location=\"test.audio16k2c.u16le.log\" sync=true \
    t. ! queue ! filesink location=\"test.audio16k2c.u16le.origin.log\" sync=true" 2-5 0 0 $PERFORMANCE
callCompareTest test.audio16k2c.u16le.origin.log test.audio16k2c.u16le.log 2-6 "Audio16k2c-u16le Golden Test" 0 0

# tensor merge test (The output is always in the format of other/tensor)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_merge name=merge mode=linear option=2 silent=true sync_mode=basepad sync_option=0:0 ! multifilesink location=testsynch08_%1d.log \
    multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)10/1\" ! pngdec ! tensor_converter !  \
        tensor_decoder mode=protobuf ! other/protobuf-tensor ! tensor_converter ! merge.sink_0 \
    multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! \
        tensor_decoder mode=protobuf ! other/protobuf-tensor ! tensor_converter ! merge.sink_1" 3 0 0 $PERFORMANCE
callCompareTest testsynch08_0.golden testsynch08_0.log 3-1 "Tensor merge Compare 3-1" 1 0
callCompareTest testsynch08_1.golden testsynch08_1.log 3-2 "Tensor merge Compare 3-2" 1 0
callCompareTest testsynch08_2.golden testsynch08_2.log 3-3 "Tensor merge Compare 3-3" 1 0
callCompareTest testsynch08_3.golden testsynch08_3.log 3-4 "Tensor merge Compare 3-4" 1 0

# tensor mux test (The output is always in the format of other/tensors)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_mux name=tensors_mux sync_mode=basepad sync_option=1:50000000 ! multifilesink location=testsynch19_%1d.log \
    tensor_mux name=tensor_mux0  sync_mode=slowest ! tensors_mux.sink_0 \
    tensor_mux name=tensor_mux1  sync_mode=slowest ! tensors_mux.sink_1 \
    multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)10/1\" ! pngdec ! \
        tensor_converter ! tensor_decoder mode=protobuf ! other/protobuf-tensor ! tensor_converter ! tensor_mux0.sink_0 \
    multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)20/1\" ! pngdec ! \
        tensor_converter ! tensor_decoder mode=protobuf ! other/protobuf-tensor ! tensor_converter ! tensor_mux0.sink_1 \
    multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! \
        tensor_converter ! tensor_decoder mode=protobuf ! other/protobuf-tensor ! tensor_converter ! tensor_mux1.sink_0 \
    multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)20/1\" ! pngdec ! \
        tensor_converter ! tensor_decoder mode=protobuf ! other/protobuf-tensor ! tensor_converter ! tensor_mux1.sink_1" 4 0 0 $PERFORMANCE
callCompareTest testsynch19_0.golden testsynch19_0.log 4-1 "Tensor mux Compare 4-1" 1 0
callCompareTest testsynch19_1.golden testsynch19_1.log 4-2 "Tensor mux Compare 4-2" 1 0
callCompareTest testsynch19_2.golden testsynch19_2.log 4-3 "Tensor mux Compare 4-3" 1 0
callCompareTest testsynch19_3.golden testsynch19_3.log 4-4 "Tensor mux Compare 4-4" 1 0
callCompareTest testsynch19_4.golden testsynch19_4.log 4-5 "Tensor mux Compare 4-5" 1 0

report
