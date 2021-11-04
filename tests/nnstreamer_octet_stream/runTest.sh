#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Gichan Jang <gichan2.jang@samsung.com>
## @date Nov 08 2021
## @brief SSAT Test Cases for octet stream subplugin of tensor decoder
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

# Test static tensor
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num-buffers=3 pattern=13 ! video/x-raw,format=RGB,width=640,height=480,framerate=5/1 ! \
        tee name=t ! queue ! tensor_converter ! multifilesink location=octet_raw_1_%1d.log \
        t. ! queue ! tensor_converter ! tensor_decoder mode=octet_stream ! application/octet-stream ! multifilesink location=octet_decoded_1_%1d.log" 1 0 0 $PERFORMANCE
callCompareTest octet_raw_1_0.log octet_decoded_1_0.log 1-0 "tensor_decoder::octet_stream compare test 1-0" 0 0
callCompareTest octet_raw_1_1.log octet_decoded_1_1.log 1-1 "tensor_decoder::octet_stream compare test 1-1" 0 0
callCompareTest octet_raw_1_2.log octet_decoded_1_2.log 1-2 "tensor_decoder::octet_stream compare test 1-2" 0 0

# Test flexible tensor
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num-buffers=3 pattern=13 ! video/x-raw,format=RGB,width=640,height=480,framerate=5/1 ! \
        tee name=t ! queue ! tensor_converter ! multifilesink location=octet_raw_2_%1d.log \
        t. ! queue ! tensor_converter ! other/tensors,format=flexible ! tensor_decoder mode=octet_stream ! application/octet-stream,framerate=5/1 ! multifilesink location=octet_decoded_2_%1d.log" 2 0 0 $PERFORMANCE
callCompareTest octet_raw_2_0.log octet_decoded_2_0.log 2-0 "tensor_decoder::octet_stream compare test 2-0" 0 0
callCompareTest octet_raw_2_1.log octet_decoded_2_1.log 2-1 "tensor_decoder::octet_stream compare test 2-1" 0 0
callCompareTest octet_raw_2_2.log octet_decoded_2_2.log 2-2 "tensor_decoder::octet_stream compare test 2-2" 0 0

# Test static tensors
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num-buffers=3 pattern=13 ! video/x-raw,format=RGB,width=640,height=480,framerate=5/1 ! tensor_converter ! raw_tensor.sink_0 \
    videotestsrc num-buffers=3 pattern=18 ! video/x-raw,format=RGB,width=320,height=240,framerate=5/1 ! tensor_converter ! raw_tensor.sink_1 \
    tensor_mux name=raw_tensor ! \
        tee name=t ! queue ! multifilesink location=octet_raw_3_%1d.log \
        t. ! queue ! tensor_decoder mode=octet_stream ! application/octet-stream ! multifilesink location=octet_decoded_3_%1d.log" 3 0 0 $PERFORMANCE
callCompareTest octet_raw_3_0.log octet_decoded_3_0.log 3-0 "tensor_decoder::octet_stream compare test 3-0" 0 0
callCompareTest octet_raw_3_1.log octet_decoded_3_1.log 3-1 "tensor_decoder::octet_stream compare test 3-1" 0 0
callCompareTest octet_raw_3_2.log octet_decoded_3_2.log 3-2 "tensor_decoder::octet_stream compare test 3-2" 0 0

# Test flexible tensors
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num-buffers=3 pattern=13 ! video/x-raw,format=RGB,width=640,height=480,framerate=5/1 ! tensor_converter ! other/tensors,format=flexible ! raw_tensor.sink_0 \
    videotestsrc num-buffers=3 pattern=18 ! video/x-raw,format=RGB,width=320,height=240,framerate=5/1 ! tensor_converter ! other/tensors,format=flexible ! raw_tensor.sink_1 \
    tensor_mux name=raw_tensor ! \
        tee name=t ! queue ! multifilesink location=octet_raw_4_%1d.log \
        t. ! queue ! tensor_decoder mode=octet_stream ! application/octet-stream ! multifilesink location=octet_decoded_4_%1d.log" 4 0 0 $PERFORMANCE
# Compare with the results of test 3 (static tensors)
callCompareTest octet_raw_3_0.log octet_decoded_4_0.log 4-0 "tensor_decoder::octet_stream compare test 4-0" 0 0
callCompareTest octet_raw_3_1.log octet_decoded_4_1.log 4-1 "tensor_decoder::octet_stream compare test 4-1" 0 0
callCompareTest octet_raw_3_2.log octet_decoded_4_2.log 4-2 "tensor_decoder::octet_stream compare test 4-2" 0 0

rm *.log

report
