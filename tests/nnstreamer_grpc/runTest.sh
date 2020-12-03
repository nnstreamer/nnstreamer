#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-2.1-only
#
# @file runTest.sh
# @author Dongju Chae <dongju.chae@samsung.com>
# @date Nov 04 2020
# @brief SSAT Test Cases for NNStreamer
#

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
NUM_BUFFERS=10

# Dump original frames, passthrough, other/tensor
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter ! multifilesink location=original_%1d.log" 0 0 0 $PERFORMANCE

PORT=`python3 get_available_port.py`
# tensor_sink (client) --> tensor_src (server), other/tensor
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=${NUM_BUFFERS} ! 'other/tensor,dimension=(string)3:640:480,type=(string)uint8,framerate=(fraction)5/1' ! multifilesink location=result_%1d.log" 1 0 0 $PERFORMANCE &
sleep 1
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter ! tensor_sink_grpc port=${PORT}" 1 0 0 $PERFORMANCE

for i in `seq 0 $((NUM_BUFFERS-1))`
do
  callCompareTest original_${i}.log result_${i}.log 1-${i} "Compare 1-${i}" 0 0
done

rm result_*.log

PORT=`python3 get_available_port.py`
# tensor_sink (server) --> tensor_src (client), other/tensor
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter ! tensor_sink_grpc port=${PORT} server=true" 2 0 0 $PERFORMANCE &
sleep 1
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=${NUM_BUFFERS} server=false ! 'other/tensor,dimension=(string)3:640:480,type=(string)uint8,framerate=(fraction)5/1' ! multifilesink location=result_%1d.log" 2 0 0 $PERFORMANCE

for i in `seq 0 $((NUM_BUFFERS-1))`
do
  callCompareTest original_${i}.log result_${i}.log 2-${i} "Compare 2-${i}" 0 0
done

rm result_*.log
rm original_*.log

# Dump original frames, passthrough, other/tensors
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter frames-per-tensor=2 ! multifilesink location=original_%1d.log" 3 0 0 $PERFORMANCE

PORT=`python3 get_available_port.py`
# tensor_sink (client) --> tensor_src (server), other/tensors
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=$((NUM_BUFFERS/2)) ! 'other/tensors,num_tensors=2,dimensions=(string)3:640:480.3:640:480,types=(string)uint8.uint8,framerate=(fraction)5/1' ! multifilesink location=result_%1d.log" 4 0 0 $PERFORMANCE &
sleep 1
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter frames-per-tensor=2 ! tensor_sink_grpc port=${PORT}" 4 0 0 $PERFORMANCE

for i in `seq 0 $((NUM_BUFFERS/2-1))`
do
  callCompareTest original_${i}.log result_${i}.log 4-${i} "Compare 4-${i}" 0 0
done

rm result_*.log

PORT=`python3 get_available_port.py`
# tensor_sink (server) --> tensor_src (client), other/tensors
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter frames-per-tensor=2 ! tensor_sink_grpc port=${PORT} server=true" 5 0 0 $PERFORMANCE &
sleep 1
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=$((NUM_BUFFERS/2)) server=false ! 'other/tensors,num_tensors=2,dimensions=(string)3:640:480.3:640:480,types=(string)uint8.uint8,framerate=(fraction)5/1' ! multifilesink location=result_%1d.log" 5 0 0 $PERFORMANCE

for i in `seq 0 $((NUM_BUFFERS/2-1))`
do
  callCompareTest original_${i}.log result_${i}.log 5-${i} "Compare 5-${i}" 0 0
done

rm result_*.log
rm original_*.log

report
